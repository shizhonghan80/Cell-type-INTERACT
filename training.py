import typing
import os
import logging
import numpy as np
from timeit import default_timer as timer
import json
from pathlib import Path
import inspect
import pickle as pkl
import loompy
import h5py
import fcntl
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from optimization import WarmupLinearSchedule
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

import utils
import errors
import visualization
from registry import registry
from models.modeling_utils import ProteinModel
try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    from apex.parallel.distributed import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False
logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]


class ForwardRunner:

    def __init__(self,
                 model: ProteinModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.local_rank = local_rank

        forward_arg_keys = inspect.getfullargspec(model.forward).args
        forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        self._forward_arg_keys = forward_arg_keys
        #assert 'methylation_data' in self._forward_arg_keys

    def initialize_distributed_model(self):
        if self.local_rank != -1:
            if not self.fp16:
                self.model = DDP(self.model)
            else:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (0,))
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False,
                no_loss: bool = False):
        # Filter out batch items that aren't used in this model
        # Requires that dataset keys match the forward args of the model
        # Useful if some elements of the data are only used by certain models
        # e.g. PSSMs / MSAs and other evolutionary data
        batch = {name: tensor for name, tensor in batch.items()
                 if name in self._forward_arg_keys}
        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}

        outputs = self.model(**batch)

        if no_loss:
            return outputs

        if isinstance(outputs[0], tuple):
            # model also returned metrics
            loss, metrics = outputs[0]
        else:
            # no metrics
            loss = outputs[0]
            metrics = {}

        if self.n_gpu > 1:  # pytorch DataDistributed doesn't mean scalars
            loss = loss.mean()
            metrics = {name: metric.mean() if isinstance(metric, int)==False else metric for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: ProteinModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 0,
                 num_train_optimization_steps: int = 1000000):

        super().__init__(model, device, n_gpu, fp16, local_rank)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._delay_accumulation = fp16 and local_rank != -1

        self.scheduler = WarmupLinearSchedule(
            self.optimizer, warmup_steps, num_train_optimization_steps)

    def initialize_fp16(self):
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O1")#, loss_scale="dynamic", master_weights=True)
            _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    def resume_from_checkpoint(self, checkpoint_dir: str) -> int:
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.fp16:
            self.optimizer._lazy_init_maybe_master_weights()
            self.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(
                    amp.master_params(self.optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch

    def save_state(self, save_directory: typing.Union[str, Path], epoch_id: int):
        save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir()
        else:
            assert save_directory.is_dir(), "Save path should be a directory"
        model_to_save = getattr(self.model, 'module', self.model)
        model_to_save.save_pretrained(save_directory)
        optimizer_state: typing.Dict[str, typing.Any] = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch_id}
        if APEX_FOUND:
            optimizer_state['master params'] = list(amp.master_params(self.optimizer))
            try:
                optimizer_state['amp'] = amp.state_dict()
            except AttributeError:
                pass
        torch.save(optimizer_state, save_directory / 'checkpoint.bin')

    def backward(self, loss) -> None:
        if not self._delay_accumulation:
            loss = loss / self.gradient_accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer,
                                delay_overflow_check=self._delay_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self._local_rank == -1:
            self._step()
        elif not self.fp16:
            # TODO: Can you do this allreduce after accumulation also?
            self._step()
        else:
            self._step_distributed_fp16()

    def _step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else \
            # torch.float32
        allreduce_dtype = torch.float16
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (
                torch.distributed.get_world_size() * self.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [allreduced_views, master_grads],
            1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = self._overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            self._step()
        else:
            # Overflow detected, print message and clear gradients
            logger.info(f"Gradient overflow.  Skipping step, reducing loss scale to "
                        f"{scaler.loss_scale()}")
            if _amp_state.opt_properties.master_weights:
                for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in self.model.parameters():
            param.grad = None

    @property
    def global_step(self) -> int:
        return self._global_step

def pearson(first_set,second_set):
    first_mean = np.mean(first_set)
    second_mean = np.mean(second_set)
    first_std = np.std(first_set)
    second_std = np.std(second_set)
    cov = np.mean((first_set - first_mean)*(second_set - second_mean))
    corr = cov / (first_std * second_std)
    return corr

def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1) -> LossAndMetrics:
    if viz is None:
        viz = visualization.DummyVisualizer()
    smoothing = 1 - 1 / num_log_iter
    accumulator = utils.MetricsAccumulator(smoothing)

    torch.set_grad_enabled(True)
    runner.train()

    def make_log_str(step: int, time: float) -> str:
        ep_percent = epoch_id + step / len(train_loader)
        if runner.scheduler is not None:
            curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
        else:
            curr_lr = runner.optimizer.param_groups[0]['lr']

        print_str = []
        print_str.append(f"[Ep: {ep_percent:.2f}]")
        print_str.append(f"[Iter: {runner.global_step}]")
        print_str.append(f"[Time: {time:5.2f}s]")
        print_str.append(f"[Loss: {accumulator.loss():.4g}]")

        for name, value in accumulator.metrics().items():
            print_str.append(f"[{name.capitalize()}: {value:.4g}]")

        print_str.append(f"[LR: {curr_lr:.4g}]")
        return ''.join(print_str)

    start_t = timer()
    for step, batch in enumerate(train_loader):
        loss, metrics = runner.forward(batch)  # type: ignore
        runner.backward(loss)
        accumulator.update(loss, metrics, step=False)
        if (step + 1) % gradient_accumulation_steps == 0:
            runner.step()
            viz.log_metrics(accumulator.step(), "train", runner.global_step)
            if runner.global_step % num_log_iter == 0:
                end_t = timer()
                logger.info(make_log_str(step, end_t - start_t))
                start_t = end_t

    final_print_str = f"Train: [Loss: {accumulator.final_loss():.4g}]"
    for name, value in accumulator.final_metrics().items():
        final_print_str += f"[{name.capitalize()}: {value:.4g}]"
    logger.info(final_print_str)
    return accumulator.final_loss(), accumulator.final_metrics()


def aupr_and_roc(preds,labels):
    precision,recall,thresholds = precision_recall_curve(labels,preds)
    aupr = auc(recall,precision)
    roc = roc_auc_score(labels,preds)
    return aupr,roc


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    is_master: bool = True) -> typing.Tuple[float, typing.Dict[str, float]]:
    num_batches = len(valid_loader)
    accumulator = utils.MetricsAccumulator()

    torch.set_grad_enabled(False)
    runner.eval()
    
    save_outputs = []
    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        accumulator.update(loss, metrics)
        predictions = outputs[1].squeeze(-1).cpu().numpy()
        targets = batch['targets'].cpu().numpy()
        for idx in range(len(targets)):
            pred, target = predictions[idx], targets[idx]
            save_outputs.append({'prediction': 1-pred, 'target': 1-target})
    # Reduce loss across ll processes if multiprocessing
    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}
     
    pred_values, target_labels = [], []
    for item in save_outputs:
        pred, target = item['prediction'], item['target']
        pred_values.append(pred)
        target_labels.append(target)
    AUPR, ROC = aupr_and_roc(pred_values, target_labels)
    metrics["AUPR"] = AUPR
    metrics["ROC"] = ROC
    
    print_str = f"Evaluation: [Loss: {eval_loss:.4g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.4g}]"

    metrics['loss'] = eval_loss
    if viz is not None:
        viz.log_metrics(metrics, "val", getattr(runner, 'global_step', epoch_id)) 
    logger.info(print_str)

    return eval_loss, metrics


def _get_outputs_to_save(batch, outputs):
    targets = batch['targets'].cpu().numpy()
    outputs = outputs.cpu().numpy()
    protein_length = batch['protein_length'].sum(1).cpu().numpy()

    reshaped_output = []
    for target, output, plength in zip(targets, outputs, protein_length):
        output_slices = tuple(slice(1, plength - 1) if dim == protein_length.max() else
                              slice(0, dim) for dim in output.shape)
        output = output[output_slices]
        target = target[output_slices]

        reshaped_output.append((target, output))
    reshaped_output


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   is_master: bool = True) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()

    save_outputs = []
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        targets = batch['targets'].clone()
        gene_ids = batch['input_genes']
        valid_mask = (targets != -1)
        predictions = outputs[1]
        predictions = predictions[valid_mask].squeeze(-1)
        targets = targets[valid_mask]
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        for idx in range(len(targets)):            
            pred, target, gene_id = predictions[idx], targets[idx], gene_ids[idx]
            save_outputs.append({'prediction': pred[gene_id], 'target': target[gene_id]})
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}
    return save_outputs,metrics


def run_predict_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   output_dir: str = './results',
                   is_master: bool = True,
                   split: str = 'test',) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    accumulator = utils.MetricsAccumulator()

    save_outputs = []
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        accumulator.update(loss, metrics)
        predictions = outputs[1].squeeze(-1).cpu().numpy()
        targets = batch['targets'].cpu().numpy()
        for idx in range(len(targets)):
            pred, target = predictions[idx], targets[idx]
            save_outputs.append({'prediction': 1-pred, 'target': 1-target})
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}
    test_loss = utils.reduce_scalar(accumulator.final_loss())
     
    pred_values, target_labels = [], []
    for item in save_outputs:
        pred, target = item['prediction'], item['target']
        pred_values.append(pred)
        target_labels.append(target)
    AUPR, ROC = aupr_and_roc(pred_values, target_labels)
    metrics["AUPR"] = AUPR
    metrics["ROC"] = ROC
    
    print_str = f"Test: [Loss: {test_loss:.4g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.4g}]"
    logger.info(print_str)

    return save_outputs, metrics


def run_reference_predict(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   output_dir: str = './results',
                   is_master: bool = True,
                   split: str = 'test',) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    accumulator = utils.MetricsAccumulator()

    save_outputs, methylation_data = [], []
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        outputs = runner.forward(batch, return_outputs=True, no_loss=True)  # type: ignore
        positions = batch['position'].cpu().numpy()
        predictions = outputs[0].squeeze(-1).cpu().numpy()
        for idx in range(len(positions)):
            prediction, position = predictions[idx], positions[idx]
            save_outputs.append({'prediction': prediction, 'position': position})
            methylation_data.append([position,prediction])
    methylation_data = sorted(methylation_data, key=lambda item: item[0])
    output = open(output_dir + "/" + split + ".txt", "w")
    for item in methylation_data: output.write(split + "\t" + str(item[0]) + "\t" + str(item[1]) + "\n")
    output.close()


def run_expression_predict(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   output_dir: str = './results',
                   is_master: bool = True,
                   split: str = 'test',) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    accumulator = utils.MetricsAccumulator()

    save_outputs = []
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        outputs = runner.forward(batch, return_outputs=True, no_loss=True)  # type: ignore
        positions = batch['position'].cpu().numpy()
        chroms = batch['chrom']
        predictions = outputs[0].squeeze(-1).cpu().numpy()
        for idx in range(len(positions)):
            prediction, position, chrom = predictions[idx], positions[idx], chroms[idx]
            save_outputs.append({'prediction': prediction.tolist(), 'position': position, 'chrom': chrom})

    output = open(output_dir + "/" + split + ".txt", "w")
    for item in save_outputs:
        chrom, position, prediction = item['chrom'], item['position'], item['prediction']
        prediction = [str(i) for i in prediction]
        line = ",".join(prediction)
        output.write(chrom + "\t" + str(position) + "\t" + line + "\n") 
    output.close()


def run_mQTL_predict(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   output_dir: str = './results',
                   is_master: bool = True,
                   split: str = 'test',) -> typing.List[typing.Dict[str, typing.Any]]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.set_grad_enabled(False)
    runner.eval()

    save_outputs = []
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        outputs = runner.forward(batch, return_outputs=True, no_loss=True)  # type: ignore
        cpg_positions = batch['CPG_pos']
        snp_positions = batch['VAR_pos']
        predictions = outputs[0].squeeze(-1).cpu().numpy()
        for idx in range(len(predictions)):
            cpg_pos, snp_pos, pred = cpg_positions[idx], snp_positions[idx], predictions[idx]
            save_outputs.append({'cpg_position': cpg_pos, 'snp_position': snp_pos, 'prediction': pred})
    
    gpu_idx = str(torch.cuda.current_device())
    output = open(output_dir+"/"+split+"_"+str(gpu_idx)+".txt","w")
    for item in save_outputs:
        cpg_position, snp_position, pred = item['cpg_position'], item['snp_position'], item['prediction']
        line = str(cpg_position.data.cpu().numpy()) + "\t" + str(snp_position.data.cpu().numpy()) + "\t" + str(pred)
        output.write(line + "\n")
    output.close()
    return save_outputs


def run_DNA_motif(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   output_dir: str = './results',
                   is_master: bool = True,
                   split: str = 'test',) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    final_motif, final_weight, save_outputs, num_batch = [], [], {}, 0
         
    target_file = output_dir + "/max_activation.npz"
    max_activation = np.load(target_file)["max_activation"]
    for motif_idx in range(0,512):
        if max_activation[motif_idx] == 0: continue
        output = open(output_dir + "/motif_"+str(motif_idx)+".txt", "w")
        output.write("chrom\tposition\tsequence\tmatch_score\tmax_score\ttarget\n")
    
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        outputs = runner.forward(batch, return_outputs=True, no_loss=True)  # type: ignore
        
        #targets = batch['targets'].cpu().numpy()     
        sequences = batch['sequence']
        chroms = batch['chrom']
        positions = batch['position']
        motifs = outputs[0].squeeze(-1).cpu().numpy()
        targets = outputs[1].squeeze(-1).cpu().numpy()
        motif_weight, num_batch = list(motifs), num_batch + 1
        for motif_idx in range(0,512):
            max_score = max_activation[motif_idx]
            if max_activation[motif_idx] == 0: continue
            if "motif_"+str(motif_idx) not in save_outputs.keys(): save_outputs["motif_"+str(motif_idx)] = []
            for seq_idx in range(0,len(motif_weight)):
                sequence = sequences[seq_idx]
                position = positions[seq_idx]
                chrom = chroms[seq_idx]
                target = targets[seq_idx]
                pos_idx = np.argmax(motif_weight[seq_idx][motif_idx])
                match_score = motif_weight[seq_idx][motif_idx][pos_idx]
                pos_idx = pos_idx - 4
                if pos_idx < 0 or (pos_idx + 10) > 2000: continue
                motif_sequence = sequence[pos_idx:pos_idx+10]
                save_outputs["motif_"+str(motif_idx)].append([chrom, position, motif_sequence, match_score, max_score, target])
        if num_batch % 200 != 0: continue
        for motif_name in save_outputs.keys():
            output = open(output_dir + "/"+str(motif_name)+".txt","a")
            for item in save_outputs[motif_name]:
                chrom,position,sequence,match_score,max_score,target = item[0],item[1],item[2],item[3],item[4],item[5]
                output.write(chrom+"\t"+str(position)+"\t"+sequence+"\t"+str(match_score)+"\t"+str(max_score)+"\t"+str(target)+"\n")
        save_outputs = {}
    for motif_name in save_outputs.keys():
        output = open(output_dir + "/"+str(motif_name)+".txt","a")
        for item in save_outputs[motif_name]:
            chrom,position,sequence,match_score,max_score,target = item[0],item[1],item[2],item[3],item[4],item[5]
            output.write(chrom+"\t"+str(position)+"\t"+sequence+"\t"+str(match_score)+"\t"+str(max_score)+"\t"+str(target)+"\n")
        output.close()
    """
        max_weight = torch.max(outputs[0], axis=2)[0]
        max_weight = torch.max(max_weight, axis=0)[0]
        final_weight.append(max_weight.data.cpu().numpy())
    
    final_weight = np.array(final_weight)
    final_weight = np.max(final_weight,axis=0)
    print(final_weight)
    print(np.sum(final_weight>0))
    target_file = output_dir + "/max_activation"
    np.savez_compressed(target_file, max_activation=final_weight)
    """

def run_DNA_attention(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   output_dir: str = './results',
                   is_master: bool = True,
                   split: str = 'test',) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    positive_attentions, negative_attentions, save_outputs, num_batch = {}, {}, {}, 0
    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        outputs = runner.forward(batch, return_outputs=True, no_loss=True)  # type: ignore
        chroms = batch['chrom']
        targets = batch['target'].cpu().numpy()
        positions = batch['position']
        attentions = outputs[0][0].cpu().numpy()
        for idx in range(0,len(positions)):
            chrom, position, target = chroms[idx], positions[idx], targets[idx]
            if target == 1: positive_attentions[chrom+"_"+str(position)] = attentions[idx]
            else: negative_attentions[chrom+"_"+str(position)] = attentions[idx]
    output =  output_dir + "/positive_attention"
    np.save(output, positive_attentions, allow_pickle=True)
    output =  output_dir + "/negative_attention"
    np.save(output, negative_attentions, allow_pickle=True)
    return positive_attentions, negative_attentions
    


def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              split: str = 'test',
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir)

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)
    train_dataset = utils.setup_dataset(task, data_dir, "train", tokenizer)#train
    valid_dataset = utils.setup_dataset(task, data_dir, "chr21", tokenizer)
    test_dataset = utils.setup_dataset(task, data_dir, "chr22", tokenizer)
    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)
    test_loader = utils.setup_loader(
        test_dataset, batch_size, local_rank, n_gpu,
        1, num_workers) 

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    num_evals_no_improvement = 0

    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_master:
            return False
        if isinstance(save_freq, int):
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    #save_outputs, metrics = run_predict_epoch(test_loader, runner, output_dir, is_master, split)
    # ACTUAL TRAIN/EVAL LOOP #
    best_performance = 0.0
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):
            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps)
            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                val_loss, valid_metrics = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                save_outputs, test_metrics = run_predict_epoch(test_loader, runner, output_dir, is_master, split)  
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_evals_no_improvement = 0
                else:
                    num_evals_no_improvement += 1

            # Save trained model
            #Strength = valid_metrics["Specificity"]
            Strength = (valid_metrics["Sensitivity"] + valid_metrics["Specificity"]) / 2
            #Strength = valid_metrics["AUPR"]
            if do_save(epoch_id, num_evals_no_improvement) and Strength > best_performance:
                best_performance = Strength
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id)
                logger.info(f"Saving model checkpoint to {save_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
                logger.log(35, f"Best Val Loss: {best_val_loss}")
                if local_rank != -1: raise errors.EarlyStopping
                else: break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
    if best_val_loss != float('inf'):
        logger.log(35, f"Best Val Loss: {best_val_loss}")


def run_pretrain(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              split: str = 'test',
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir)

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    valid_dataset = utils.setup_dataset(task, data_dir, ['chr21'], tokenizer)
    #test_dataset = utils.setup_dataset(task, data_dir, ['chr22'], tokenizer)
    valid_loader = utils.setup_loader(valid_dataset, batch_size, local_rank, n_gpu,
                          gradient_accumulation_steps, num_workers)
    #test_loader = utils.setup_loader(test_dataset, batch_size, local_rank, n_gpu,
    #                      gradient_accumulation_steps, num_workers)
    num_train_optimization_steps = 500000

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_val_loss = float('inf')
    num_evals_no_improvement = 0

    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_master:
            return False
        if isinstance(save_freq, int):
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    #val_loss, valid_metrics = run_valid_epoch(0, valid_loader, runner, viz, is_master)
    #save_outputs, metrics = run_predict_epoch(test_loader, runner, output_dir, is_master, split)
    # ACTUAL TRAIN/EVAL LOOP #
    best_performance = 0.0
    chroms = [["chr1","chr20"],["chr2","chr19"],["chr3","chr18"],["chr4","chr17"],["chr5","chr16"],
            ["chr6","chr15"],["chr7","chr14"],["chr8","chr13"],["chr9","chr12"],["chr10","chr11"]]
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):
            for idx in range(0, len(chroms)):
                train_dataset = utils.setup_dataset(task, data_dir, chroms[idx], tokenizer)#train
                train_loader = utils.setup_loader(train_dataset, batch_size, local_rank, n_gpu,
                                      gradient_accumulation_steps, num_workers)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(train_dataset))
                logger.info("  Batch size = %d", batch_size)
                logger.info("  Num epochs = %d", num_train_epochs)
                logger.info("  Num train steps = %d", num_train_optimization_steps)
                logger.info("  Num parameters = %d", num_trainable_parameters)

                run_train_epoch(epoch_id, train_loader, runner, viz, num_log_iter, gradient_accumulation_steps)
                if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                    val_loss, valid_metrics = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                    #save_outputs, test_metrics = run_predict_epoch(test_loader, runner, output_dir, is_master, split)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        num_evals_no_improvement = 0
                    else:
                        num_evals_no_improvement += 1

                # Save trained model
                Strength = valid_metrics["Pearson"]
                if do_save(epoch_id, num_evals_no_improvement) and Strength > best_performance:
                    best_performance = Strength
                    logger.info("** ** * Saving trained model ** ** * ")
                    # Only save the model itself
                    runner.save_state(save_path, epoch_id)
                    logger.info(f"Saving model checkpoint to {save_path}")

                utils.barrier_if_distributed()
                if patience > 0 and num_evals_no_improvement >= patience:
                    logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
                    logger.log(35, f"Best Val Loss: {best_val_loss}")
                    if local_rank != -1: raise errors.EarlyStopping
                    else: break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
    if best_val_loss != float('inf'):
        logger.log(35, f"Best Val Loss: {best_val_loss}")


def run_eval(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              split: str = 'test',
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:

    from captum.attr import (
        GradientShap,
        DeepLift,
        DeepLiftShap,
        IntegratedGradients,
        LayerConductance,
        NeuronConductance,
        NoiseTunnel)

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir)
    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir)

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)
    test_dataset = utils.setup_dataset(task, data_dir, split, tokenizer)
    test_loader = utils.setup_loader(test_dataset, batch_size, local_rank, n_gpu, 1, num_workers)
    num_train_optimization_steps = utils.get_num_train_optimization_steps(
                                   test_dataset, batch_size, num_train_epochs)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)
    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(model, optimizer, gradient_accumulation_steps, device, n_gpu,
             fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()

    is_master = local_rank in (-1, 0)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    utils.barrier_if_distributed()
    run_predict_epoch(test_loader, runner, output_dir, is_master, split)    
 

def run_predict(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              split: str = 'test',
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)
    test_dataset = utils.setup_dataset(task, data_dir, split, tokenizer)
    test_loader = utils.setup_loader(
        test_dataset, batch_size, local_rank, n_gpu,
        1, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        test_dataset, batch_size, num_train_epochs)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    utils.barrier_if_distributed()
    run_mQTL_predict(test_loader, runner, output_dir, is_master, split)
    #run_reference_predict(test_loader, runner, output_dir, is_master, split)
    

def run_motif(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              split: str = 'test',
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)
    test_dataset = utils.setup_dataset(task, data_dir, "chr22", tokenizer)#
    test_loader = utils.setup_loader(
        test_dataset, batch_size, local_rank, n_gpu, 1, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        test_dataset, batch_size, num_train_epochs)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        test_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    utils.barrier_if_distributed()
    run_DNA_attention(test_loader, runner, output_dir, is_master, split)
    #run_DNA_motif(test_loader, runner, output_dir, is_master, split)
