#
This repository contains codes to predict DNA methylation regulatory variants in specific brain cell types.

# Hardware requirements
scMeformer package requires only a standard computer with GPUs and enough RAM to support the in-memory operations.


# Software requirements
## OS Requirements
This package is supported by Linux. The package has been tested on Rocky Linux 9.2.

## Python Dependencies
scMeformer mainly depends on the following Python packages. <br/>
PyTorch <br/>
apex <br/>
numpy <br/>
scipy <br/>
scikit-learn <br/>
pandas <br/>
loompy <br/>
json <br/>
h5py

# Usage

## 1. Pretraining

### 1.1. Calculate DNAm levels for CpG sites in pseudo-bulk of neuronal or glial cells.

### Example
```bash
Construct training data and validation data for neuron to prerain INTERACT model.

$python run_feature.py neuorn

```

### Example
```bash
Constructs training data and validation data for glia to prerain INTERACT model.

$python run_feature.py glia

```

### 1.2 Pretrain INTERACT model

### Example
```bash
pretrain INTERACT model with methylation data from neuron using four GPUs

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer wgbs_methylation_regression \
        --exp_name wgbs_methylation_regression \
        --learning_rate 0.000176 \
        --batch_size 128 \
        --data_dir ./datasets/small_eval/neuron \
        --output_dir ./outputs/merge_eval/neuron \
        --warmup_steps 10000 \
        --gradient_accumulation_steps 1 \
        --fp16 --local_rank 0 \
        --nproc_per_node 4 \
        --model_config_file ./config/config.json
```

### Example
```bash
Pretrain INTERACT model with methylation data from glia using four GPUs

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer wgbs_methylation_regression \
        --exp_name wgbs_methylation_regression \
        --learning_rate 0.000176 \
        --batch_size 128 \
        --data_dir ./datasets/small_eval/glia \
        --output_dir ./outputs/merge_eval/glia \
        --warmup_steps 10000 \
        --gradient_accumulation_steps 1 \
        --fp16 --local_rank 0 \
        --nproc_per_node 4 \
        --model_config_file ./config/config.json
```


## 2. Training

### 2.1. Calculate DNAm levels for CpG sites in pseudo-bulk of each specific cell type.

### Example
```bash
Construct training data, validation data and test data for L23 to finetune INTERACT model.

$python run_feature.py L23

```

### 2.2. Finetune pre-trained INTERACT models for each cell type. 

### Example
```bash
finetune the INTERACT model for L23 using four GPUs from the pretrained neuron model

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer array_methylation_regression \
        --exp_name array_methylation_regression \
        --learning_rate 0.000176 \
        --batch_size 128 \
        --data_dir ./datasets/small_eval/L23 \
        --output_dir ./outputs/merge_eval/L23 \
        --warmup_steps 10000 \
        --gradient_accumulation_steps 1 \
        --fp16 --local_rank 0 \
        --nproc_per_node 4 \
        --model_config_file ./config/config.json
	--from_pretrained ./outputs/merge_eval/neuron
```

### Example
```bash
finetune the INTERACT model for Astro using four GPUs from the pretrained glia model

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer array_methylation_regression \
        --exp_name array_methylation_regression \
        --learning_rate 0.000176 \
        --batch_size 128 \
        --data_dir ./datasets/small_eval/Astro \
        --output_dir ./outputs/merge_eval/Astro \
        --warmup_steps 10000 \
        --gradient_accumulation_steps 1 \
        --fp16 --local_rank 0 \
        --nproc_per_node 4 \
        --model_config_file ./config/config.json
        --from_pretrained ./outputs/merge_eval/glia
```

## 3. Prediction

### 3.1. Predict DNAm levels of CpG sites from DNA sequences with reference allele using one GPU.
### Example
```bash
predict DNAm levels of CpG sites from DNA sequences with reference allel for L23 using the finetuned INTERACT model

CUDA_VISIBLE_DEVICES=0 python3 main.py transformer array_mQTL_regression \
	--exp_name array_mQTL_regression \
	--batch_size 2048 \
	--num_workers 2 \
	--learning_rate 0.000176 \
	--warmup_steps 20000 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/double_genome/positive_strand/reference \
	--output_dir ./outputs/merge_genome/Astro/reference/chr1 \
	--num_train_epochs 1 \
	--from_pretrained ./outputs/merge_eval/L23
	--split chr1
```

### Example
```bash
predict DNAm levels of CpG sites from DNA sequences with variation allel for L23 using the finetuned INTERACT model

CUDA_VISIBLE_DEVICES=0 python3 main.py transformer array_mQTL_regression \
	--exp_name array_mQTL_regression \
	--batch_size 2048 \
	--num_workers 2 \
	--learning_rate 0.000176 \
	--warmup_steps 20000 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/double_genome/positive_strand/variation/ \
	--output_dir ./outputs/merge_genome/Astro/variation/chr1 \
	--num_train_epochs 1 \
	--from_pretrained ./outputs/merge_eval/Astro \
	--split chr1
```

### 3.2. Calculate absolute difference of DNAm levels between the two DNA sequences with reference and alternative alleles.
### Example
```bash
Calculates the absolute DNAm difference for CpGs in chromsome 1 for L23

python run_snmQTL.py L23 chr1
