from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random
import h5py
import json
from collections import Counter

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from tokenizers import TAPETokenizer
from registry import registry

logger = logging.getLogger(__name__)


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.loom':
        return LOOMDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None, len_max=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    if len_max != None: shape[1] = max(shape[1], len_max)

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class FastaDataset(Dataset):
    """Creates a dataset from a fasta file.
    Args:
        data_file (Union[str, Path]): Path to fasta file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        from Bio import SeqIO
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        # if in_memory:
        cache = list(SeqIO.parse(str(data_file), 'fasta'))
        num_examples = len(cache)
        self._cache = cache
        # else:
            # records = SeqIO.index(str(data_file), 'fasta')
            # num_examples = len(records)
#
            # if num_examples < 10000:
                # logger.info("Reading full fasta file into memory because number of examples "
                            # "is very low. This loads data approximately 20x faster.")
                # in_memory = True
                # cache = list(records.values())
                # self._cache = cache
            # else:
                # self._records = records
                # self._keys = list(records.keys())

        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        # if self._in_memory and self._cache[index] is not None:
        record = self._cache[index]
        # else:
            # key = self._keys[index]
            # record = self._records[key]
            # if self._in_memory:
                # self._cache[index] = record

        item = {'id': record.id,
                'primary': str(record.seq),
                'protein_length': len(record.seq)}
        return item


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class LOOMDataset(Dataset):
    """Creates a dataset from a loom file. Assumes that the data a main matrix
    (numpy ndarray or scipy sparse matrix) and two dictionaries of row and column
    attributes (with attribute names as keys, and numpy ndarrays as values)
    Args:
        loom_file (Union[str, Path]): Path to loom file.
        in_memory (bool): Dummy variable to match API of other datasets
    """
    def __init__(self, loom_path: Union[str, Path], in_memory: bool = True):
        with h5py.File(loom_path, "r", libver='latest', swmr=True) as f:
            self.gene_name = f["row_attrs/Gene"][...].astype(np.str)
            dset = f["matrix"]
            this_shape = dset.shape
            self.gene_number = this_shape[0]
            self.cell_number = this_shape[1]
            self.cell_ids = f["col_attrs/CellID"][...].astype(np.str)
            dset = np.array(dset)
        print("completed loading loom!")
        self.library_size = np.sum(dset, 0)
        self.library_size_factor = np.median(self.library_size)
        self._matrix = np.log1p(np.divide(dset, self.library_size, out=np.zeros_like(dset), where=self.library_size!=0) * self.library_size_factor)
        self._num_examples = self.cell_number

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = {}
        primary = self._matrix[:,index]
        item["cell_id"] = self.cell_ids[index]
        item["primary"] = primary
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        return item


class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self, data_file: Union[str, Path], in_memory: bool = True):
        import loompy
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        records = json.loads(data_file.read_text())

        if not isinstance(records, list):
            raise TypeError(f"TAPE JSONDataset requires a json serialized list, "
                            f"received {type(records)}")
        self._records = records
        self._num_examples = len(records)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = self._records[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = str(index)
        return item

class NPZDataset(Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = True,
                 split_files: Optional[Collection[str]] = None):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)
        file_glob = data_file.glob('*.npz')
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.name in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory")

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = file_list

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = dict(np.load(self._file_list[index]))
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = self._file_list[index].stem
        return item


@registry.register_task('embed')
class EmbedDataset(Dataset):

    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False,
                 convert_tokens_to_ids: bool = True):
        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = dataset_factory(data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return item['id'], token_ids, input_mask

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        ids, tokens, input_mask = zip(*batch)
        ids = list(ids)
        tokens = torch.from_numpy(pad_sequences(tokens))
        input_mask = torch.from_numpy(pad_sequences(input_mask))
        return {'ids': ids, 'input_ids': tokens, 'input_mask': input_mask}  # type: ignore


@registry.register_task('wgbs_methylation_regression')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset to pretrain INTERACT by methylation data of CpGs from neuron or glia cell type
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        self.complementary_genome = np.load("./datasets/complementary_genome.npy",allow_pickle=True).item()

        self.methylation_data = []
        for chrom in split:
            print("reading\t" + str(chrom))
            data_path = Path(data_path)
            data_file = f'{chrom}.json'
            methylation_data = dataset_factory(data_path / data_file, in_memory)
            if len(self.methylation_data) == 0: self.methylation_data = methylation_data
            else: self.methylation_data = self.methylation_data + methylation_data

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        label_data = item_data["target"]
        strand = item_data["strand"]
        position = item_data["pos"]
        chrom = item_data["chrom"]
        if strand == "+":
            start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
            DNA_data = self.genome[chrom][start-1:stop-1]
        else:
            position = len(self.genome[chrom]) - position + 1
            start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
            DNA_data = self.complementary_genome[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, label_data

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, label_data = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        label_data = torch.from_numpy(np.array(label_data, dtype = np.float32))

        return {'DNA_data': DNA_data,
                'targets': label_data}


@registry.register_task('array_mQTL_regression')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset for CpGs from genome by DNA sequence with reference allel or variation allel
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.window = 2001
        self.nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        self.complementary_genome = np.load("./datasets/complementary_genome.npy",allow_pickle=True).item()

        self.methylation_data = []
        data_path = Path(data_path)
        data_file = f'{split}.json'
        methylation_data = dataset_factory(data_path / data_file, in_memory)
        if len(self.methylation_data) == 0: self.methylation_data = methylation_data
        else: self.methylation_data = self.methylation_data + methylation_data
        print(str(len(self.methylation_data)))

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        strand = item_data["strand"]
        cpg_pos = item_data["cpg_pos"]
        snp_pos = item_data["snp_pos"]
        chrom = item_data["chrom"]
        alias = item_data["alias"]
        start, stop = cpg_pos-self.window//2, cpg_pos+self.window//2+1
        if strand == "+": DNA_data = self.genome[chrom][start-1:stop-1].copy()
        else: DNA_data = self.complementary_genome[chrom][start-1:stop-1].copy()
        if len(DNA_data) == 0: DNA_data = np.zeros(self.window)
        index = self.window//2+(snp_pos-cpg_pos)
        DNA_data[index] = self.nucleotide_ind[alias]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, cpg_pos, snp_pos

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, CPG_pos, VAR_pos = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        CPG_pos = torch.from_numpy(np.array(CPG_pos, dtype = np.long))
        VAR_pos = torch.from_numpy(np.array(VAR_pos, dtype = np.long))

        return {'DNA_data': DNA_data,
                'CPG_pos': CPG_pos,
                'VAR_pos': VAR_pos}


@registry.register_task('array_methylation_regression')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset to finetune INTERACT by methylation data of CpGs from one of the 13 cell types
    #Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        self.complementary_genome = np.load("./datasets/complementary_genome.npy",allow_pickle=True).item()

        self.methylation_data = []
        if split == "train":
            for idx in range(1,21):
                print("reading\t" + "chr" + str(idx))
                data_path = Path(data_path)
                data_file = f'chr{idx}.json'
                methylation_data = dataset_factory(data_path / data_file, in_memory)
                if len(self.methylation_data) == 0: self.methylation_data = methylation_data
                else: self.methylation_data = self.methylation_data + methylation_data
        else:
            print("reading\t" + str(split))
            data_path = Path(data_path)
            data_file = f'{split}.json'
            methylation_data = dataset_factory(data_path / data_file, in_memory)
            if len(self.methylation_data) == 0: self.methylation_data = methylation_data
            else: self.methylation_data = self.methylation_data + methylation_data

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        label_data = item_data["target"]
        strand = item_data["strand"]
        position = item_data["pos"]
        chrom = item_data["chrom"]
        if strand == "+": 
            start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
            DNA_data = self.genome[chrom][start-1:stop-1]
        else:
            position = len(self.genome[chrom]) - position + 1
            start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
            DNA_data = self.complementary_genome[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, label_data

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, label_data = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        label_data = torch.from_numpy(np.array(label_data, dtype = np.float32))

        return {'DNA_data': DNA_data,
                'targets': label_data}    
 

@registry.register_task('array_methylation_prediction')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset for prediction by finetuned INTERACT model from CpGs in genome
    #Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        
        self.methylation_data = []
        print("reading\t" + str(split))
        data_path = Path(data_path)
        data_file = f'{split}.json'
        methylation_data = dataset_factory(data_path / data_file, in_memory)
        if len(self.methylation_data) == 0: self.methylation_data = methylation_data
        else: self.methylation_data = self.methylation_data + methylation_data
    
    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        position, chrom = item_data["pos"], item_data["chrom"]
        start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
        DNA_data = self.genome[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, position

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, positions = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        positions = torch.from_numpy(np.array(positions, dtype = np.long))

        return {'DNA_data': DNA_data,
                'position': positions}


@registry.register_task('cnn_methylation_regression')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset to train CNN model by methylation data of CpGs from one of the 13 cell types
    #Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        self.complementary_genome = np.load("./datasets/complementary_genome.npy",allow_pickle=True).item()

        self.methylation_data = []
        if split == "train":
            for idx in range(1,21):
                print("reading\t" + "chr" + str(idx))
                data_path = Path(data_path)
                data_file = f'chr{idx}.json'
                methylation_data = dataset_factory(data_path / data_file, in_memory)
                if len(self.methylation_data) == 0: self.methylation_data = methylation_data
                else: self.methylation_data = self.methylation_data + methylation_data
        else:
            print("reading\t" + str(split))
            data_path = Path(data_path)
            data_file = f'{split}.json'
            methylation_data = dataset_factory(data_path / data_file, in_memory)
            if len(self.methylation_data) == 0: self.methylation_data = methylation_data
            else: self.methylation_data = self.methylation_data + methylation_data

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        label_data = item_data["target"]
        strand = item_data["strand"]
        position = item_data["pos"]
        chrom = item_data["chrom"]
        if strand == "+":
            start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
            DNA_data = self.genome[chrom][start-1:stop-1]
        else:
            position = len(self.genome[chrom]) - position + 1
            start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
            DNA_data = self.complementary_genome[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, label_data

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, label_data = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        label_data = torch.from_numpy(np.array(label_data, dtype = np.float32))

        return {'DNA_data': DNA_data,
                'targets': label_data}


@registry.register_task('cnn_methylation_prediction')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset for prediction by trained CNN model from CpGs in genome
    #Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()

        self.methylation_data = []
        print("reading\t" + str(split))
        data_path = Path(data_path)
        data_file = f'{split}.json'
        methylation_data = dataset_factory(data_path / data_file, in_memory)
        if len(self.methylation_data) == 0: self.methylation_data = methylation_data
        else: self.methylation_data = self.methylation_data + methylation_data

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        position, chrom = item_data["pos"], item_data["chrom"]
        start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
        DNA_data = self.genome[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, position

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, positions = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        positions = torch.from_numpy(np.array(positions, dtype = np.long))

        return {'DNA_data': DNA_data,
                'position': positions}


@registry.register_task('DNA_motif_discovery')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset to output DNA motifs learned by finetuned INTERACT model from CpGs in genome
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        self.genome_sequence = np.load("./datasets/backup/genome.npy",allow_pickle=True).item()
        
        self.methylation_data = []
        if split == "train":
            for idx in range(1,23):
                print("reading\t" + "chr" + str(idx))
                data_path = Path(data_path)
                data_file = f'chr{idx}.json'
                methylation_data = dataset_factory(data_path / data_file, in_memory)
                if len(self.methylation_data) == 0: self.methylation_data = methylation_data
                else: self.methylation_data = self.methylation_data + methylation_data
        else:
            print("reading\t" + str(split))
            data_path = Path(data_path)
            data_file = f'{split}.json'
            methylation_data = dataset_factory(data_path / data_file, in_memory)
            if len(self.methylation_data) == 0: self.methylation_data = methylation_data
            else: self.methylation_data = self.methylation_data + methylation_data

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        strand, position, chrom = item_data["strand"], item_data["pos"], item_data["chrom"]
        start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
        DNA_data = self.genome[chrom][start-1:stop-1]
        sequence = self.genome_sequence[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, sequence, chrom, position

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, sequences, chroms, positions = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        #targets = torch.from_numpy(np.array(targets, dtype = np.float32)) 
 
        return {'DNA_data': DNA_data,
                'chrom': chroms,
                'position': positions,
                'sequence': sequences}


@registry.register_task('DNA_attention_discovery')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Dataset to output attentions learned by finetuned INTERACT model from CpGs in genome
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()

        self.genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
        self.genome_sequence = np.load("./datasets/backup/genome.npy",allow_pickle=True).item()

        self.methylation_data = []
        if split == "train":
            for idx in range(1,23):
                print("reading\t" + "chr" + str(idx))
                data_path = Path(data_path)
                data_file = f'chr{idx}.json'
                methylation_data = dataset_factory(data_path / data_file, in_memory)
                if len(self.methylation_data) == 0: self.methylation_data = methylation_data
                else: self.methylation_data = self.methylation_data + methylation_data
        else:
            print("reading\t" + str(split))
            data_path = Path(data_path)
            data_file = f'{split}.json'
            methylation_data = dataset_factory(data_path / data_file, in_memory)
            if len(self.methylation_data) == 0: self.methylation_data = methylation_data
            else: self.methylation_data = self.methylation_data + methylation_data

    def __len__(self) -> int:
        return len(self.methylation_data)

    def __getitem__(self, index):
        item_data =  self.methylation_data[index]
        label_data = item_data["target"]
        strand, position, chrom = item_data["strand"], item_data["pos"], item_data["chrom"]
        start, stop = max(0,position-1000), min(position+1000+1,len(self.genome[chrom]))
        DNA_data = self.genome[chrom][start-1:stop-1]
        sequence = self.genome_sequence[chrom][start-1:stop-1]
        DNA_data = np.array(DNA_data, dtype = np.long)
        return DNA_data, sequence, chrom, position, label_data

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        DNA_data, sequences, chroms, positions, label_data = tuple(zip(*batch))
        DNA_data = torch.from_numpy(pad_sequences(DNA_data,0))
        targets = torch.from_numpy(np.array(label_data, dtype = np.float32))

        return {'DNA_data': DNA_data,
                'chrom': chroms,
                'target': targets, 
                'position': positions}

