import os
import glob
import numpy as np
import h5py
import json
import dataclasses
from collections import abc, namedtuple

import torch
from torch_geometric.data import Data, HeteroData
from torch.utils.data import random_split, default_collate
from tango.common.registrable import Registrable
import pytorch_lightning as pl

from .utils import read_obj, quantizer

from typing import Any
import logging
logger = logging.getLogger("pytorch_lightning")

@dataclasses.dataclass
class DataContainer():
    X: torch.Tensor
    H: torch.Tensor
    
    def __iter__(self):
        yield from dataclasses.asdict(self).values()

#     def __getitem__(self, key):
#         return getattr(self, key)

#     def __len__(self):
#         return 

class Dataset(torch.utils.data.Dataset, Registrable):
    ...
    

class DataLoader(torch.utils.data.DataLoader, Registrable):
    default_implementation = "default"
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
        ) -> None:
        super().__init__(dataset, **kwargs)
DataLoader.register("default")(DataLoader)


@Dataset.register("shapenet")
class ShapenetDataset(Dataset):
    def __init__(self, file, keys, max_n_face):
        super().__init__()
        self.file = file
        self.keys = keys
        self.max_n_face = max_n_face
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        subset, obj = self.keys[idx]
        data = self.file[obj]
        verts = torch.from_numpy(data['vertex'][()])
        faces = torch.from_numpy(data['face'][()])
        faces = torch.cat([faces, torch.zeros(self.max_n_face - faces.size(0), faces.size(1), dtype=faces.dtype)], dim=0)

        return {
            "X": quantizer.quantize(verts),
            "H": faces
        }


@Dataset.register("cuboid")
class CuboidDataset(Dataset):
    static_hyperedge = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ], dtype=torch.long)
    
    def __init__(self, length: int):
        super().__init__()
        self.length = length
        assert self.static_hyperedge.size() == torch.Size([6, 8])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        verts = CuboidDataset.gen_verts()

        return {
            "X": verts,
            "H": self.static_hyperedge,
            "mask": torch.ones(8, dtype=torch.long),
        }
        
    @staticmethod
    def gen_verts():
        edges = torch.rand(3) * 18 + 2 # 3
        diag = torch.norm(edges)

        edges = edges / diag
        verts = torch.stack([edges/2, -edges/2], dim=1) # 3x2
        verts = torch.cartesian_prod(verts[0], verts[1], verts[2])
        return quantizer.quantize(verts)


def mask_collate_fn(batchs):
    max_num_nodes = max([batch['X'].size(0) for batch in batchs])
    for batch in batchs:
        X = batch['X']
        pad_len = max_num_nodes - X.size(0)
        mask = torch.zeros(max_num_nodes, dtype=torch.long)
        mask[:X.size(0)] = 1
        batch['mask'] = mask
        batch['X'] = torch.cat([X, torch.zeros(pad_len, X.size(1), dtype=X.dtype)], dim=0)
        batch['X'].masked_fill(~mask.bool(), -100)
        H = batch['H']
        batch['H'] = torch.cat([H, torch.zeros(H.size(0), pad_len, dtype=H.dtype)], dim=1)


    return default_collate(batchs)
    
    
class DataModule(pl.LightningDataModule, Registrable):
    ...
    
    
@DataModule.register("cuboid")
class CuboidDataModule(DataModule):
    def __init__(
            self, 
            batch_size: int = 32
        ):
        super().__init__()
        self.batch_size = batch_size
        
        self.train_len = 10000
        self.val_len = 100

    def setup(self, stage: str):
        self.train = CuboidDataset(self.train_len)
        self.val = CuboidDataset(self.val_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
    
    @property
    def meta_info(self):
        length = self.train_len + self.val_len
        return {
            'n_faces': [6] * length,
            'n_nodes': [8] * length,
        }
    

@DataModule.register("shapenet")
class ShapenetDataModule(DataModule):
    def __init__(
        self,
        data_path: str,
        subset: str,
        batch_size: int,
        num_workers: int = 0
    ):
        super().__init__()
        self.data_path = data_path
        self.subset = subset
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        self.file = h5py.File(os.path.join(self.data_path, "data.h5"), 'r')
        file = self.file
        if (subset := self.subset) is not None:
            file = file[subset]
            keys = [(subset, k) for k in file.keys()]
        else:
            raise NotImplementedError
        self.keys = keys
        
        dataset = ShapenetDataset(file, keys, max(self.meta_info['n_faces']))
        val_len = int(0.1 * len(dataset))

        # from tqdm import tqdm
        # num_faces = [data['H'].shape[0] for data in tqdm(dataset)]

        self.train, self.val = random_split(dataset, [len(dataset) - val_len, val_len])
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=mask_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=mask_collate_fn, num_workers=self.num_workers)

    def teardown(self, stage: str):
        self.file.close()
        
    @property
    def meta_info(self):
        return json.load(open(os.path.join(self.data_path, "meta.json"), "r"))