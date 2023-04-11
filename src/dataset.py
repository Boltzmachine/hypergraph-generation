import os
import glob
import numpy as np
import h5py
import json
import dataclasses
from collections import abc, namedtuple
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch.utils.data import random_split, default_collate
from tango.common.registrable import Registrable
import pytorch_lightning as pl

from .utils import read_obj, quantizer

from typing import Any
import logging
logger = logging.getLogger("pytorch_lightning")


class BipartiteData(Data):
    def __init__(self, edge_index1=None, edge_index2=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index1" or key == "edge_index2":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

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
        data = self.file[subset][obj]
        verts = torch.from_numpy(data['vertex'][()]).float()
        faces = torch.from_numpy(data['face'][()]).long()
        faces = torch.cat([faces, torch.zeros(self.max_n_face - faces.size(0), faces.size(1), dtype=faces.dtype)], dim=0)

        return {
            "X": verts,
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
    static_edge = torch.tensor([
        [0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0],
    ])
    
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
            "E": self.static_edge,
            "mask": torch.ones(8, dtype=torch.long),
        }
        
    @staticmethod
    def gen_verts():
        edges = torch.rand(3) * 18 + 2 # 3
        diag = torch.norm(edges)

        edges = edges / diag
        verts = torch.stack([edges/2, -edges/2], dim=1) # 3x2
        verts = torch.cartesian_prod(verts[0], verts[1], verts[2])
        return verts
    

@Dataset.register("prism")
class PrismDataset(Dataset):
    def __init__(self, length: int, min_verts: int = 3, max_verts: int = 4) -> None:
        super().__init__()
        self.length = length
        self.min_verts = min_verts
        self.max_verts = max_verts

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        n = random.randint(3, 4)

        theta = torch.linspace(0, 2 * torch.pi, n+1)[:-1]

        x = torch.cos(theta) * 0.5
        y = torch.sin(theta) * 0.5

        z = random.random() * 0.4 + 0.1

        verts1 = torch.stack([x, -torch.ones_like(x) * z, y], dim=1)
        verts2 = torch.stack([x, torch.ones_like(x) * z, y], dim=1)

        verts = torch.cat([verts1, verts2], dim=0)

        face = torch.zeros(n+2, 2 * n, dtype=torch.long)
        face[0, :n] = 1
        face[1, n:] = 1

        idx2 = (torch.tensor([0, 1, n, n+1])[None, :] + torch.arange(n)[:, None]) % (2 * n)
        idx2 = idx2.view(-1)
        idx1 = torch.arange(n).repeat_interleave(4)
        face[idx1+2, idx2] = 1

        return {
            "X": verts,
            "H": face,
        }
        

# def get_mask_collate_fn_by_max_face(max_face = None):
    
#     def mask_collate_fn(batchs):
#         max_num_nodes = max([batch['X'].size(0) for batch in batchs])
#         for batch in batchs:
#             X = batch['X']
#             pad_len = max_num_nodes - X.size(0)
#             mask = torch.zeros(max_num_nodes, dtype=torch.long)
#             mask[:X.size(0)] = 1
#             batch['mask'] = mask
#             batch['X'] = torch.cat([X, torch.zeros(pad_len, X.size(1), dtype=X.dtype)], dim=0)
#             H = batch['H']
#             H = torch.cat([H, torch.zeros(H.size(0), pad_len, dtype=H.dtype)], dim=1)
#             batch['H'] = torch.cat([H, torch.zeros(max_face - H.size(0), H.size(1), dtype=H.dtype)], dim=0)

#         return default_collate(batchs)
    
#     return mask_collate_fn
    
    
class DataModule(pl.LightningDataModule, Registrable):
    collate_fn = default_collate
    
    
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

        self.collate_fn = default_collate

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
    

@DataModule.register("prism")
class PrismDataModule(DataModule):
    def __init__(
            self, 
            batch_size: int = 32,
            min_verts: int = 3,
            max_verts: int = 8,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.min_verts = min_verts
        self.max_verts = max_verts
        
        self.train_len = 10000
        self.val_len = 100
        
        max_face = max(self.meta_info['n_faces'])
        self.collate_fn = get_mask_collate_fn_by_max_face(max_face)

    def setup(self, stage: str):
        self.train = PrismDataset(self.train_len, self.min_verts, self.max_verts)
        self.val = PrismDataset(self.val_len, self.min_verts, self.max_verts)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    @property
    def meta_info(self):
        length = self.train_len + self.val_len
        n_range = self.max_verts - self.min_verts + 1
        rtime = length // n_range

        return {
            'n_faces': [n + 2 for n in range(self.min_verts, self.max_verts+1)] * rtime,
            'n_nodes': [2 * n for n in range(self.min_verts, self.max_verts+1)] * rtime,
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
        
    def prepare_data(self) -> None:
        self.file = h5py.File(os.path.join(self.data_path, "data.h5"), 'r')
        file = self.file
        if (subset := self.subset) is not None:
            keys = [(subset, k) for k in file[subset].keys()]
        else:
            raise NotImplementedError
        self.keys = keys[:1]
        self.subset = subset
    
    def setup(self, stage: str):
        dataset = ShapenetDataset(self.file, self.keys, max(self.meta_info['n_faces']))
        val_len = int(0.1 * len(dataset))

        # from tqdm import tqdm
        # pos = 0
        # neg = 0
        # for data in tqdm(dataset):
        #     pos += data['H'].sum()
        #     neg += (1 - data['H']).sum()

        self.train, self.val = random_split(dataset, [len(dataset) - val_len, val_len])
        self.val = self.train
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=mask_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=mask_collate_fn, num_workers=self.num_workers)

    def teardown(self, stage: str):
        self.file.close()
        
    @property
    def meta_info(self):
        return json.load(open(os.path.join(self.data_path, "meta.json"), "r"))