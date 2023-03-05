import os
import glob
import numpy as np
import dataclasses
from collections import abc, namedtuple

import torch
from torch_geometric.data import Data, HeteroData
from tango.common.registrable import Registrable

from .utils import read_obj, quantizer

from typing import Any

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
    

@Dataset.register("shapenet")
class ShapenetDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = self._read_data(path)
        
    def _read_data(self, path):
        obj_paths = glob.glob(os.path.join(path, "**", "*.obj"), recursive=True)

        data = []
        for obj_path in obj_paths:
            vertices, faces = read_obj(obj_path)
            data.append({
                "vertices": vertices,
                "faces": faces
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@Dataset.register("cuboid")
class CuboidDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.static_hyperedge = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ], dtype=torch.long)
        assert self.static_hyperedge.size() == torch.Size([6, 8])

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        verts = CuboidDataset.gen_verts()

        return {
            "X": verts,
            "H": self.static_hyperedge
        }
        
    @staticmethod
    def gen_verts():
        edges = torch.rand(3) * 18 + 2 # 3
        diag = torch.norm(edges)

        edges = edges / diag
        verts = torch.stack([edges/2, -edges/2], dim=1) # 3x2
        verts = torch.cartesian_prod(verts[0], verts[1], verts[2])
        return quantizer.quantize(verts)