import os
import glob
import numpy as np
import dataclasses

import torch
from tango.common.registrable import Registrable


def read_obj(path):
    vertices = []
    faces = []
    with open(path, 'r') as file:
        for line in file:
            if line == "":
                continue
            split = line.split()
            if len(split) > 0:
                if split[0] == 'v':
                    _, x, y, z = line.split()
                    # x, y, z = map(float, (x, y, z))
                    vertices.append((x, y, z))
                elif split[0] == 'f':
                    vs = split[1:]
                    vs = list(map(lambda v: v.split('/')[0], vs))
                    faces.append(vs)
                else:
                    pass
                    # print("not vertex or face, skip line:", line[:-1])
            else:
                pass
                # print("skip line: ", line[-1])
    return vertices, faces


@dataclasses.dataclass
class DataContainer:
    X: torch.Tensor
    H: torch.Tensor
    
    def __iter__(self):
        yield from dataclasses.asdict(self).values()


class Dataset(torch.utils.data.Dataset, Registrable):
    """
    Registrable dataset
    """

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

        self.static_hyperedge = None

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        edges = torch.randint(2, 20, size=3) # 3
        diag = torch.norm(edges)

        edges = edges / diag
        verts = torch.stack([edges/2, -edges/2], dim=1) # 3x2
        verts = torch.cartesian_prod(verts[0], verts[1], verts[2])
        import ipdb; ipdb.set_trace()

        return