import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_dense_batch


class VerticesMutedError(Exception):
    ...
    

def dequantize_verts(verts, n_bits=8, add_noise=False):
    """Convert quantized vertices to floats."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2**n_bits - 1
    verts = verts.float()
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
    return verts


class Quantizer:
    def __init__(self, n_bits: int = 8) -> None:
        self.n_bits = n_bits
        
        b = 0
        a = 1
        self.scale = (a - b) / (2**n_bits - 1)
        self.zeroPt = 0
    
    def quantize(self, x):
        x = x + 0.5
        x = torch.clamp(torch.round(x/self.scale + self.zeroPt), 0, 2**self.n_bits-1)
        return x.long()

    def dequantize(self, x):
        x = (x.float() - self.zeroPt) * self.scale
        x = x - 0.5
        return x
    
quantizer = Quantizer()


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


def to_dense_data(data: HeteroData):
    X, _ = to_dense_batch(data['verts'].x, batch=data['verts'].batch)
    H, _ = to_dense_batch(data['face'].x, batch=data['verts'].batch)
    return X, H
    
def to_sparse_data(X: torch.Tensor, H: torch.Tensor):
    data = HeteroData()
    data['verts'].x = X.view(-1, *X.size()[1:])
    data['verts'].batch = torch.arange(X.size(0)).repeate_interleave(X.size(1))
    data['face'].x = H.view(-1, *H.size()[1:])
    data['face'].batch = torch.arange(H.size(0)).repeate_interleave(H.size(1))
    