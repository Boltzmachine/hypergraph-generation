import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_dense_batch
from torchvision.transforms.functional import to_pil_image

from PIL import Image

class VerticesMutedError(Exception):
    ...


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
        assert x.dtype == torch.long
        x = (x.float() - self.zeroPt) * self.scale
        x = x - 0.5
        return x
    
    def quantize2(self, x):
        return self.dequantize(self.quantize(x))
    
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


def masked_select_X(X, mask):
    """
    X - [bs, n_nodes, 3] or [bs, n_nodes, 3, 256]
    mask - [bs, n_nodes]
    """
    mask = mask.bool()
    if X.dim() == 3:
        return torch.masked_select(X, mask[..., None]).view(-1, X.size(-1))
    elif X.dim() == 4:
        return torch.masked_select(X, mask[..., None, None]).view(-1, X.size(-2), X.size(-1)).transpose(1, -1)
    else:
        raise ValueError
    
    
def masked_select_H(H, mask):
    """
    H - [bs, n_hyper, n_nodes]
    mask - [bs, n_nodes]
    
    return H - [bs * n_nodes, n_hyper]
    """
    assert H.dim() == 3
    mask = mask.bool()
    return torch.masked_select(H.transpose(1, 2), mask[..., None]).view(-1, H.size(1))


def prepare_for_loss_and_metrics(X, batch_X, H, batch_H, mask):
    """
    mask out X and H, flatten all of them
    X: [bs, n_nodes, 3, 256]
    batch_X: [bs, n_nodes, 3]
    H: [bs, n_hyper, n_nodes]
    batch_H: [bs, n_hyper_n_nodes]
    mask: [bs, n_nodes]
    """
    X = masked_select_X(X, mask)
    batch_X = masked_select_X(batch_X, mask)
    H = masked_select_H(H, mask)
    batch_H = masked_select_H(batch_H, mask)
    assert X.size(-1) == batch_X.size(-1) == 3
    return X, batch_X, H, batch_H

def create_mask_from_length(length: torch.Tensor):
    mask = torch.arange(length.max(), device=length.device)[None, :] < length[:, None]
    return mask.long()


def make_gif(frames, path):
    frames = [to_pil_image(frame) for frame in frames[::5]]
    frames = frames + [frames[-1]] * 50
    frame_one = frames[0]
    frame_one.save(f"{path}.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=2, loop=0)