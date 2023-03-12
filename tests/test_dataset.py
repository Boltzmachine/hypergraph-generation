import torch
from src.dataset import quantizer

def diff_abs(a, b):
    return torch.abs(a - b)

def test_quantize():
    quantize_verts = quantizer.quantize
    dequantize_verts = quantizer.dequantize
    verts = torch.tensor([-0.5, -0.4999, 0.4999, 0., -0.0001, 0.0001, 0.5])
    assert (quantize_verts(verts) == torch.ByteTensor([0, 0, 255, 127, 127, 128, 255])).all(), print(quantize_verts(verts))
    
    assert ( diff_abs(dequantize_verts(quantize_verts(verts)), verts) <= diff_abs(dequantize_verts(quantize_verts(verts+1)), verts) ).all(), print(dequantize_verts(quantize_verts(verts)), (dequantize_verts(quantize_verts(verts+1))))
    assert ( diff_abs(dequantize_verts(quantize_verts(verts)), verts) <= diff_abs(dequantize_verts(quantize_verts(verts-1)), verts) ).all(), print(dequantize_verts(quantize_verts(verts)), (dequantize_verts(quantize_verts(verts-1))))