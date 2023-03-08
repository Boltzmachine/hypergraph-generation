import torch

from src.utils import masked_select_H, masked_select_X, prepare_for_loss_and_metrics, create_mask_from_length

import pytest


def test_masked_select():
    bs = 3
    true_X = torch.rand(bs, 7, 3)
    pred_X = torch.rand(bs, 7, 3, 256)
    
    pred_H = torch.rand(bs, 12, 7)
    true_H = (torch.rand_like(pred_H) > 0.5).float()
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0],
    ], dtype=torch.long)
    
    pred_X_, true_X_, pred_H_, true_H_ = prepare_for_loss_and_metrics(pred_X, true_X, pred_H, true_H, mask)
    assert (pred_X_ == torch.cat([pred_X[0, :5], pred_X[1], pred_X[2, :6]], dim=0).transpose(1, -1)).all()
    assert (true_X_ == torch.cat([true_X[0, :5], true_X[1], true_X[2, :6]], dim=0).transpose(1, -1)).all()
    pred_H = pred_H.transpose(1, 2)
    true_H = true_H.transpose(1, 2)
    assert (pred_H_ == torch.cat([pred_H[0, :5], pred_H[1], pred_H[2, :6]], dim=0)).all()
    assert (true_H_ == torch.cat([true_H[0, :5], true_H[1], true_H[2, :6]], dim=0)).all()

def test_create_mask_from_length():
    length = torch.tensor([9, 10, 11, 12, 6, 3])
    mask = create_mask_from_length(length)
    mask_ = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    ])
    assert (mask == mask_).all()
    

    