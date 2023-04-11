import torch
from torch import nn
from tango.integrations.torch import Model


class Criterion(Model):
    ...
    
@Criterion.register("bce")
class BCEWithLogitLoss(Criterion, nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight: float = None, **kwargs) -> None:
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        super().__init__(pos_weight=pos_weight, **kwargs)
    
@Criterion.register("mse")
class MSELoss(Criterion, nn.MSELoss):
    ...

@Criterion.register("ce")
class CrossEntropyLoss(Criterion, nn.CrossEntropyLoss):
    ...