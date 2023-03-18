from torch import nn
from tango.integrations.torch import Model


class Criterion(Model):
    ...
    
@Criterion.register("bce")
class BCEWithLogitLoss(Criterion):
    ...