import torch
from torch import nn
import pytorch_lightning as pl
from collections import namedtuple

from .noise_scheduler import NoiseScheduler

from tango.integrations.torch import Model
from tango.common.registrable import Registrable


class LightningModule(pl.LightningModule, Registrable):
    """
    """
LightningModule.register("default")(LightningModule)


@LightningModule.register("diffusion")
class Diffusion(pl.LightningModule):
    def __init__(
            self,
            model: Model,
            noise_scheduler: NoiseScheduler,
            criterion: Model = nn.BCEWithLogitsLoss(),
            ) -> None:
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.criterion = criterion
        
    def training_step(self, batch):
        X = batch['X']
        H = batch['H']
        
        X, H = self.transit(X, H, t)
        X, H = self.forward(X, H)
        
        loss = self.criterion(H)
        self.log("train/loss", loss)
        
    def validataion_step(self):
        samples = sample_batch()
        
    def forward(self, X, H):
        X, H = self.model(X, H)
  
        return X, H
    
    def transit(self, X, H, t):
        Qt_bar = self.noise_scheduler.get_Qt_bar(t)
        
        H = H @ Qt_bar
        
        return DataContainer(X=X, H=H)
    
    @torch.no_grad()
    def sample_batch(self):
        return
