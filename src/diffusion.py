import os
import random
import shutil
from collections import namedtuple
import wandb

import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchvision.utils import make_grid, save_image

from .dataset import DataContainer, CuboidDataset, quantizer
from .noise_scheduler import NoiseScheduler, Transition
from .visualize import BlenderRenderer, MatplotlibPlotter, Visualizer
from .utils import VerticesMutedError, masked_select_H, masked_select_X, prepare_for_loss_and_metrics, create_mask_from_length
from .distributions import Distribution

from tango.integrations.torch import Model
from tango.common.registrable import Registrable

from typing import Any


class LightningModule(pl.LightningModule, Registrable):
    default_implementation = "default"
LightningModule.register("default")(LightningModule)


@LightningModule.register("diffusion")
class Diffusion(pl.LightningModule):
    def __init__(
            self,
            model: Model,
            learning_rate: float,
            transition: Transition,
            dist: Distribution,
            edge_criterion: Model = nn.BCEWithLogitsLoss(reduction="none"),
            node_criterion: Model = nn.CrossEntropyLoss(reduction="none"),
            visualizer: Visualizer = BlenderRenderer(),
            ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.transition = transition
        self.edge_criterion = edge_criterion
        self.node_criterion = node_criterion
        self.dist = dist
        self.visualizer = visualizer
        
        self.edge_acc = torchmetrics.Accuracy(task="binary")
        self.node_acc = torchmetrics.Accuracy(task="multiclass", num_classes=transition.node_scheduler.n_classes)
        self.node_mae = torchmetrics.MeanAbsoluteError()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "train/node_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
    def training_step(self, batch, batch_idx):
        X = batch['X']
        H = batch['H']
        mask = batch['mask']
        
        t = torch.randint(1, self.transition.T+1, size=(X.size(0),), device=X.device)

        X, H = self.transit(X, H, t)
        X, H = self.forward(X, H, t, mask)
        
        pred_X, true_X, pred_H, true_H = prepare_for_loss_and_metrics(X, batch['X'], H, batch['H'], mask)
        edge_loss = self.edge_criterion(pred_H, true_H.float())
        node_loss = self.node_criterion(pred_X, true_X)

        reweight = (1 - torch.repeat_interleave(t, mask.sum(1)) / self.transition.T) * 2
        assert (reweight >= 0).all()

        # node_loss = (node_loss * reweight[:, None]).mean()
        # edge_loss = (edge_loss * reweight[:, None]).mean()
        edge_loss = edge_loss.mean()
        node_loss = node_loss.mean()

        self.log("train/node_loss", node_loss)
        self.log("train/edge_loss", edge_loss)

        return node_loss + edge_loss
        
    # def on_train_epoch_end(self):
    #     self.log("train/node_acc", self.node_acc)
    #     self.log("train/node_mae", self.node_mae)
    #     self.log("train/edge_acc", self.edge_acc)
        
    
    def validation_step(self, batch, batch_idx):
        X = batch['X']
        H = batch['H']
        mask = batch['mask']
        
        t = torch.randint(1, self.transition.T+1, size=(X.size(0),))
        X, H = self.transit(X, H, t)
        X, H = self.forward(X, H, t, mask)
        
        pred_X, true_X, pred_H, true_H = prepare_for_loss_and_metrics(X, batch['X'], H, batch['H'], mask)
        self.edge_acc(pred_H, true_H)
        self.node_acc(pred_X, true_X)
        self.node_mae(quantizer.dequantize(pred_X.transpose(1, -1).argmax(-1)), quantizer.dequantize(true_X))

        return 

    def on_validation_epoch_start(self):
        X, H = self.sample_batch()
        images = self.render_samples(quantizer.dequantize(X), H)
        grid = make_grid(images, nrow=4)
        grid = grid.float()
        wandb.log({
            "samples": wandb.Image(grid), 
            "global_step": self.trainer.global_step
            })
        # self.renderer.save_file()

    def on_validation_epoch_end(self) -> None:
        self.log("val/node_acc", self.node_acc)
        self.log("val/node_mae", self.node_mae)
        self.log("val/edge_acc", self.edge_acc)

    def visualize_sequence(self):
        assert isinstance(self.visualizer, MatplotlibPlotter)
        gt, pr = self.sample_batch(bs=1, return_every_step=True, fake_sample=True)
        for t, (X, H) in enumerate(gt):
            assert X.size(0) == H.size(0) == 1
            x = quantizer.dequantize(X.squeeze())
            h = H.squeeze()
            image = self.visualizer.visualize_object(x, h) / 255.
            save_image(image, f"results/steps/gt/{t}.png")
        for t, (X, H) in enumerate(pr):
            x = quantizer.dequantize(X.squeeze())
            h = H.squeeze()
            image = self.visualizer.visualize_object(x, h) / 255.
            save_image(image, f"results/steps/pr/{t}.png")
        
    def forward(self, X, H, t, mask):
        # X = quantizer.dequantize(X)
        X, H = self.model(X, H, t, mask)
  
        return X, H
    
    def denoise(self, *args, deterministic=False, **kwargs):
        X, H = self.forward(*args, **kwargs)
        
        X_prob = torch.softmax(X, -1)
        H = torch.sigmoid(H)
        H_prob = torch.stack([1-H, H], dim=-1)
        
        if deterministic:
            H = H_prob.argmax(-1)
            X = X_prob.argmax(-1)
        else:
            H_prob = H_prob.view(-1, H_prob.size(-1))
            H = H_prob.multinomial(1).view(*H.size())
            X_prob = X_prob.view(-1, X_prob.size(-1))
            X = X_prob.multinomial(1).view(*X.size()[:-1])
        
        return X, H
    
    def transit(self, X, H, t):
        return self.transition.transit(X, H, t)
    
    @torch.no_grad()
    def sample_batch(self, bs: int = 16, device = torch.device('cuda' if torch.cuda.is_available() else "cpu"), return_every_step: bool = False, fake_sample: bool = False):
        if device is None:
            device = self.trainer.device
        n_nodes = self.dist.sample_n(bs, device)
        max_nodes = n_nodes.max()
        
        # sampling noise
        mask = create_mask_from_length(n_nodes)
        
        if return_every_step:
            pr = []
        if fake_sample:
            # self.transition.T = 50
            X = torch.stack([CuboidDataset.gen_verts() for _ in range(bs)]).to(device)
            H = CuboidDataset.static_hyperedge[None, ...].expand(bs, -1, -1).to(device)
            if return_every_step:
                gt = [(X, H)] + [self.transit(X, H, t * torch.ones(bs, dtype=torch.long, device=device)) for t in range(1, self.transition.T+1)]
            X, H = gt[-1]

        else:
            if self.transition.node_scheduler.name == "identity":
                X = torch.stack([CuboidDataset.gen_verts() for _ in range(bs)]).to(device)
            else:
                X = (torch.randint(0, 256, size=(bs, max_nodes, 3), device=device))
            if self.transition.edge_scheduler.name == "identity":
                H = CuboidDataset.static_hyperedge[None, ...].expand(bs, -1, -1).to(device)
            else:
                H = (torch.rand(bs, self.dist.max_faces, max_nodes, device=device) > 0.5).long()
            
        # reverse process
        for t in reversed(range(2, self.transition.T+1)):
            # if t == 2:
            #     import pudb; pu.db
            t = torch.ones(bs, dtype=torch.long, device=device) * t
            X, H = self.denoise(X, H, t, mask, deterministic=False)
            X, H = self.transit(X, H, t-1)
            if return_every_step:
                pr.insert(0, (X, H))

        t = torch.ones_like(t)
        X, H = self.denoise(X, H, t, mask, deterministic=True)

        if return_every_step:
            pr.insert(0, (X, H))
                
        if return_every_step:
            assert len(gt) == len(pr) + 1
            return gt, pr
        else:
            return X, H
    
    def render_samples(self, X, H):
        # shutil.rmtree(dir_path)
        # os.makedirs(dir_path, exist_ok=True)
        
        # write obj
        # obj_paths = []
        images = []
        for i, (x, h) in enumerate(zip(list(X), list(H))):
            try:
                image = self.visualizer.visualize_object(x, h, i)
            except VerticesMutedError:
                image = torch.zeros(4, 256, 256, dtype=torch.uint8)
            images.append(image)
            
        images = torch.stack(images)
        return images
