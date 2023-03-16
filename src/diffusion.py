import os
import random
import shutil
from collections import namedtuple
import wandb
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchvision.utils import make_grid, save_image

from .dataset import DataContainer, CuboidDataset, quantizer, mask_collate_fn
from .transition import Transition
from .visualize import BlenderRenderer, MatplotlibPlotter, Visualizer
from .utils import VerticesMutedError, masked_select_H, masked_select_X, prepare_for_loss_and_metrics, create_mask_from_length, make_gif
from .distributions import Distribution
from .model import HyperInitialModel

from tango.integrations.torch import Model
from tango.common.registrable import Registrable

from typing import Any, Union, List


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
            node_criterion: Model = nn.L1Loss(reduction="none"),
            visualizer: Union[Visualizer, List[Visualizer]] = BlenderRenderer(),
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
        # self.node_acc = torchmetrics.Accuracy(task="multiclass", num_classes=transition.node_scheduler.n_classes)
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
        
    def shared_step(self, batch, batch_idx):
        """
        step shared by training and validation
        """
        X = batch['X']
        H = batch['H']
        mask = batch['mask']
                
        t = torch.randint(1, self.transition.T+1, size=(X.size(0),), device=X.device)
        
        XT, _ = self.transit(X, H, torch.ones_like(t) * self.transition.T)[0]
        forward_kwargs = self.construct_forward_kwargs(XT=XT)
        
        (X, X_noise), (H, H_noise) = self.transit(X, H, t)
        pred_X, H = self.forward(X, H, t, mask, **forward_kwargs)
        
        pred_X, true_X, pred_H, true_H = prepare_for_loss_and_metrics(pred_X, X_noise, H, batch['H'], mask)
        
        if self.transition.node_scheduler.name == "identity":
            pred_X = true_X
        if self.transition.edge_scheduler.name == "identity":
            pred_H = F.one_hot(true_H) * 2e10 - 1e10
        
        return pred_X, true_X, pred_H, true_H, t, mask
        
    def training_step(self, batch, batch_idx):
        pred_X, true_X, pred_H, true_H, t, mask = self.shared_step(batch, batch_idx)
        edge_loss = self.edge_criterion(pred_H, true_H.float())
        node_loss = self.node_criterion(pred_X, true_X)

        reweight = (1 - torch.repeat_interleave(t, mask.sum(1)) / self.transition.T) * 2
        assert (reweight >= 0).all()

        # node_loss = (node_loss * reweight[:, None]).mean()
        # edge_loss = (edge_loss * reweight[:, None]).mean()
        edge_loss = edge_loss.mean()
        node_loss = node_loss.mean() * 3

        self.log("train/node_loss", node_loss)
        self.log("train/edge_loss", edge_loss)

        return node_loss + edge_loss
    
    def validation_step(self, batch, batch_idx):
        pred_X, true_X, pred_H, true_H, t, mask = self.shared_step(batch, batch_idx)

        self.edge_acc(pred_H, true_H)
        self.node_mae(pred_X, true_X)

        return 

    def on_validation_epoch_start(self):
        X, H = self.sample_batch(fake_sample=False)
        images = self.render_samples(quantizer.quantize2(X), H)
        grid = make_grid(images, nrow=4)
        grid = grid.float()
        wandb.log({
            "samples": wandb.Image(grid), 
            "global_step": self.trainer.global_step
            })
        # self.renderer.save_file()

    def on_validation_epoch_end(self) -> None:
        # self.log("val/node_acc", self.node_acc)
        self.log("val/node_mae", self.node_mae)
        self.log("val/edge_acc", self.edge_acc)

    def visualize_sequence(self):
        assert isinstance(self.visualizer, MatplotlibPlotter)
        gt, pr = self.sample_batch(bs=8, return_every_step=True, fake_sample=True)
                
        def to_gif(tensors, path):
            images = [[] for _ in range(tensors[0][0].size(0))]
            for t, (X, H) in enumerate(tqdm(tensors)):
                X = quantizer.quantize2(X.squeeze())
                H = H.squeeze()
                for i, (x, h) in enumerate(zip(X, H)):
                    image = self.visualizer.visualize_object(x, h) / 255.
                    images[i].insert(0, image)
            for i, image in enumerate(images):
                make_gif(image, path + str(i))
            # save_image(image, f"results/steps/gt/{t}.png")
        # to_gif(gt, "results/steps/gt")
        to_gif(pr, "results/steps/pr")
            # save_image(image, f"results/steps/pr/{t}.png")
        
    def forward(self, X, H, t, mask, **kwargs):
        # X = quantizer.dequantize(X)
        X, H = self.model(X, H, t, mask, **kwargs)
  
        return X, H
    
    def denoise(self, *args, deterministic=False, **kwargs):
        X0, H0 = self.forward(*args, **kwargs)
        Xt, Ht, t, _ = args
        assert Xt.size() == X0.size()
        
        return self.transition.denoise((X0, Xt), (H0, Ht), t, deterministic)
    
    def transit(self, X, H, t):
        return self.transition.transit(X, H, t)
    
    @torch.no_grad()
    def sample_batch(self, bs: int = 16, device = torch.device('cuda' if torch.cuda.is_available() else "cpu"), return_every_step: bool = False, fake_sample: bool = False):
        n_nodes = self.dist.sample_n(bs, device)
        max_nodes = n_nodes.max()

        sample_start = self.transition.T
        if return_every_step:
            pr = []
        if fake_sample:
            # sample_start = 50
            datalist = mask_collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
            X = datalist['X'].to(device)
            mask = datalist['mask'].to(device)
            H = datalist['H'].to(device)

            if return_every_step:
                gt = [(X, H)]
                for t in range(1, sample_start+1):
                    (Xt, _), Ht = self.transit(X, H, t * torch.ones(bs, dtype=torch.long, device=device))
                    gt.append((Xt, Ht))
                X, H = gt[-1]
            else:
                (X, _), H = self.transit(X, H, sample_start * torch.ones(bs, dtype=torch.long, device=device))

        else:
            if self.transition.node_scheduler.name == "identity" or self.transition.edge_scheduler.name == "identity":
                datalist = mask_collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
            if self.transition.node_scheduler.name == "identity":
                X = datalist['X'].to(device)
                mask = datalist['mask'].to(device)
            else:
                mask = create_mask_from_length(n_nodes)
                X = torch.randn(size=(bs, max_nodes, 3), device=device)
            if self.transition.edge_scheduler.name == "identity":
                H = datalist['H'].to(device)
            else:
                H = (torch.rand(bs, self.dist.max_faces, max_nodes, device=device) > 0.5).long()
        
        forward_kwargs = self.construct_forward_kwargs(XT=X)
        # reverse process
        for t in reversed(range(2, sample_start+1)):
            t = torch.ones(bs, dtype=torch.long, device=device) * t
            X, H = self.denoise(X, H, t, mask, deterministic=False, **forward_kwargs)

            if return_every_step:
                pr.insert(0, (X, H))

        t = torch.ones_like(t)
        X, H = self.denoise(X, H, t, mask, deterministic=True, **forward_kwargs)

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
    
    def construct_forward_kwargs(self, **kwargs):
        forward_kwargs = {}
        if isinstance(self.model, HyperInitialModel):
            forward_kwargs["XT"] = kwargs["XT"]
        return forward_kwargs