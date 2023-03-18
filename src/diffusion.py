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
from .modules.distributions import Distribution
from .modules.model import HyperInitialModel
from .noise_scheduler import DiscreteNoiseScheduler, ContinuousNoiseScheduler

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
            edge_criterion: Model = None,
            node_criterion: Model = nn.MSELoss(reduction="none"),
            visualizer: Union[Visualizer, List[Visualizer]] = BlenderRenderer(),
            ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.transition = transition
        self.edge_criterion = nn.BCEWithLogitsLoss(reduce="none") if isinstance(self.transition.edge_scheduler, DiscreteNoiseScheduler) else nn.MSELoss(reduction="none")
        self.node_criterion = node_criterion
        self.dist = dist
        self.visualizer = visualizer
        
        # self.edge_acc = torchmetrics.Accuracy(task="binary")
        self.edge_acc = torchmetrics.MeanAbsoluteError()
        # self.edge_acc = torchmetrics.Accuracy(task="multiclass", num_classes=transition.edge_scheduler.n_classes)
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
        
    def shared_step(self, batch, batch_idx, t=None):
        """
        step shared by training and validation
        """
        X = batch['X']
        H = batch['H']
        mask = batch['mask']
        
        if t is None:
            t = torch.randint(1, self.transition.T+1, size=(X.size(0),), device=X.device)
        
        XT, _ = self.transit(X, H, torch.ones_like(t) * self.transition.T)[0]
        forward_kwargs = self.construct_forward_kwargs(XT=XT)
        
        (X, X_noise), (H, H_noise) = self.transit(X, H, t)
        pred_X, pred_H = self.forward(X, H, t, mask, **forward_kwargs)
        
        pred_X, true_X, pred_H, true_H = prepare_for_loss_and_metrics(pred_X, X_noise, pred_H, H_noise, mask)
        
        if self.transition.node_scheduler.name == "identity":
            pred_X = true_X
        if self.transition.edge_scheduler.name == "identity":
            if isinstance(self.transition.edge_scheduler, ContinuousNoiseScheduler):
                pred_H = true_H
            else:
                pred_H = F.one_hot(true_H) * 2e10 - 1e10
        
        return pred_X, true_X, pred_H, true_H, t, mask
        
    def training_step(self, batch, batch_idx):
        pred_X, true_X, pred_H, true_H, t, mask = self.shared_step(batch, batch_idx)
        edge_loss = self.edge_criterion(pred_H, true_H.float())
        node_loss = self.node_criterion(pred_X, true_X)

        # reweight = (1 - torch.repeat_interleave(t, mask.sum(1)) / self.transition.T) * 2
        # assert (reweight >= 0).all()
        # node_loss = (node_loss * reweight[:, None]).mean()
        # edge_loss = (edge_loss * reweight[:, None])
        edge_loss = edge_loss.mean()
        node_loss = node_loss.mean()# * 3

        self.log("train/node_loss", node_loss)
        self.log("train/edge_loss", edge_loss)

        return node_loss + edge_loss
    
    def validation_step(self, batch, batch_idx):
        t = torch.ones(batch['X'].size(0), dtype=torch.long, device=batch['X'].device)
        pred_X, true_X, pred_H, true_H, t, mask = self.shared_step(batch, batch_idx, t)

        # pred_H = (pred_H > 0).long()
        # true_H = (true_H > 0).long()
        self.edge_acc(pred_H, true_H)
        self.node_mae(pred_X, true_X)

        return 

    def on_validation_epoch_end(self) -> None:
        # self.log("val/node_acc", self.node_acc)
        self.log("val/node_mae", self.node_mae)
        self.log("val/edge_acc", self.edge_acc)
        
        X, H = self.sample_batch(fake_sample=False)
        images = self.render_samples(quantizer.quantize2(X), H)
        grid = make_grid(images, nrow=4)
        grid = grid.float()
        wandb.log({
            "samples": wandb.Image(grid), 
            "global_step": self.trainer.global_step
            })
        self.plot_every_step_loss()
        # self.visualize_sequence()
        # self.renderer.save_file()

    def visualize_sequence(self):
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
    
    def denoise(self, *args, deterministic=False, last_step=False, **kwargs):
        X0, H0 = self.forward(*args, **kwargs)
        Xt, Ht, t, _ = args
        assert Xt.size() == X0.size()
        
        return self.transition.denoise((X0, Xt), (H0, Ht), t, deterministic, last_step)
    
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
            sample_start = 200
            datalist = mask_collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
            X = datalist['X'].to(device)
            mask = datalist['mask'].to(device)
            H = datalist['H'].to(device)

            if return_every_step:
                gt = [(X, H)]
                for t in range(1, sample_start+1):
                    (Xt, _), (Ht, _) = self.transit(X, H, t * torch.ones(bs, dtype=torch.long, device=device))
                    gt.append((Xt, Ht))
                X, H = gt[-1]
            else:
                (X, _), (H, _) = self.transit(X, H, sample_start * torch.ones(bs, dtype=torch.long, device=device))

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
                if isinstance(self.transition.edge_scheduler, DiscreteNoiseScheduler):
                    H = (torch.rand(bs, self.dist.max_faces, max_nodes, device=device) > 0.5).long()
                else:
                    H = torch.randn(size=(bs, self.dist.max_faces, max_nodes), device=device)
        
        forward_kwargs = self.construct_forward_kwargs(XT=X)
        # reverse process
        for t in reversed(range(2, sample_start+1)):
            t = torch.ones(bs, dtype=torch.long, device=device) * t
            X, H = self.denoise(X, H, t, mask, deterministic=False, **forward_kwargs)

            if return_every_step:
                pr.insert(0, (X, H))

        t = torch.ones_like(t)
        X, H = self.denoise(X, H, t, mask, deterministic=True, last_step=True, **forward_kwargs)

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
    
    def plot_every_step_loss(self, bs: int = 256, device = torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        import matplotlib.pyplot as plt
        batch = mask_collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
        H_mse = []
        X_mse = []
        for t in range(1, self.transition.T+1):
            t = torch.ones(bs, device=device, dtype=torch.long) * t
            
            X = batch['X'].to(device)
            H = batch['H'].to(device)
            mask = batch['mask'].to(device)
            
            (X, X_noise), (H, H_noise) = self.transit(X, H, t)
            pred_X, pred_H = self.forward(X, H, t, mask)
            pred_X, true_X, pred_H, true_H = prepare_for_loss_and_metrics(pred_X, X_noise, pred_H, H_noise, mask)
            H_mse.append(F.mse_loss(pred_H, true_H).item())
            X_mse.append(F.mse_loss(pred_X, true_X).item())
            
        ax1.plot(range(len(H_mse)), H_mse)
        ax2.plot(range(len(X_mse)), X_mse)
        
        plt.savefig("1.png")
            