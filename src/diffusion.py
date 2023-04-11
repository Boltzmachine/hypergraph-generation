import os
import random
import shutil
from collections import namedtuple
import wandb
from tqdm import tqdm
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchvision.utils import make_grid, save_image

from .dataset import DataContainer, CuboidDataset, quantizer
from .transition import Transition
from .visualize import BlenderRenderer, MatplotlibPlotter, Visualizer
from .utils import VerticesMutedError, masked_select_H, masked_select_X, prepare_for_loss_and_metrics, create_mask_from_length, make_gif
from .modules.distributions import Distribution
from .modules.criterion import Criterion
from .noise_scheduler import DiscreteNoiseScheduler, ContinuousNoiseScheduler

from tango.integrations.torch import Model
from tango.common.registrable import Registrable

from typing import Any, Union, List, Optional


def scatter_loss_per_step(loss, mask):
    loss = loss.detach()
    grp = torch.repeat_interleave(torch.arange(mask.size(0), device=mask.device), mask.sum(1))
    mean = torch.zeros(mask.size(0), dtype=torch.float, device=mask.device)
    mean = torch.scatter_add(mean, 0, grp, loss)
    mean = mean / mask.sum(1)
    return mean


class StepEMA(Model):
    def __init__(self, T: int, name: str) -> None:
        super().__init__()
        self.T = T
        self.alpha = 0.99
        self.name = name
        
        self.register_buffer("buffer", torch.zeros(T, dtype=torch.float))
        self.register_buffer("initial", torch.ones(T, dtype=torch.bool)) # True if no value
    
    @torch.no_grad()
    def update(self, t, data):
        t = t-1
        alpha = self.alpha
        data = torch.clamp(data, max=1)
        index, counts = torch.unique(t, return_counts=True)
        add = torch.zeros(self.T, dtype=torch.float, device=t.device)
        add = torch.scatter_add(add, 0, t, data)
        add[index] = add[index] / counts 
        
        index_bool = torch.zeros_like(self.buffer).bool()
        index_bool[index] = True
        
        conquer_index = torch.logical_and(index_bool, self.initial)
        update_index = torch.logical_xor(index_bool, conquer_index)
        
        self.buffer[update_index] = alpha * self.buffer[update_index] + (1 - alpha) * add[update_index]
        self.buffer[conquer_index] = add[conquer_index]

        self.initial = torch.logical_and(self.initial, ~index_bool)
    
    def log(self):
        import matplotlib.pyplot as plt
        buffer = self.buffer.cpu().numpy()

        plt.figure()
        plt.plot(range(len(buffer)), buffer)
        plt.savefig(f"{self.name}.png")
        self.buffer = torch.zeros(self.T, dtype=torch.float, device=self.buffer.device)
        self.initial = torch.ones(self.T, dtype=torch.bool, device=self.initial.device)
        plt.close()
        

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
            dist: Optional[Distribution] = None,
            face_criterion: Optional[Criterion] = None,
            edge_criterion: Optional[Criterion] = None,
            node_criterion: Optional[Criterion] = None,
            visualizer: Union[Visualizer, List[Visualizer]] = BlenderRenderer(),
            sample_bs: int = 16,
            ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.transition = transition
        self.face_criterion = face_criterion
        self.node_criterion = node_criterion
        self.edge_criterion = edge_criterion
        self.dist = dist
        self.visualizer = visualizer
        self.sample_bs = sample_bs

        self.do_face = face_criterion is not None
        self.do_edge = edge_criterion is not None
        assert (self.do_face or self.do_edge)
        assert not (self.do_face and self.do_edge)
        
        self.edge_acc = torchmetrics.Accuracy(task="binary") if self.do_face else torchmetrics.Accuracy(task="multiclass", num_classes=transition.edge_scheduler.n_classes)
        self.node_mae = torchmetrics.MeanAbsoluteError()
        
        # self.node_ema = StepEMA(transition.T, "node")
        # self.edge_ema = StepEMA(transition.T, "edge")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        #         "monitor": "train/edge_acc",
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }
        
    def shared_step(self, batch, batch_idx, t=None):
        """
        step shared by training and validation
        """
        X = batch['X']
        E = batch['E']
        H = batch['H']
        mask = batch['mask']
        
        if t is None:
            t = torch.randint(1, self.transition.T+1, size=(X.size(0),), device=X.device)
        
        # XT, _ = self.transit(X, H, torch.ones_like(t) * self.transition.T)[0]
        forward_kwargs = self.construct_forward_kwargs()
        
        (X, X_noise), (E, E_noise), (H, H_noise) = self.transit(X, E, H, t)

        pred_X, pred_E, pred_H = self.forward(X, E, H, t, mask, **forward_kwargs)
        
        pred_X, true_X, pred_E, true_E, pred_H, true_H = prepare_for_loss_and_metrics(pred_X, X_noise, pred_E, E_noise, pred_H, H_noise, mask)
        
        if self.transition.node_scheduler.name == "identity":
            pred_X = true_X
        if self.transition.edge_scheduler.name == "identity":
            if isinstance(self.transition.edge_scheduler, ContinuousNoiseScheduler):
                pred_H = true_H
            else:
                pred_H = true_H * 2e10 - 1e10
                pred_E = F.one_hot(true_E, num_classes=2).float() * 2e10 - 1e10
        
        return pred_X, true_X, pred_E, true_E, pred_H, true_H, t, mask
        
    def training_step(self, batch, batch_idx):
        pred_X, true_X, pred_E, true_E, pred_H, true_H, t, mask = self.shared_step(batch, batch_idx)

        if self.do_face:
            edge_loss = self.face_criterion(pred_H.unsqueeze(-1), true_H.float().unsqueeze(-1))
        if self.do_edge:
            edge_loss = self.edge_criterion(pred_E, true_E)

        node_loss = self.node_criterion(pred_X, true_X)
        # reweight = (1 - torch.repeat_interleave(t, mask.sum(1)) / self.transition.T) * 2
        # assert (reweight >= 0).all()
        # node_loss = (node_loss * reweight[:, None]).mean()
        # edge_loss = (edge_loss * reweight[:, None])
        edge_loss = edge_loss.mean(-1).mean(-1)
        node_loss = node_loss.mean(-1)# * 3
                
        # self.node_ema.update(t, scatter_loss_per_step(node_loss, mask))
        # self.edge_ema.update(t, scatter_loss_per_step(edge_loss, mask))
        
        edge_loss = edge_loss.mean()
        node_loss = node_loss.mean()# * 3

        self.log("train/node_loss", node_loss)
        self.log("train/edge_loss", edge_loss)
        
        if self.transition.edge_scheduler.name == "identity" and self.transition.node_scheduler.name == "identity":
            loss = torch.zeros(1, device=pred_X.device, requires_grad=True)
        else:
            loss = node_loss + edge_loss

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        # t = torch.ones(batch['X'].size(0), dtype=torch.long, device=batch['X'].device)
        pred_X, true_X, pred_E, true_E, pred_H, true_H, t, mask = self.shared_step(batch, batch_idx)

        if self.do_face:
            self.edge_acc(pred_H, true_H)
        if self.do_edge:
            self.edge_acc(pred_E, true_E)

        self.node_mae(pred_X, true_X)


    def on_validation_epoch_end(self) -> None:
        # self.log("val/node_acc", self.node_acc)
        self.log("val/node_mae", self.node_mae)
        self.log("val/edge_acc", self.edge_acc)
        # self.node_ema.log()
        # self.edge_ema.log()

        X, E, H, mask = self.sample_batch(bs=self.sample_bs)        
        X = quantizer.quantize2(X)
        
        images = self.render_samples(X, E, H, mask)
        grid = make_grid(images, nrow=4)
        grid = grid.float()
        wandb.log({
            "samples": wandb.Image(grid), 
            "global_step": self.trainer.global_step
            })
        # self.plot_every_step_loss()
        # self.visualize_sequence()
        # self.renderer.save_file()

    def visualize_sequence(self):
        gt, pr = self.sample_batch(bs=8, return_every_step=True, fake_sample=500)
                
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
        
    def forward(self, X, E, H, t, mask, **kwargs):
        # X = quantizer.dequantize(X)
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        mask - [bs, n_nodes]
        """
        X, E, H = self.model(X, E, H, t, mask, **kwargs)
        X = X * mask[..., None]
        H = H * mask[:, None, :]
  
        return X, E, H
    
    def denoise(self, *args, deterministic=False, last_step=False, **kwargs):
        X0, E0, H0 = self.forward(*args, **kwargs)
        Xt, Et, Ht, t, *_ = args
        assert Xt.size() == X0.size()
        
        return self.transition.denoise((X0, Xt), (E0, Et), (H0, Ht), t, deterministic, last_step)
    
    def transit(self, X, E, H, t):
        return self.transition.transit(X, E, H, t)
    
    @torch.no_grad()
    def sample_batch(self, bs: int = 16, device = torch.device('cuda' if torch.cuda.is_available() else "cpu"), return_every_step: bool = False, fake_sample: Union[int, bool] = False):
        n_nodes = self.dist.sample_n(bs, device)
        max_nodes = n_nodes.max()
        collate_fn = self.trainer.datamodule.collate_fn

        sample_start = self.transition.T
        if return_every_step:
            pr = []
        if fake_sample:
            sample_start = fake_sample
            datalist = collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
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
                datalist = collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
                n_nodes = datalist['mask'].sum(1).to(device)
                max_nodes = n_nodes.max()
            if self.transition.node_scheduler.name == "identity":
                X = datalist['X'].to(device)
                mask = datalist['mask'].to(device)
            else:
                mask = create_mask_from_length(n_nodes)
                X = torch.randn(size=(bs, max_nodes, 3), device=device)
            if self.transition.edge_scheduler.name == "identity":
                H = datalist['H'].to(device)
                E = datalist['E'].to(device)
            else:
                if isinstance(self.transition.edge_scheduler, DiscreteNoiseScheduler):
                    H = (torch.rand(bs, self.dist.max_faces, max_nodes, device=device) > 0.5).long()
                    E = (torch.rand(bs, max_nodes, max_nodes, device=device) > 0.5).long()
                else:
                    H = torch.randn(size=(bs, self.dist.max_faces, max_nodes), device=device)

        forward_kwargs = self.construct_forward_kwargs(XT=X)
        # reverse process
        for t in reversed(range(2, sample_start+1)):
            t = torch.ones(bs, dtype=torch.long, device=device) * t
            
            X, E, H = self.denoise(X, E, H, t, mask, deterministic=False, **forward_kwargs)

            if return_every_step:
                pr.insert(0, (X, H))

        t = torch.ones_like(t)
        X, E, H = self.denoise(X, E, H, t, mask, deterministic=True, last_step=True, **forward_kwargs)

        if return_every_step:
            pr.insert(0, (X, H))
                
        if return_every_step:
            assert len(gt) == len(pr) + 1
            return gt, pr
        else:
            return X, E, H, mask
    
    def render_samples(self, X, E, H, mask):
        images = []
        for i, (x, e, h, m) in enumerate(zip(list(X), list(E), list(H), list(mask))):
            try:
                image = self.visualizer.visualize_object(x, e, h, m, i)
            except VerticesMutedError:
                image = torch.zeros(4, 256, 256, dtype=torch.uint8)
            images.append(image)
            
        images = torch.stack(images)
        return images
    
    def construct_forward_kwargs(self, **kwargs):
        forward_kwargs = {}
        return forward_kwargs
    
    def plot_every_step_loss(self, bs: int = 256, device = torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        import matplotlib.pyplot as plt
        batch = self.trainer.datamodule.collate_fn([self.trainer.datamodule.val[i] for i in range(bs)])
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