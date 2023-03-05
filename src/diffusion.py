import os
import random
import shutil
import wandb

import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchvision.utils import make_grid

from collections import namedtuple

from .dataset import DataContainer, CuboidDataset, quantizer
from .noise_scheduler import NoiseScheduler, Transition
from .render import BlenderRenderer
from .utils import VerticesMutedError

from tango.integrations.torch import Model
from tango.common.registrable import Registrable


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
            edge_criterion: Model = nn.BCEWithLogitsLoss(),
            node_criterion: Model = nn.CrossEntropyLoss()
            ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.transition = transition
        self.edge_criterion = edge_criterion
        self.node_criterion = node_criterion
        
        self.renderer = BlenderRenderer()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
        
    def training_step(self, batch):
        X = batch['X']
        H = batch['H']
        
        t = torch.randint(1, self.transition.T+1, size=(X.size(0),))
        X, H = self.transit(X, H, t)
        X, H = self.forward(X, H)
        
        edge_loss = self.edge_criterion(H, batch['H'].float()) 
        node_loss = self.node_criterion(X.transpose(1, -1), batch['X'].transpose(1, -1))

        self.log("train/node_loss", node_loss)
        self.log("train/edge_loss", edge_loss)
        
        return node_loss + edge_loss
        
    def on_train_epoch_end(self):
        X, H = self.sample_batch()
        images = self.render_samples(quantizer.dequantize(X), H)
        grid = make_grid(images, nrow=4)
        grid = grid.float()
        wandb.log({
            "samples": wandb.Image(grid), 
            "global_step": self.trainer.global_step
            })
        
    def forward(self, X, H):
        # X = quantizer.dequantize(X)
        X, H = self.model(X, H)
  
        return X, H
    
    def denoise(self, *args, deterministic=False, **kwargs):
        X, H = self.forward(*args, **kwargs)
        
        X_prob = torch.softmax(X, -1)
        H = torch.sigmoid(H)
        H_prob = torch.stack([1-H, H], dim=-1)
        
        if deterministic:
            H = H_prob.argmax(-1)
            X = X_prob.softmax(-1).argmax(-1)
        else:
            H_prob = H_prob.view(-1, H_prob.size(-1))
            H = H_prob.multinomial(1).view(*H.size())
            X_prob = X_prob.view(-1, X_prob.size(-1))
            X = X_prob.multinomial(1).view(*X.size()[:-1])
        
        return X, H
    
    def transit(self, X, H, t):
        return self.transition.transit(X, H, t)
    
    @torch.no_grad()
    def sample_batch(self, device = torch.device('cuda')):
        if device is None:
            device = self.trainer.device
        X = torch.stack([CuboidDataset.gen_verts() for _ in range(16)], dim=0).to(device)
        H = (torch.rand(X.size(0), 6, 8, device=device) > 0.5).long()
        for t in reversed(range(1, self.transition.T+1)):
            t = torch.ones(X.size(0), dtype=torch.long, device=device)
            X, H = self.denoise(X, H)
            X, H = self.transit(X, H, t)
        X, H = self.denoise(X, H, deterministic=True)
        return X, H
    
    def render_samples(self, X, H):
        in_mem_dir = "/dev/shm" # store file in memory so it would be faster and still compatible to API
        temp_dir = "hypergen_render"
        dir_path = os.path.join(in_mem_dir, temp_dir)
        dir_path = "./results/wavefront"
        # shutil.rmtree(dir_path)
        # os.makedirs(dir_path, exist_ok=True)
        
        # write obj
        # obj_paths = []
        images = []
        for i, (x, h) in enumerate(zip(list(X), list(H))):
            obj_path = os.path.join(dir_path, f"{i}.obj") 
            # obj_paths.append(obj_path)
            with open(obj_path, 'w') as file:
                for vert in x.tolist():
                    file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
            # This face's index is form 0 istead of 1!
            faces = set()
            for face in h:
                face = face.nonzero(as_tuple=True)[0].tolist()
                # file.write(f"#f " + " ".join([str(f) for f in face]))
                faces.add(tuple(face))

            try:
                image = self.renderer.render_obj(obj_path, faces)
            except VerticesMutedError:
                image = torch.zeros(4, 256, 256, dtype=torch.uint8)
            images.append(image)
            
        images = torch.stack(images)
        return images
