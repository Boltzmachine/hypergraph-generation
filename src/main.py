import numpy as np

from tango import Step
from tango.common import Registrable, Lazy

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .diffusion import LightningModule, Diffusion
from .dataset import Dataset, DataLoader, DataModule
from .modules.distributions import HistDistribution


class LightningTrainer(pl.Trainer, Registrable):
    default_implementation = "default"
LightningTrainer.register("default")(LightningTrainer)


@Step.register("train")
class Train(Step):
    def run(
        self,
        trainer: Lazy[LightningTrainer],
        model: Lazy[LightningModule],
        data_module: DataModule,
        run_name: str = None,
    ) -> None:
        pl.seed_everything(42)
        model = model.construct(dist=HistDistribution(data_module.meta_info))
        
        logger = WandbLogger(
            project="hypergraph-generation",
            save_dir="results/",
            name=run_name
        )
        logger.watch(model)
        
        checkpoint_callback = ModelCheckpoint(dirpath="results/ckpts/", save_top_k=2, monitor="val/edge_acc", mode='max')
        
        trainer = trainer.construct(
            logger=logger, 
            # default_root_dir="results/ckpts/",
            check_val_every_n_epoch=10,
            gradient_clip_val=0.5,
            # gradient_clip_algorithm="value",
            # num_sanity_val_steps=0,
            callbacks=[checkpoint_callback, LearningRateMonitor()]
        )
        trainer.fit(
            model,
            data_module
        )
        # model.load_state_dict(torch.load("results/ckpts/epoch=99-step=2000.ckpt")['state_dict'])
        # trainer.validate(
        #     model,
        #     data_module,
        # )
        
        # model.trainer = trainer
        # model.trainer.datamodule = data_module
        # data_module.setup(None)
        # model.load_state_dict(torch.load("results/ckpts/epoch=59-step=1200.ckpt")['state_dict'])
        # model.to('cuda')
        # model.visualize_sequence()
        # # trainer.validate(
        #     model,
        #     data_module
        # )