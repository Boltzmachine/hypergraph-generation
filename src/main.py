import numpy as np

from tango import Step
from tango.common import Registrable, Lazy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from .diffusion import LightningModule
from .dataset import Dataset, DataLoader, DataModule
from .distributions import HistDistribution


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
    ) -> None:
        model = model.construct(dist=HistDistribution(data_module.meta_info))
        
        logger = WandbLogger(
            project="hypergraph-generation",
            save_dir="results/"
        )
        logger.watch(model)
        
        trainer = trainer.construct(
            logger=logger, 
            default_root_dir="results/ckpts",
            check_val_every_n_epoch=5,
            gradient_clip_val=0.6,
            callbacks=[LearningRateMonitor()]
        )
        trainer.fit(
            model,
            data_module
        )