from tango import Step
from tango.common import Registrable, Lazy
from tango.integrations.torch import Model, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from .diffusion import LightningModule
from .dataset import Dataset


class LightningTrainer(pl.Trainer, Registrable):
    default_implementation = "default"
LightningTrainer.register("default")(LightningTrainer)


@Step.register("construct_dataset")
class ConstructDataset(Step):
    def run(
        self,
        dataset: Dataset
    ) -> None:
        return dataset


@Step.register("train")
class Train(Step):
    def run(
        self,
        trainer: Lazy[LightningTrainer],
        model: LightningModule,
        dataloader: DataLoader
    ) -> None:
        logger = WandbLogger(
            project="hypergraph-generation",
            save_dir="results/"
        )
        logger.watch(model)
        trainer = trainer.construct(
            logger=logger, 
            log_every_n_steps=5 * len(dataloader),
            default_root_dir="results/ckpts"
        )
        trainer.fit(
            model=model,
            train_dataloaders=dataloader
        )