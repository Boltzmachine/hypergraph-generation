from tango import Step
from tango.common import Registrable
from tango.integrations.torch import Model, DataLoader

import pytorch_lightning as pl

from src.diffusion import LightningModule

class LightningTrainer(pl.Trainer, Registrable):
    """
    """
LightningTrainer.register("default")(LightningTrainer)


@Step.register("train")
class Train:
    def run(
        self,
        trainer: LightningTrainer,
        model: LightningModule,
        dataloader: DataLoader
    ) -> None:
        trainer.fit(
            model=model,
            train_dataloaders=dataloader
        )