import torch

import pytorch_lightning as pl
import mlflow

class PlModel(pl.LightningModule):
    """
    Lightning module that is compatible
    with PyTorch models included in Deep-NILMtk.
    """

    def __init__(self, net, optimizer='adam', learning_rate=1e-4, patience_optim=5 ):
        super().__init__()
        self.model = net
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.patience_optim = patience_optim


    def forward(self, x,):
        return self.model(x)


    def training_step(self, batch, batch_idx):

        loss, mae = self.model.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # mlflow.log_metrics({
        #     "train_loss":float(loss.detach().cpu().numpy()),
        #     'train_mae':float(mae.detach().cpu().numpy())
        # })

        return loss



    def validation_step(self, batch, batch_idx):
        loss, mae = self.model.step(batch)

        self.model.sample_id = 1
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # mlflow.log_metrics({
        #     "val_loss":float(loss.detach().cpu().numpy()),
        #     'val_mae':float(mae.detach().cpu().numpy())
        # })


    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.


        Returns:
            two lists: a list of optimzer and a list of scheduler
        """

        if self.optimizer == 'adamw':
            optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer  == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer  == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                           patience=self.patience_optim,
                                                           verbose=True, mode="min")
        scheduler = {'scheduler':sched,
                     'monitor': 'val_loss',
                     'interval': 'epoch',
                     'frequency': 1}

        return [optim], [scheduler]

