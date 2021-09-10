import torch

import pytorch_lightning as pl


class pilModel(pl.LightningModule):
    """
    Lightning module that is compatible 
    with PyTorch models included in Deep-NILMtk.
    """
    
    def __init__(self, net, hparams):
        super().__init__()
        self.q=torch.tensor(hparams['quantiles'])
        self.model = net
        self.hparams.update(hparams)
        
    def forward(self, x,):
        return self.model(x)

    
    def training_step(self, batch):
        
        loss, mae = self.model.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    
    def validation_step(self, batch):
        
        loss, mae = self.model.step(batch)
        self.model.sample_id = 1
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
    def configure_optimizers(self): 
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization. 
        

        Returns:
            two lists: a list of optimzer and a list of scheduler
        """
        
        if self.hparams['optimizer'] == 'adamw':
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams['learning_rate'])
        elif self.hparams['optimizer'] == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        elif self.hparams['optimizer'] == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams['learning_rate'], momentum=0.9)
        else:
            raise ValueError
                
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                           patience=self.hparams['patient'], 
                                                           verbose=True, mode="min")
        scheduler = {'scheduler':sched, 
                 'monitor': 'val_mae',
                 'interval': 'epoch',
                 'frequency': 1}
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optim], [scheduler]
    
    
