import os
import glob
from pytorch_lightning.loggers import TensorBoardLogger

class DictLogger(TensorBoardLogger):
    """PyTorch Lightning `dict` logger."""

    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/50881c0b31/pytorch_lightning/logging/base.py

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def log_hyperparams(*args, **kwargs):
        # We will do this manually with final metrics
        pass

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)


def get_latest_checkpoint(checkpoint_path):
    """
    Returns the latest checkpoint for the model
    Args:
        checkpoint_path (str): The path to the checkpoints' folder

    Returns:
        file: the latest checkpoint saved during training
    """
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + '/*.ckpt')
    
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file

