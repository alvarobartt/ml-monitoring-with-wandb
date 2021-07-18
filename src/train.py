# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

from __future__ import absolute_import

import click

import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch

from mnist import SimpsonsMNISTDataModule
from model import SimpsonsNet


@click.command()
@click.option('-b', '--batch-size', required=True, type=int)
@click.option('-e', '--epochs', required=True, type=int)
def train(batch_size: int, epochs: int):
    """Trains a PyTorch Lightning model using Weights and Biases"""

    # Instantiate the model
    model = SimpsonsNet()

    # Make sure that the Tensor shapes for the model's input/output work as expected
    try:
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == torch.Size([1, 10])
    except Exception as e:
        raise e

    # Instantiate the LightningDataModule
    data_module = SimpsonsMNISTDataModule(dataset_path="../dataset", batch_size=batch_size)

    # Load the DataLoaders for both the train and validation datasets
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create the configuration of the current run
    wandb_config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'layers': len(list(filter(lambda param: param.requires_grad and len(param.data.size()) > 1, model.parameters()))),
        'parameters': sum(param.numel() for param in model.parameters() if param.requires_grad),
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'dataset': 'Simpsons-MNIST',
        'dataset_train_size': len(data_module.train_image_folder),
        'dataset_val_size': len(data_module.val_image_folder),
        'input_shape': '[3,32,32]',
        'channels_last': False,
        'criterion': 'CrossEntropyLoss',
        'optimizer': 'Adam'
    }

    # Init the PyTorch Lightning WandbLogger (you need to `wandb login` first!)
    wandb_logger = WandbLogger(project='ml-monitoring-with-wandb', job_type='train', config=wandb_config)

    # Instantiate the PyTorch Lightning Trainer and fit the model
    trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=10, max_epochs=epochs, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)

    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    train()
