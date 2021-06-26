# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy # LightningDeprecationWarning: `pytorch_lightning.metrics.*` module has been renamed to `torchmetrics.*`


class SimpsonsNet(LightningModule):
    def __init__(self):
        super(SimpsonsNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(16*16*32, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _evaluate(self, batch, batch_idx, stage):
        x, y = batch
        out = self.forward(x)
        logits = F.log_softmax(out, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._evaluate(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
