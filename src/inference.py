# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

from __future__ import absolute_import

import torch

from model import SimpsonsNet


def model_inference():
    model = SimpsonsNet().load_from_checkpoint('wandb/latest-run/files/ml-monitoring-with-wandb/2fzailcj/checkpoints/epoch=18-step=3875.ckpt')

    x = torch.randn((1, 3, 32, 32))
    y = model(x)

    pred = torch.argmax(y, dim=1)
    print(pred.item())


if __name__ == '__main__':
    model_inference()
