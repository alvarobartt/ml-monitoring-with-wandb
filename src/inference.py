# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

from __future__ import absolute_import

import torch

from model import SimpsonsNet
from mnist import SimpsonsMNISTDataModule, IDX2CLASS


def model_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpsonsNet().load_from_checkpoint('wandb/latest-run/files/ml-monitoring-with-wandb/2fzailcj/checkpoints/epoch=18-step=3875.ckpt')
    model = model.to(device)
    model.eval();

    data = SimpsonsMNISTDataModule(dataset_path="../dataset", batch_size=5)
    test_data = data.test_dataloader()
    x, target = next(iter(test_data))
    x = x.to(device)
    
    with torch.no_grad():
        y = model(x)

    pred = torch.argmax(y, dim=1)
    print([IDX2CLASS[idx] for idx in pred.cpu().numpy()])
    print([IDX2CLASS[idx] for idx in target.numpy()])


if __name__ == '__main__':
    model_inference()
