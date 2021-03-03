# :detective::robot: Monitoring a PyTorch Lightning CNN with Weights & Biases

- https://arxiv.org/pdf/1409.1556v6.pdf

---

## :closed_book: Table of Contents

- [:hammer_and_wrench: Requirements](#hammer_and_wrench-requirements)
- [:open_file_folder: Dataset](#open_file_folder-dataset)
- [:robot: Modelling](#robot-modelling)
- [:detective: Monitoring](#detective-monitoring)
- [:computer: Credits](#computer-credits)
- [:crystal_ball: Future Tasks](#crystal_ball-future-tasks)

---

## :hammer_and_wrench: Requirements

...

```
pip install -r requirements.txt
```

__Note__. If you are using Jupyter Lab, either on a local environment or hosted on AWS, Azure or GCP, you will 
need to install the following Jupyter Lab extensions so as to see the training progress bar in your Notebook, otherwise
you will just see a text similar to: `HBox(children=(FloatProgress(value=0.0, ...)`.

If you are using conda you will need to install nodejs first, and the proceed with the next steps. If you are not
using conda just skip this step.

```
conda install nodejs
```

And then install and activate the following Jupyter Lab widget so that you can see the tqdm progress bar properly
in your Notebook, while the PyTorch Lightning model is being trained.

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter nbextension enable --py widgetsnbextension
```

---

## :open_file_folder: Dataset

The dataset that is going to be used to train the image classification model is 
"[The Simpsons Characters Data](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)", 
which is Kaggle dataset that contains images of some of the main The Simpsons characters.

The original dataset contains 42 classes of The Simpsons characters, with an unbalanced number of samples per 
class, and a total of 20,935 training images and 990 test images in JPG format.

bla bla bla

Find all the information about the dataset in [dataset/README.md](https://github.com/alvarobartt/serving-tensorflow-models/tree/master/dataset).

![]()

---

## :robot: Modelling

PyTorch Lightning bla bla bla

```python
import torch
import torch.nn as nn

class SimpsonsNet(nn.Module):
    def __init__(self):
        super(SimpsonsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 32 x 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 32 x 32
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 16 x 16
        self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(16*16*32, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
```

Which translated to PyTorch Lightning it ends up as easy as:

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy


class SimpsonsNet(LightningModule):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*16*32, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
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
```

And then the trainer...

```python
import pytorch_lightning as pl

trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=10, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
```

bla bla bla

---

## :detective: Monitoring

WandB register, creating project, monitoring, installation, bla bla bla

__Note__. Both PyTorch Lightning and Weights & Biases log directories are included in the `.gitignore` file, which means
that the logs will not be updated to GitHub, feel free to remove those lines so that GIT does not ignore these directories.
Anyway as you are using Weights & Biases, the logs will be stored there, so there's no need to store them locally.

---

## :computer: Credits

Credits to [Alexandre Attia](https://github.com/alexattia) creating [The Simpsons Characters Dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset), 
as well as to the Kaggle community that made it possible, as they included a lot of manually curated images to the 
original dataset that scaled from 20 characters originally to 42.

Credits to [Lezwon Castelino](https://github.com/lezwon) for solving the PyTorch Lightning progress bar issue as he 
nicely provided a solution to the issue in [this PyTorch Lightning issue](https://github.com/PyTorchLightning/pytorch-lightning/issues/1112)
sharing the following [StackOverflow post](https://stackoverflow.com/questions/60656978/display-tqdm-in-aws-sagemakers-jupyterlab).

Last but not least, credits to [Charles Frye](https://github.com/charlesfrye) for creating and explaining in detail the integration
of Weights & Biases with the PyTorch Lightning training in the [PyTorch Lightning + W&B example](https://github.com/wandb/examples/blob/master/colabs/pytorch-lightning/Supercharge_your_Training_with_Pytorch_Lightning_%2B_Weights_%26_Biases.ipynb).

---

## :crystal_ball: Future Tasks

- [ ] 