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