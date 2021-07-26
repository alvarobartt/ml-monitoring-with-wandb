# :computer: Code

This directory contains all the source files required to train the model and reproduce
what was explained in the [README.md](https://github.com/alvarobartt/ml-monitoring-with-wandb/blob/master/README.md)
with just minor changes required.

All the code and files are pretty self explanatory, that's why we won't get into much detail
in this section unless requested.

So on, to train the presented PyTorch Lightning model using its trainer and monitor it with Weights and Biases,
you just need to use the following command:

```bash
python train.py --batch-size 32 --epochs 10
```

:pushpin: __Note__. You can freely tweak that parameters as well as include new ones, since the code
is pretty simple and easy to modify to fit your own scenario, if applicable.
