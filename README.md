Pytorch Classification Experiment Platform
===
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

# Table of Contents
[TOC]

# Introduction
This is a cinputer vision deep learning experimental platform for Pytorch. With just one line of Linux command you can do the following: 
* Training neural network for muilti classifion task.
* Training with muilti gpus.
* Use acc, precision, recall, f1 score, confusion matrix to evaluate the model.
* Use tensorboard to record and visualize model performance.
* Save each checkpoint.
* Save the best performing model.
* Save the prediction result for each evaluation.

In addition, this platform is flexible:
* Easy to change other datasets.
* Modify `net.py` to change or customize your network.
* Modify output layer in `net.py` and loss function in `loss.py` to change the task to single classification or regression task.


> [name=weidaolee]

# Requirements
Python 3.6.0 or later with all of the `env/requirments.txt`
* `torch==1.2.0`
* `torchvision==0.4.0`
* `tensorboardX==1.8`
* `tensorboard==1.14.0`

# Installation
Clone and install requirements
```
$ git clone https://github.com/weidaolee/classification_platform.git
$ pip install -r env/requirements.txt
```

Modify configs and paths
```
$ vim config/defualt.cfg
```
# Training
```
$ python train.py -h    ## show this help message and exit
$ train.py [-h] [--prefix PREFIX] [--gpu GPU] [--cfg CFG]
                [--weights_path WEIGHTS_PATH] [--n_cpu N_CPU]
                [--checkpoint CHECKPOINT] [--checkpoint_dir CHECKPOINT_DIR]
                [--tfboard TFBOARD]`
```

# Tensorboard
```
$ cd logs
$ tensorboard [--logdir PREFIX] [--port PORT]
```

# Appendix and FAQ


**Find this document incomplete?** Leave a comment!


###### tags: `pytorch` `classification` `mulit-class` `regression`
