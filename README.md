# [Sample Selection for Fair and Robust Training](https://openreview.net/forum?id=IZNR0RDtGp3)

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS), 2021

----------------------------------------------------------------------

## Introduction
The goal of the project is to perform model selection and evaluation of a research work, for the
*Model Selection for Large Scale Learning* course at Grenoble INP - Ensimag, year 2021/2022.

This directory is for simulating fair and robust sample selection on the 
synthetic dataset. The program needs PyTorch, Jupyter Notebook, and CUDA.

## Project organization

The directory contains a total of 6 files and 2 child directory:

- this [README](./README.md)
- 4 python files:
	- [`FairRobustSampler.py`](./FairRobustSampler.py) defines the FairRobust sampler and a PyTorch dataset for sensitive data
	- [`models.py`](./models.py) contains logistic regression and SVM architecture, a test function and a plotting function
	- [`utils.py`](./utils.py) contains utility functions for data generation
	- [`main.py`](./main.py)
- a [report](./report.ipynb) as jupyter notebook
- [`synthetic_data`](./synthetic_data) contains 11 numpy files for synthetic data. The synthetic data is composed of training set, validation set, and test set.
- [`datasets`](./datasets) contains a `xls` file, related to the real [credit card clients dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Description

To simulate the algorithm, please use the [jupyter notebook](./report.ipynb), which contains detailed instructions, or [`main.py`](./main.py).

The jupyter notebook will load the data and train the models with two 
different fairness metrics: equalized odds and demographic parity.

Each training utilizes the FairRobust sampler.
The PyTorch dataloader serves the batches to the model via the FairRobust sampler described in the paper.
After the training, the test accuracy and fairness will be shown.
