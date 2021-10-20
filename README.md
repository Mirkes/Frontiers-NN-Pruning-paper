# Frontiers-NN-Pruning-paper
Datasets and codes for Frontiers NN pruning paper

This repository contains two folders, each of which contains code and data for one dataset
Derivation of explicit formulae for the first order sensitivity indicators for used type of networks is presented in file Formulae derivation.docx

### folder MNIST contains used fragment of MNIST dataset and code for work with this dataset:
* DigitDataSetSplit.mat contains used fragment of MNIST dataset and indices for train, validation and test sets.
* NN_work.m is the main script developed and notebook with separate sections each of which can be started by ctrl+Enter.
* myNN is automatically generated function which implement NN.
* myNN0 is version of myNN which allow to change weights of hidden layer.
* myNNDif is version of myNN which allows to change weights of hidden layer and calculates the first order sensitivity indicators for input signals.
* weights.mat file with weights matrix of hidden layer used in this study.

### folder BreastCancer contains breast cancer dataset and code for work with this dataset:
* wdbc.data is file downloaded from [UCI repository](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
* BCDataSetSplit.mat contains used fragment of MNIST dataset and indices for train, validation and test sets.
* NN_work.m is the main script developed and notebook with separate sections each of which can be started by ctrl+Enter.
* myNN is automatically generated function which implement NN.
* myNN0 is version of myNN which allow to change weights of hidden layer.
* myNNDif is version of myNN which allows to change weights of hidden layer and calculates the first order sensitivity indicators for input signals.
* weights.mat file with weights matrix of hidden layer used in this study.
