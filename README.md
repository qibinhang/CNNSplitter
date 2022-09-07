# Patching Weak Convolutional Neural Network Models through Modularization and Composition

## Abstract
This repository includes the code and experimental data in our paper entitled 
"Patching Weak Convolutional Neural Network Models through Modularization and Composition". 

In this paper, we propose a structured modularization approach, CNNSplitter, which decomposes a strong CNN model for $N$-class classification into $N$ CNN modules. 
Each module is a sub-model containing a part of the convolution kernels of the strong model. 
To patch a weak CNN model with low performance on a target class, we compose the weak CNN model with the corresponding module obtained from a strong CNN model. 
The ability of the weak CNN model to recognize the target class can thus be improved through patching.


## Requirements
+ python 3.8.10<br>
+ pytorch 1.8.1<br>
+ numpy 1.19.2<br>
+ tqdm 4.61.0<br>
+ matplotlib 3.4.2<br>
+ seaborn 0.11.1<br>
+ GPU with CUDA support is also needed


## How to install
Install the dependent packages via pip:

    $ pip install numpy==1.19.2 tqdm==4.61.0 matplotlib==3.4.2 seaborn==0.11.1
    
Install pytorch according to your environment, see https://pytorch.org/.


## How to modularize a trained CNN model
1. modify `global_configure.py` to set the `root_dir`.
2. run `python train.py --model simcnn --dataset cifar10` to get the pre-trained model `SimCNN-CIFAR`.
3. run `python kernel_importance_analyzer.py --model simcnn --dataset cifar10` in directory `preprocess/` to get the importance of each kernel in `SimCNN-CIFAR`.
4. run `python run_layer_sensitivity_analyzer.py --model simcnn --dataset cifar10` in directory `scripts/` to analyze the sensitivity of `SimCNN-CIFAR`.
5. modify `configures/simcnn_cifar10.py` to set the configures of GA searching.
6. run `python module_recorder.py --model simcnn --dataset cifar10`.
7. run `python module_explorer.py --model simcnn --dataset cifar10 --target_class 0` with 10 instances in parallel (`--target_class` from 0 to 9), each of which searches for one class. 

We provide the four trained CNN models and the corresponding modules, as well as the weak models.\
One can download `data/` from [here](https://mega.nz/folder/UhtBCIZI#zZMWav7aJMHvEMYDdZ8IIg) and reuse a module to patch a weak CNN model following the description below. 

## How to patch a weak CNN model
### preparing
1. run `python module_output_collector.py --model simcnn --dataset cifar10` in directory `preprocess/` to collect the outputs of 10 modules.

### Patching a simple model
1. run `python train.py --model simcnn --dataset cifar10` in directory `experiments/patch/patch_for_weak_model` to train an overly simple SimCNN-CIFAR.
2. run `python apply_patch.py --model simcnn --dataset cifar10 --exp_type weak --target_class 0 --target_epoch 99` in directory `experiments/patch` to patch the simple SimCNN-CIFAR.

### Patching an overfitting/underfitting model
1. run `python train.py --model simcnn --dataset cifar10` in directory `experiments/patch/patch_for_poor_model` to train an overfitting/underfitting SimCNN-CIFAR.
2. run `python apply_patch.py --model simcnn --dataset cifar10 --exp_type poor_fit --target_class 0 --target_epoch 169` in directory `experiments/patch` to patch the overfitting SimCNN-CIFAR. 
3. run `python apply_patch.py --model simcnn --dataset cifar10 --exp_type poor_fit --target_class 0 --target_epoch 84` in directory `experiments/patch` to patch the underfitting SimCNN-CIFAR.
