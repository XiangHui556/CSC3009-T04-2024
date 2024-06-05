# Installation Guide

## 1. Create a conda environment

Instructions for creating a conda environment, setting up the GPU (if applicable), and installing TensorFlow along with its dependencies.

Create a new conda environment named tf with the following command:

```bash
conda create --name tf python=3.9
```

Activate the conda environment with the following command:

```bash
conda deactivate
conda activate tf
```

Install TensorFlow with the following command:
(You can skip this part if you are not usinng GPU)
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Install the dependencies with the following command:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
