# CSC3009 Machine Learning Group 04 Project

## Overview

This project aims to develop and evaluate open-source machine-learning models using deep learning techniques for classifying brain MRI images into four distinct categories: glioma, meningioma, notumor and pituitary.

## Dataset

Dataset used for this project is under `DATASETS/dataset_4` folder.

## Models Used

1. DenseNet169
2. InceptionV3 (GoogleNet)
3. MobileNet
4. NasNetLarge
5. VGG19

We have attempted other models as well under `Other Model Attempts`. However, we have chosen the above five as the main focus for this project based on the results.

## Installation Guide

### 1. Create a conda environment

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

If running to error message "partially initialized module 'charset_normalizer'" run the following command

```bash
pip install -U --force-reinstall charset-normalizer
```

## Team Members

- Chia Keng Li (2102718)
- Dylan Tok Hong Xun (2101372)
- Goh Yee Kit (2100649)
- Kwok Jun Peng Derick (2100689)
- Zhang XiangHui (2101993)
