## TripletMAML: A metric-based model-agnostic meta-learning algorithm for few-shot classification
TripletMAML implements a novel few-shot learning approach by integrating triplet-loss-based metric learning within the Model-Agnostic Meta-Learning (MAML) framework. This repository supports both classification and retrieval tasks while providing reproducible, transparent, and well-documented experiments.

---

## Table of Contents

- [Citation](#citation)
- [Overview](#overview)
- [Features](#features)
- [Installation & Dependencies](#installation--dependencies)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Detailed Documentation](#detailed-documentation)
  - [Algorithm Details](#algorithm-details)
  - [Code Modules](#code-modules)
  - [Step-by-Step Usage Guides](#step-by-step-usage-guides)
- [Permanent Links & Reproducibility](#permanent-links--reproducibility)




## Citation
If you find TripletMAML useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@misc{gulcu2025tripletmaml,
  author =       {Ayla Gülcü, Zeki Kuş, İsmail Taha Samed Özkan and Osman Furkan Karakuş},
  title =        {TripletMAML: A metric-based model-agnostic meta-learning algorithm for few-shot classification},
  howpublished = {\url{https://github.com/aylagulcu/TripletMAML}},
  journal =      {The Visual Computer}
  year =         {2025}
}
```

## Overview

TripletMAML leverages the strengths of both meta-learning and metric-based learning to tackle the challenges of few-shot classification. By incorporating a triplet loss into the MAML framework, the algorithm learns robust and discriminative feature representations even when data is scarce. This repository includes:

- **Classification Tasks:** Training and evaluation scripts for few-shot classification.
- **Retrieval Tasks:** Scripts to test the retrieval capability using learned embeddings.
- **Interactive Experimentation:** A Jupyter Notebook for task control and hyperparameter tuning.

---

## Features

- **Few-Shot Learning:** Designed specifically for few-shot scenarios.
- **Metric-Based Learning:** Uses triplet loss to improve representation learning.
- **Meta-Learning Framework:** Builds on MAML for fast adaptation to new tasks.
- **Comprehensive Dataset Support:** Includes scripts and instructions for Omniglot, MiniImageNet, CUB-200-2011, and CIFAR Few-Shot.
- **Detailed Documentation:** Clear instructions, algorithm details, and permanent link assurance for reproducibility.

---

## Installation & Dependencies

### Prerequisites

- **Operating System:** Ubuntu 20.04 LTS is recommended.
- **Python Version:** 3.9.7 (Anaconda distribution is suggested).

### Required Libraries

- `python` == 3.9.7  
- `learn2learn` == 0.1.7  
- `numpy` == 1.20.3  
- `pytorch` == 1.10.2 (compatible with py3.9_cuda11.3_cudnn8.2.0_0)  
- `scikit-learn` == 1.2.0  

### Setup Instructions

```bash
git clone https://github.com/aylagulcu/TripletMAML.git
cd TripletMAML
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---



## Dataset Preparation
![Datasets Visualization](./Images/Datasets.png)
| Dataset | Download | Download Needed |
| ---     | ---      | ---             |
| Omniglot | [download](https://github.com/brendenlake/omniglot) | NO |
| MiniImageNet | [download](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) | NO |
| CUB-200-2011 | [download](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) | YES |
| CIFAR Few-Shot | [download](https://drive.google.com/u/1/uc?id=1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI&export=download) | YES |

- for Omniglot
  ```
  TripletOmniglot.py Files will handle downloading automatically if download mode is set as True.
  ```
- for MiniImageNet
  ```
  ./data/MiniImageNet/MiniImageNet_downloader.py will handle downloading test-train-validation files accordingly. 
  No needed to install another dataset.
  ```
- for CUB-200-2011
  ```
  CUB-200-2011 files are needed to be downloaded from given link. Then replaced into corresponding folder which is ./data/CUB/
  Choose the generator of your preference so they can create the test-train-validation files accordingly.
  ```
- for CIFAR Few-Shot
  ```
  CIFAR Few-Shot files are needed to be downloaded from given link. Then replaced into corresponding folder which is ./data/CIFARFS100/
  First run the processor then you can use generator to create the test-train-validation files accordingly.
  ```

## Usage

### Classification
```bash
python TripletMAML/maml_triplet_train_test_val.py
```

### Retrieval
```bash
python TripletMAML/maml_triplet_test_retrieval.py
```

### Interactive Notebook
Open:
```bash
TripletMAML/TaskControl.ipynb
```


## Directory Structure
```
|—— .gitignore
|—— data
|    |—— Generators
|        |—— CIFARFS100
|            |—— CIFARFS100_generator.py
|            |—— CIFARFS100_processor.py
|        |—— CUB
|            |—— CUB_BB_NoResize_generator.py
|            |—— CUB_BB_Resize_generator.py
|            |—— CUB_NoBB_generator.py
|        |—— Flowers
|            |—— Flowers_generator.py
|        |—— MiniImageNet
|            |—— MiniImageNet_downloader.py
|—— HPO
|    |—— backbone.py
|    |—— de.py
|    |—— losses.py
|    |—— model.py
|    |—— rs.py
|    |—— train.py
|    |—— Triplets
|        |—— TripletCUB.py
|        |—— TripletFlowers.py
|        |—— TripletFSCIFAR100.py
|        |—— TripletMiniImageNet.py
|        |—— TripletOmniglot.py
|        |—— __init__.py
|—— TripletMAML
|    |—— backbone.py
|    |—— losses.py
|    |—— maml_triplet_test_retrieval.py
|    |—— maml_triplet_train_test_val.py
|    |—— TaskControl.ipynb
|    |—— Triplets
|        |—— TripletCUB.py
|        |—— TripletFlowers.py
|        |—— TripletFSCIFAR100.py
|        |—— TripletMiniImageNet.py
|        |—— TripletOmniglot.py
|        |—— __init__.py
```

## Detailed Documentation

### Algorithm Details

- **Triplet Loss:** Enhances feature separability across classes.
- **MAML:** Allows quick adaptation using meta-learning.
- **Integration:** Triplet loss is incorporated in the inner loop of MAML.

### Code Modules

- `backbone.py`: Defines the neural network architecture.
- `losses.py`: Implements the triplet loss.
- `maml_triplet_train_test_val.py`: Training and validation logic.
- `maml_triplet_test_retrieval.py`: Retrieval evaluation logic.

### Step-by-Step Usage

1. Prepare dataset.
2. Configure parameters in the relevant script.
3. Run the experiment (see Usage section).

---

## Permanent Links & Reproducibility

All dataset and code links are verified to be permanent. This documentation is designed for full reproducibility by other researchers.