# IVUS-3D-SEG
> 3D IVUS Dataset and SlidingStripFormer

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)

[中文版本](README_zh.md)

## 📖 Introduction

This project provides a complete solution for 3D Intravascular Ultrasound (IVUS) image segmentation, including:
- **IVUS-3D-SEG Dataset**: Contains 75 3D IVUS sequences with annotations for External Elastic Membrane (EEM) and Lumen
- **SlidingStripFormer Model**: An innovative segmentation architecture based on the nnUNet framework

> ⚠️ **Data Availability Notice**: The dataset will be made publicly available after paper acceptance.

### 🏷️ Annotation Description

The dataset contains the following annotation categories:
- **Background**: Label 0
- **External Elastic Membrane (EEM)**: Label 1
- **Lumen**: Label 2

---

## 📑 Table of Contents

- [1. Installation](#1-installation)
  - [1.1 Requirements](#11-requirements)
  - [1.2 Installation Steps](#12-installation-steps)
- [2. Data Processing](#2-data-processing)
  - [2.1 Directory Structure](#-21-directory-structure)
  - [2.2 Quick Start](#-22-quick-start)
  - [2.3 Dataset Split](#-23-dataset-split)
- [3. Model Training](#3-model-training)
- [4. Model Prediction](#4-model-prediction)
- [5. Model Evaluation](#5-model-evaluation)
- [6. Visualization](#6-visualization)

---

## 1. Installation

### 1.1 Requirements

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.6.0+

### 1.2 Installation Steps

```shell
conda create -n nnunet python=3.10
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# nnUnet Installation
cd nnUNet
pip install -e .            
```

## 2. Data Processing

### 📁 2.1 Directory Structure

The project follows the nnUNet standard directory organization:

```
nnunetData/
├── nnUNet_raw/              # Raw data
├── nnUNet_preprocessed/     # Preprocessed data
├── nnUNet_results/          # Training results
└── IVUS-3D-SEG/            # Original IVUS dataset
```

### 🚀 2.2 Quick Start

#### Option 1: Use Preprocessed Data

If you want to start training quickly, you can download our preprocessed data directly:

- **nnUNet_raw**: [Download Link](https://drive.google.com/file/d/1vpoVBetn6P8jIRzMpe3HPEWAACbA_dhv/view?usp=drive_link)
- **nnUNet_preprocessed**: [Download Link](https://drive.google.com/file/d/1qQrl2_i9AgZPB8Zl0YK18TzKz1buQxLh/view?usp=drive_link)

#### Option 2: Process from Raw Data

If you want to process from raw data, please follow these steps:

**Step 1: Download Raw Dataset**

Download the IVUS-3D-SEG dataset from [Google Drive](https://drive.google.com/file/d/1VZJ_5eK1a53ddEsfZ-UStWdfamFkfy4G/view?usp=drive_link) and extract it to the `nnunetData/` directory.

> 📊 **Dataset Note**: The dataset contains 75 sequences. Two discontinuous sequences in the raw data have been split at the breakpoints to ensure data continuity.

**Step 2: Create Necessary Directories**

```bash
cd nnunetData
mkdir -p nnUNet_raw nnUNet_preprocessed nnUNet_results
```

You need to set nnUNet environment variables to tell nnUNet where to store data:

```bash
export nnUNet_raw="/path/to/your/nnunetData/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/nnunetData/nnUNet_preprocessed"
export nnUNet_results="/path/to/your/nnunetData/nnUNet_results"
```

> 💡 **Tip**: Replace the above paths with your actual absolute paths. It is recommended to add these environment variables to `~/.bashrc` or `~/.zshrc` for permanent effect.

**Step 3: Convert to nnUNet Format**

Run the data conversion script to organize raw data into nnUNet standard format:

```bash
python DataProcess/convert_to_nnunet_use.py
```

This script will automatically:
- Copy files according to the predefined train/val/test split
- Rename files to nnUNet standard naming format
- Generate a data copy report (`copy_report.txt`)

**Step 4: Generate Dataset Configuration File**

Run the following script to generate `dataset.json`:

```bash
python nnUNet/nnunetv2/dataset_conversion/Dataset789_ultrasound.py
```

**Step 5: Data Preprocessing**

Use nnUNet built-in command for data preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 789 -c 3d_lowres 3d_fullres -np 16
```

### 📝 2.3 Dataset Split

Dataset split information is stored in the `DataProcess/split_result/` directory:
- `train_cases.txt`: List of training cases
- `val_cases.txt`: List of validation cases
- `test_cases.txt`: List of test cases

## 3. Model Training

Train the model using the SlidingStripFormer trainer:

```bash
nnUNetv2_train 789 3d_lowres 0 -tr nnUNetTrainer_StripFormer
```

> 💡 **Tip**: The training process will automatically save checkpoints to `nnUNet_results/Dataset789_ultrasound/nnUNetTrainer_StripFormer__nnUNetPlans__3d_lowres/fold_0/`

---

## 4. Model Prediction

Predict on the test set using the trained model:

```bash
nnUNetv2_predict -i nnunetData/nnUNet_raw/Dataset789_ultrasound/imagesTs \
                 -o evaluation/pred/nnUNetTrainer_StripFormer \
                 -d 789 \
                 -c 3d_lowres \
                 -f 0 \
                 -chk checkpoint_best.pth \
                 -tr nnUNetTrainer_StripFormer
```

Prediction results will be saved in NIfTI format (`.nii.gz`).

---

## 5. Model Evaluation

Run the evaluation script to calculate performance metrics:

```bash
python evaluation/evaluation.py \
       --gt_dir nnunetData/nnUNet_raw/Dataset789_ultrasound/mask \
       --pred_dir evaluation/pred/nnUNetTrainer_StripFormer \
       --output_dir evaluation/results
```

### 5.1 Evaluation Metrics

This script will calculate the following metrics:
- **Dice Coefficient**: Measures segmentation overlap
- **Hausdorff Distance (HD)**: Measures boundary distance
- **Percentage Area Difference (PAD)**: Measures area difference
- **Intersection over Union (IoU)**: Measures region overlap

Evaluation results will be saved as a JSON file, containing detailed metrics for each sample and average performance.

---

## 6. Visualization

Run the visualization script to generate visualized images of segmentation results:

Run `visual/visual_all.py` to generate visualization results.

## 🙏 Acknowledgements

This project is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. Thanks to the original authors for their contribution.
