# IVUS-3D-SEG
> 3D IVUS Dataset and SlidingStripFormer

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)

## 📖 简介

本项目提供了一个用于 3D 血管内超声（IVUS）图像分割的完整解决方案，包括：
- **IVUS-3D-SEG 数据集**：包含 75 条 3D IVUS 序列，标注了外弹力膜（EEM）和管腔（Lumen）
- **SlidingStripFormer 模型**：基于 nnUNet 框架的创新分割架构

> ⚠️ **数据获取声明**: 数据集将在论文录用后公开下载。

### 🏷️ 标注说明

数据集包含以下标注类别：
- **背景 (Background)**: 标签 0
- **外弹力膜 (EEM)**: 标签 1
- **管腔 (Lumen)**: 标签 2

---

## 📑 目录

- [1. 安装](#1-安装)
  - [1.1 环境要求](#11-环境要求)
  - [1.2 安装步骤](#12-安装步骤)
- [2. 数据处理](#2data-processing)
  - [2.1 目录结构](#-21-目录结构)
  - [2.2 快速开始](#-22-快速开始)
  - [2.3 数据集划分](#-23-数据集划分)
- [3. 训练模型](#3-训练模型)
- [4. 预测模型](#4-预测模型)
- [5. 模型评估](#5-模型评估)
- [6. 可视化](#6-可视化)

---

## 1. 安装

### 1.1 环境要求

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.6.0+

### 1.2 安装步骤

```shell
conda create -n nnunet python=3.10
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# nnUnet的安装
cd nnUNet
pip install -e .            
```


## 2.Data Processing

### 📁 2.1 目录结构

项目采用 nnUNet 标准目录组织形式：

```
nnunetData/
├── nnUNet_raw/              # 原始数据
├── nnUNet_preprocessed/     # 预处理后的数据
├── nnUNet_results/          # 训练结果
└── IVUS-3D-SEG/            # 原始 IVUS 数据集
```

### 🚀 2.2 快速开始

#### 选项 1: 使用预处理好的数据

如果你希望快速开始训练，可以直接下载我们预处理好的数据：

- **nnUNet_raw**: [下载链接](https://drive.google.com/file/d/view?usp=drive_link)
- **nnUNet_preprocessed**: [下载链接](https://drive.google.com/file/d/view?usp=drive_link)

#### 选项 2: 从原始数据处理

如果你想从原始数据开始处理，请按照以下步骤操作：

**步骤 1: 下载原始数据集**

从 [Google Drive](https://drive.google.com/file/d/view?usp=drive_link) 下载 IVUS-3D-SEG 数据集，并解压到 `nnunetData/` 目录下。

> 📊 **数据集说明**: 该数据集包含 75 条序列。原始数据中有两条不连续序列已在断点处分割，以保证数据连续性。

**步骤 2: 创建必要的目录**

```bash
cd nnunetData
mkdir -p nnUNet_raw nnUNet_preprocessed nnUNet_results
```


你需要设置 nnUNet 的环境变量，告诉 nnUNet 数据存储的位置：

```bash
export nnUNet_raw="/path/to/your/nnunetData/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/nnunetData/nnUNet_preprocessed"
export nnUNet_results="/path/to/your/nnunetData/nnUNet_results"
```

> 💡 **提示**: 将上述路径替换为你实际的绝对路径。建议将这些环境变量添加到 `~/.bashrc` 或 `~/.zshrc` 中，以便永久生效。

**步骤 3: 转换为 nnUNet 格式**

运行数据转换脚本，将原始数据组织为 nnUNet 标准格式：

```bash
python DataProcess/convert_to_nnunet_use.py
```

该脚本会自动：
- 根据预定义的训练/验证/测试集划分复制文件
- 重命名文件为 nnUNet 标准命名格式
- 生成数据复制报告（`copy_report.txt`）

**步骤 4: 生成数据集配置文件**

运行以下脚本生成 `dataset.json`：

```bash
python nnUNet/nnunetv2/dataset_conversion/Dataset789_ultrasound.py
```

**步骤 5: 数据预处理**

使用 nnUNet 内置命令进行数据预处理：

```bash
nnUNetv2_plan_and_preprocess -d 789 -c 3d_lowres 3d_fullres -np 16
```

### 📝 2.3 数据集划分

数据集划分信息存储在 `DataProcess/split_result/` 目录中：
- `train_cases.txt`: 训练集样本列表
- `val_cases.txt`: 验证集样本列表  
- `test_cases.txt`: 测试集样本列表




## 3. 训练模型

使用 SlidingStripFormer 训练器进行模型训练：

```bash
nnUNetv2_train 789 3d_lowres 0 -tr nnUNetTrainer_StripFormer
```

> 💡 **提示**: 训练过程将自动保存 checkpoint 到 `nnUNet_results/Dataset789_ultrasound/nnUNetTrainer_StripFormer__nnUNetPlans__3d_lowres/fold_0/`



---

## 4. 预测模型

使用训练好的模型对测试集进行预测：

```bash
nnUNetv2_predict -i nnunetData/nnUNet_raw/Dataset789_ultrasound/imagesTs \
                 -o evaluation/pred/nnUNetTrainer_StripFormer \
                 -d 789 \
                 -c 3d_lowres \
                 -f 0 \
                 -chk checkpoint_best.pth \
                 -tr nnUNetTrainer_StripFormer
```

预测结果将保存为 NIfTI 格式（`.nii.gz`）文件。

---

## 5. 模型评估

运行评估脚本计算各项性能指标：

```bash
python evaluation/evaluation.py \
       --gt_dir nnunetData/nnUNet_raw/Dataset789_ultrasound/mask \
       --pred_dir evaluation/pred/nnUNetTrainer_StripFormer \
       --output_dir evaluation/results
```

### 5.1 评估指标

该脚本会计算以下指标：
- **Dice 系数**: 衡量分割重叠度
- **Hausdorff 距离 (HD)**: 衡量边界距离
- **百分比面积差 (PAD)**: 衡量面积差异
- **交并比 (IoU)**: 衡量区域重叠

评估结果将保存为 JSON 文件，包含每个样本的详细指标和平均性能。

---

## 6. 可视化

运行可视化脚本生成分割结果的可视化图像：

运行`visual/visual_all.py`以生成可视化结果。

---

## 🙏 致谢

本项目基于 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 框架开发，感谢原作者的贡献。

