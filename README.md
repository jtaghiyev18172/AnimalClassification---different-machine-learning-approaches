# Animal Classification - Benchmarking Machine Learning Approaches for Animal Image Classification

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-green)
![MLflow](https://img.shields.io/badge/MLflow-experiment_tracking-purple)
![Status](https://img.shields.io/badge/status-work_in_progress-yellow)

A structured experimental pipeline for **animal image classification** comparing:

- classical computer vision approaches
- deep feature extraction
- custom CNN models trained from scratch

The goal of this project is to **systematically benchmark different modeling strategies** under a shared dataset split and transformation pipeline.

The repository is designed to be **reproducible, modular, and experiment-tracked**, allowing fair comparisons between approaches.

---

# Overview

This project investigates how different machine learning paradigms perform on the same classification task:

1. **Handcrafted feature pipelines**
2. **Deep feature extraction using pretrained models**
3. **CNN architectures trained from scratch**

All experiments share:

- a **fixed dataset split**
- a **common transformation pipeline**
- centralized **experiment tracking**
- standardized **metrics and reporting**

The task is a **3-class image classification problem**:

| Class |
|------|
| cats |
| dogs |
| wildlife |

---

# Dataset

The dataset is organized using a deterministic split called:

```
split_v1
```

Dataset sizes:

| Split | Samples |
|------|-------|
| Train | 50,127 |
| Validation | 6,266 |
| Test | 6,266 |

Class distribution:

| Split | Cats | Dogs | Wildlife |
|------|------|------|------|
| Train | 18,954 | 18,315 | 12,858 |
| Validation | 2,369 | 2,290 | 1,607 |
| Test | 2,370 | 2,289 | 1,607 |

The split manifests are stored as:

```
data/splits/split_v1/
```

Files:

```
train.csv
val.csv
test.csv
classes.json
```

Each CSV contains:

```
filepath,label
```

---

# Image Transformations and Augmentation

All models rely on the shared transformation configuration:

```
configs/transforms_v1.yaml
```

Two transformation pipelines are defined.

---

## Training Transform Pipeline

Identifier:

```
transforms_v1_train_runtime_aug
```

This pipeline includes runtime augmentation to improve generalization.

Typical operations include:

- random horizontal flips
- random cropping
- random resizing
- color normalization
- tensor conversion

Example transformation flow:

```
Image
↓
Random Resize
↓
Random Horizontal Flip
↓
Random Crop
↓
To Tensor
↓
Normalize (ImageNet statistics)
```

---

## Evaluation Transform Pipeline

Identifier:

```
transforms_v1_eval_resize256_centercrop224_imagenetnorm
```

This pipeline is deterministic.

```
Image
↓
Resize (256)
↓
Center Crop (224)
↓
To Tensor
↓
Normalize (ImageNet mean/std)
```

---

## Example Transformations

Example visualization placeholders:

```
docs/images/transform_examples/
```

Example grid showing:

- original image
- augmented variants
- evaluation transform

```
docs/images/augmentation_grid.png
```

---

# Experiment Tracking

All experiments are tracked using **MLflow**.

Tracking directory:

```
mlruns/
```

Each training run logs:

- parameters
- metrics
- artifacts
- configuration

Example run contents:

```
params
metrics
artifacts
config.json
```

---

# Models Implemented

The project includes multiple model families.

---

# 1 — Classical Computer Vision Pipelines

These models use **handcrafted feature extractors** combined with classical machine learning classifiers.

Advantages:

- extremely fast inference
- interpretable features
- minimal compute requirements

---

## HOG + Approximate RBF SVM

Pipeline:

```
Image
↓
HOG feature extraction
↓
StandardScaler
↓
Nyström RBF feature mapping
↓
LinearSVC
```

Purpose:

Capture structural edge patterns using **Histogram of Oriented Gradients**.

---

## LBP + Approximate RBF SVM

Pipeline:

```
Image
↓
Local Binary Patterns
↓
StandardScaler
↓
Nyström RBF feature mapping
↓
LinearSVC
```

Purpose:

Capture **local texture patterns**.

---

## HSV Histogram + Logistic Regression

Pipeline:

```
Image
↓
HSV color histogram
↓
StandardScaler
↓
Logistic Regression
```

Purpose:

Capture **global color distributions**.

---

# 2 — Deep Feature Pipelines

These models use **pretrained CNN encoders** as feature extractors.

The CNN weights remain **frozen**.

Classifier is trained on extracted embeddings.

---

## Embedding Extraction

Backbone:

```
ResNet50
```

Embedding dimension:

```
2048
```

Embeddings cached to:

```
data/processed/embeddings/split_v1/encoder_resnet50/
```

Files produced:

```
train.npy
val.npy
test.npy
labels_train.npy
labels_val.npy
labels_test.npy
meta.json
```

---

## Logistic Regression on ResNet50 Embeddings

Pipeline:

```
ResNet50 embeddings
↓
StandardScaler
↓
LogisticRegression
```

---

## Approximate RBF SVM on ResNet50 Embeddings

Exact RBF SVM is computationally expensive at this scale.

Approximation used:

```
StandardScaler
↓
Nyström RBF kernel approximation
↓
LinearSVC
```

This retains nonlinear decision boundaries while remaining tractable.

---

# 3 — CNN Trained From Scratch

The project also explores models trained entirely from scratch.

---

## CustomCNN v1 Architecture

```
Input (224x224 RGB)

Conv2D 3→32
ReLU
MaxPool

Conv2D 32→64
ReLU
MaxPool

Conv2D 64→128
ReLU
MaxPool

AdaptiveAvgPool

Flatten

Linear 128→256
ReLU
Dropout 0.5

Linear 256→3
```

Model size:

```
127,043 parameters
≈0.485 MB
```

Training configuration:

| Parameter | Value |
|------|------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Epochs | 30 |
| Dropout | 0.5 |
| Scheduler | ReduceLROnPlateau |

Training artifacts:

```
checkpoint.pt
config.json
metrics.json
loss_curve.png
accuracy_curve.png
exported.onnx (optional)
```

Example curves:

```
docs/images/customcnn_v1_loss_curve.png
docs/images/customcnn_v1_accuracy_curve.png
```

---

# Experimental Results

Current results (Phase 3 still in progress).

| Model | Category | Test Accuracy | Macro F1 | Latency (ms/image) | Throughput (img/s) | Params | Size MB |
|------|------|------|------|------|------|------|------|
| HOG + RBF SVM | Handcrafted | 0.8024 | - | - | - | - | - |
| LBP + RBF SVM | Handcrafted | 0.6432 | 0.6542 | - | - | - | - |
| HSV Hist + Logistic Regression | Handcrafted | 0.5115 | 0.5123 | - | - | - | - |
| ResNet50 Embeddings + Logistic Regression | Deep Features | 0.9949 | 0.9950 | - | - | - | - |
| ResNet50 Embeddings + RBF SVM | Deep Features | 0.9877 | 0.9882 | - | - | - | - |
| CustomCNN v1 | Scratch CNN | 0.9454 | 0.9472 | 0.194 | 5152.97 | 127043 | 0.485 |

Values marked with **-** will be added in the future centralized benchmark.

---

# Example Predictions

Prediction examples stored in:

```
docs/images/predictions/
```

Example output:

```
cat_01_pred.png
dog_04_pred.png
wildlife_03_pred.png
```

Each example shows:

- input image
- predicted class
- model confidence

---

# Project Structure

```
AnimalClassification/
│
├── configs/
│   └── transforms_v1.yaml
│
├── data/
│   ├── prepared/
│   ├── processed/
│   │   └── embeddings/
│   └── splits/
│       └── split_v1/
│
├── docs/
│   └── images/
│
├── mlruns/
│
├── models/
│   ├── ml_basic_features/
│   ├── ml_deep_features/
│   └── cnn_scratch/
│
├── notebooks/
│   ├── 10_data_preparation/
│   ├── 20_ml_deep_features_fixed_encoder/
│   └── 30_cnn_scratch_custom/
│
├── reports/
│   ├── metrics/
│   └── figures/
│
├── src/
│   ├── data/
│   └── models/
│
├── requirements.txt
└── README.md
```

---

# Folder Descriptions

### configs

Configuration files controlling preprocessing and transforms.

---

### data

Contains prepared dataset, splits, and cached embeddings.

---

### docs

Documentation images used in the README.

---

### models

Stores trained models and experiment outputs.

Each run creates a timestamped directory.

---

### notebooks

Experimental notebooks grouped by phase.

---

### reports

Metrics, figures, and exported experiment summaries.

---

### src

Reusable project code.

Includes:

```
dataset loaders
transform utilities
model architectures
training utilities
evaluation helpers
```

---

# Hardware

Training environment detected in current runs:

```
CUDA GPU available
```

Device information and benchmarking metrics will be standardized in a future benchmarking notebook.

---

# Future Work

Planned improvements include:

- CustomCNN v2 architecture
- centralized inference benchmark
- additional evaluation metrics
- ONNX export improvements
- model comparison dashboard
- deployment experiments

---

# License

MIT License

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files to deal in the Software
without restriction.
```

---

# Acknowledgements

Libraries used:

- PyTorch
- Scikit-Learn
- NumPy
- Pandas
- MLflow
- Pillow
- Matplotlib
- ONNX

---

# Status

This project is currently **in active development**.

Additional models and benchmarking results will be added as experiments complete.