# AnimalClassification — Benchmarking Machine Learning Approaches for Animal Image Classification

## Overview

**AnimalClassification** is a research-oriented project designed to benchmark and compare multiple machine learning approaches for **animal image classification**.

The project evaluates different families of models on the **same dataset, preprocessing pipeline, and evaluation framework** to provide a fair comparison between classical computer vision methods and modern deep learning architectures.

The classification task consists of **three classes**:

- **Cats**
- **Dogs**
- **Wildlife**

The goal is not only to measure **classification performance**, but also to evaluate **computational efficiency and deployment cost**.

Metrics include:

- Accuracy
- Macro F1 Score
- Precision / Recall
- Inference latency
- Throughput (images/sec)
- Model size
- Parameter count

This allows comparison between models from both **performance** and **deployment feasibility** perspectives.

---

# Dataset

The dataset is constructed by merging several public datasets:

| Dataset | Description |
|------|------|
| Microsoft Cats vs Dogs | Internet images of cats and dogs |
| AFHQv2 | High quality images of cats, dogs, and wild animals |
| Animal Face Dataset (AFD) | Various wildlife species |
| HuggingFace Animal Faces | Cat and dog facial dataset |

After cleaning and deduplication the final dataset contains approximately:
```

Total images ≈ 62,659

Cats ≈ 23,693  
Dogs ≈ 22,894  
Wildlife ≈ 16,072

```

Dataset location:
```

data/prepared/  
cats/  
dogs/  
wildlife/

```

---

# Dataset Splits

To ensure **fair benchmarking and reproducibility**, dataset splits are generated once and reused for all experiments.
```

Train: 80%  
Validation: 10%  
Test: 10%  
Seed: 42

```

Split manifests are stored in:
```

data/splits/split\_v1/  
train.csv  
val.csv  
test.csv  
classes.json

```

CSV format:
```

filepath,label  
data/prepared/cats/img001.jpg,cats  
data/prepared/dogs/img002.jpg,dogs

```

Using CSV manifests instead of duplicating images ensures:

- reproducibility
- faster dataset loading
- easier auditing
- no duplicated storage

---

# Project Structure
```

AnimalClassification/

data/  
datasets\_raw/ # original datasets  
prepared/ # merged cleaned dataset  
splits/  
split\_v1/ # dataset split manifests  
processed/ # cached features and embeddings

configs/  
transforms\_v1.yaml # shared augmentation configuration

models/  
ml\_basic\_features/  
ml\_deep\_features/  
cnn\_scratch/  
cnn\_pretrained/  
vit/

notebooks/

```
00_project_setup.ipynb
01_data_prep_and_splits.ipynb
02_transforms_and_augmentation.ipynb

10_ml_basic_features/
20_ml_deep_features_fixed_encoder/
30_cnn_scratch_custom/
40_cnn_pretrained/
50_vit/
```

reports/  
metrics/  
figures/

src/  
data/  
dataset\_loader.py  
split\_generator.py  
transforms.py

scripts/

90\_evaluate\_all\_models.ipynb

```

---

# Experiment Pipeline

The overall workflow follows this pipeline:
```

Dataset preparation  
↓  
Dataset split generation  
↓  
Transform configuration  
↓  
Model training  
↓  
Model export  
↓  
Benchmark evaluation

```

All models operate on the **same dataset splits and transforms** to ensure fair comparison.

---

# Model Families

The project benchmarks five groups of models.

## 1. Classical ML — Handcrafted Features

Traditional computer vision features extracted directly from images.

Examples:

- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Color Histograms

Pipeline:
```

image → feature extractor → classifier

```

Classifiers include:

- Logistic Regression
- Linear SVM
- Kernel approximations (Nystroem)

---

## 2. Classical ML — Deep Features

Images are encoded using a **fixed pretrained neural network**, and the resulting embeddings are used by classical classifiers.

Pipeline:
```

image → pretrained encoder → embedding → classifier

```

Example encoder:

- ResNet50 (ImageNet pretrained)

Embeddings are cached to avoid recomputation.

---

## 3. CNNs Trained From Scratch

Custom convolutional neural networks trained directly on the dataset.

These models serve as a baseline for learning without transfer learning.

---

## 4. CNNs Using Pretrained Backbones

Transfer learning with pretrained networks.

Small models:

- ResNet18
- MobileNetV3
- EfficientNet-B0

Medium models:

- ResNet50
- EfficientNet-B2

---

## 5. Vision Transformers (ViT)

Transformer-based architectures for image classification.

Two variants are tested:

- ViT trained from scratch
- ViT fine-tuned from pretrained models

Examples:

- ViT Tiny
- DeiT Small
- ViT Base

---

# Caching Strategy

To reduce redundant computation, the project uses caching.

## Dataset Splits
```

data/splits/split\_v1/

```

Generated once and reused by all experiments.

---

## Embedding Cache

Deep feature embeddings are stored in:
```

data/processed/embeddings/  
split\_v1/  
encoder\_resnet50/

```

Example files:
```

train.npy  
val.npy  
test.npy  
labels\_train.npy  
labels\_val.npy  
labels\_test.npy

```

This allows multiple classifiers to reuse the same embeddings.

---

## Transform Configuration

All models use the same transform configuration defined in:
```

configs/transforms\_v1.yaml

```

Example configuration:
```

image\_size: 224

train\_transforms:

*   random\_resized\_crop
*   random\_flip
*   random\_rotation
*   color\_jitter

eval\_transforms:

*   resize
*   center\_crop
*   normalize

```

Transforms are applied **dynamically during dataset loading**.

---

# Experiment Tracking

Experiments are tracked using **MLflow**.

MLflow logs:

- parameters
- metrics
- artifacts

Typical logged metrics:
```

accuracy  
f1\_macro  
precision  
recall  
latency\_ms  
model\_size\_mb  
parameter\_count

```

Run artifacts include:
```

model.pkl  
metrics.json  
confusion\_matrix.png

```

MLflow tracking directory:
```

mlruns/

```

---

# Global Evaluation Notebook

The notebook:
```

90\_evaluate\_all\_models.ipynb

```

acts as the **central benchmarking system**.

Responsibilities:

- discover trained models
- load configurations
- run evaluation on the test dataset
- measure inference latency
- aggregate metrics
- produce comparison reports

Outputs:
```

reports/metrics/leaderboard.csv  
reports/figures/accuracy\_vs\_latency.png  
reports/figures/model\_size\_vs\_accuracy.png

```

---

# Reproducibility

The project ensures reproducibility by enforcing:

- fixed dataset splits
- shared transform configuration
- logged experiment parameters
- saved model configurations
- MLflow experiment tracking

This guarantees that all models are evaluated under **identical conditions**.

---

# Project Status

Current development phases:

| Phase | Description |
|------|------|
| Phase 1 | Dataset preparation and pipeline infrastructure |
| Phase 2 | Classical ML baselines |
| Phase 3 | CNN models trained from scratch |
| Phase 4 | CNN transfer learning |
| Phase 5 | Vision Transformers |
| Phase 6 | Global model benchmarking |

---

# Final Goal

The final output of the project will be a comprehensive benchmark comparing models across multiple dimensions.

Example leaderboard:

| Model | Accuracy | F1 | Latency | Params | Size |
|------|------|------|------|------|------|
| HOG + Logistic Regression | 0.71 | 0.69 | 2ms | - | 5MB |
| ResNet18 | 0.89 | 0.88 | 7ms | 11M | 44MB |
| EfficientNet-B0 | 0.91 | 0.90 | 9ms | 5M | 20MB |
| ViT Small | 0.92 | 0.91 | 15ms | 22M | 85MB |

This allows evaluation of **accuracy vs deployment cost tradeoffs**.

---

# License

This project is intended for research and educational purposes.

Dataset licenses remain with their original sources.
