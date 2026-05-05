# AnimalClassification - Benchmarking Machine Learning Approaches for Animal Image Classification

## CSCI 4701: Deep Learning Course Submission

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10.0-orange)
![Torchvision](https://img.shields.io/badge/torchvision-0.25.0-darkgreen)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-green)
![Scikit-Image](https://img.shields.io/badge/scikit--image-0.26.0-teal)
![MLflow](https://img.shields.io/badge/MLflow-3.10.0-purple)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

A structured experimental pipeline for **animal image classification** comparing:

- classical computer vision approaches
- deep feature extraction with a fixed pretrained encoder
- custom CNN models trained from scratch
- ImageNet-pretrained CNN transfer learning
- ImageNet-pretrained Vision Transformer-family transfer learning
- custom Vision Transformers trained from scratch

The goal of this project is to **systematically benchmark different modeling strategies** under a shared dataset split, transformation pipeline, and experiment-tracking setup.

The repository is designed to be **reproducible, modular, and experiment-tracked**, allowing fair comparisons between approaches on a common three-class dataset consisting of **cats, dogs, and wildlife** images.

---

# Course Submission Information

This repository is submitted as part of **CSCI 4701: Deep Learning** and represents a group project submission.

## Team Members and Contribution

| Student | ID | Email | Contribution |
|------|------:|------|------:|
| Rufiz Bayramov | 16980 | rbayramov16980@ada.edu.az | 1/3 |
| Javad Taghiyev | 18172 | jtaghiyev18172@ada.edu.az | 1/3 |
| Asliddin Isroilov | 16788 | aisroilov16788@ada.edu.az | 1/3 |

## Scope Clarification

The **core coursework contribution** of this repository is concentrated in the scratch-CNN notebooks:

- `notebooks/30_cnn_scratch_custom/30_01_customcnn_v1.ipynb`
- `notebooks/30_cnn_scratch_custom/30_02_customcnn_v2.ipynb`

These notebooks contain the main implementation where course knowledge was applied most directly, including dataset loading, augmentation, model definition, optimization, regularization, training, evaluation, and comparison of CNN architectures trained from scratch in PyTorch.

The project was later expanded into a broader benchmark that also includes classical machine learning baselines, fixed pretrained feature extraction, pretrained CNN transfer learning, pretrained Vision Transformer-family models, and custom Vision Transformers from scratch. These additional experiments provide a stronger comparative context for understanding the behavior of scratch-trained deep learning models.

Earlier parts of the repository, including some baseline models and exploratory utilities, were initiated near the beginning of the semester with the assistance of AI tools/agents. Those components were useful for establishing baseline comparisons and building the broader experimental framework, but the final repository has been organized, audited, rerun, and documented as a coherent course project.

Some implementation work also preceded the full formal coverage of the corresponding theory in class. As the course progressed, the project was updated with stronger interpretation of optimization, regularization, convolutional inductive bias, transfer learning, and attention-based architectures.

---

# Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Experiment Tracking](#experiment-tracking)
- [Models Implemented](#models-implemented)
- [Classical Computer Vision Pipelines](#1---classical-computer-vision-pipelines)
- [Deep Feature Pipelines](#2---deep-feature-pipelines)
- [CNNs Trained From Scratch](#3---cnns-trained-from-scratch)
- [Pretrained CNN Transfer Learning](#4---pretrained-cnn-transfer-learning)
- [Pretrained Vision Transformer Transfer Learning](#5---pretrained-vision-transformer-transfer-learning)
- [Vision Transformers From Scratch](#6---vision-transformers-from-scratch)
- [Experimental Results](#experimental-results)
- [Loss Curve Snapshot](#loss-curve-snapshot)
- [Course Syllabus Relevance](#course-syllabus-relevance)
- [Project Structure](#project-structure)
- [Conclusion](#conclusion)

---

# Overview

This project investigates how different machine learning paradigms perform on the same image classification task:

1. **Handcrafted feature pipelines**
2. **Deep feature extraction using pretrained models**
3. **CNN architectures trained from scratch**
4. **Pretrained CNNs fine-tuned end to end**
5. **Pretrained Vision Transformer families**
6. **Custom Vision Transformers trained from scratch**

The comparison is designed to be fair by keeping dataset splits, preprocessing logic, evaluation metrics, and artifact conventions as consistent as possible across experiments.

All experiments share:

- a **fixed dataset split**
- a **common transformation pipeline**
- centralized **MLflow experiment tracking**
- standardized **metrics and reporting**
- saved artifacts for reproducibility

The task is a **3-class image classification problem**:

- cats
- dogs
- wildlife

---

# Dataset

The dataset is constructed by merging several public datasets in order to create a larger and more diverse three-class benchmark for animal image classification.

| Dataset | Description |
|------|------|
| Microsoft Cats vs Dogs | Internet images of cats and dogs |
| AFHQv2 | High quality images of cats, dogs, and wild animals |
| Animal Face Dataset | Various wildlife species |
| HuggingFace Animal Faces | Cat and dog facial dataset |

After cleaning and deduplication the final dataset contains approximately:

```text
Total images ~= 62,659
cats ~= 23.7k
dogs ~= 23.8k
wildlife ~= 16k
```

The dataset is organized using a deterministic split called:

```text
split_v1
```

Dataset sizes:

| Split | Samples |
|------|------:|
| Train | 50,127 |
| Validation | 6,266 |
| Test | 6,266 |

Class distribution:

| Split | Cats | Dogs | Wildlife |
|------|------:|------:|------:|
| Train | 18,958 | 18,312 | 12,857 |
| Validation | 2,370 | 2,289 | 1,607 |
| Test | 2,370 | 2,289 | 1,607 |

The deterministic split is important because all model families are evaluated on the same validation and test data. This makes the benchmark more meaningful than comparing models trained on different random splits.

---

# Image Transformations and Augmentation

The project uses a shared transform configuration stored in:

```text
configs/transforms_v1.yaml
```

## Training Transform Pipeline

The training pipeline applies stochastic augmentation to improve generalization:

- random resized crop
- random horizontal flip
- mild color jitter
- ImageNet-style normalization for pretrained models
- tensor conversion

## Evaluation Transform Pipeline

The validation and test pipeline is deterministic:

- resize
- center crop
- tensor conversion
- normalization

This separation is important because random augmentation should only affect training. Validation and test metrics must be deterministic and comparable across runs.

Example augmented samples are stored at:

```text
reports/figures/sample_augmented_images.png
```

---

# Experiment Tracking

All experiments are tracked using **MLflow**.

Tracking directory:

```text
mlruns/
```

Each training run logs:

- parameters
- metrics
- artifacts
- configuration
- run metadata

Typical run artifacts include:

```text
checkpoint.pt
config.json
metrics.json
loss_curve.png
accuracy_curve.png
```

Classical ML and fixed-embedding pipelines store serialized sklearn models instead:

```text
model.pkl
classifier.pkl
```

This tracking structure makes the project reproducible and allows later comparison of models without rerunning every experiment.

---

# Models Implemented

This project includes the following implemented and benchmarked model families.

## Classical ML with Handcrafted Features

- HOG + Approximate RBF SVM
- LBP + Approximate RBF SVM
- HSV Color Histogram + Logistic Regression

## Deep Features with Fixed Pretrained Encoder

- ResNet50 Embedding Extraction
- ResNet50 Embeddings + Logistic Regression
- ResNet50 Embeddings + Approximate RBF SVM

## CNNs Trained From Scratch

- CustomCNN v1
- CustomCNN v2

## Pretrained CNN Transfer Learning

- ResNet18 Pretrained
- MobileNetV3 Large Pretrained
- EfficientNet-B0 Pretrained
- ResNet50 Pretrained
- EfficientNet-B2 Pretrained

## Pretrained Vision Transformers

- ViT-B/16 Pretrained
- Swin-T Pretrained
- Swin V2-S Pretrained
- MaxViT-T Pretrained

## Vision Transformers From Scratch

- CustomViT v1
- CustomViT v2

---

# 1 - Classical Computer Vision Pipelines

These models use handcrafted feature extractors combined with classical machine learning classifiers.

Advantages:

- fast to train compared with deep neural networks
- interpretable feature design
- useful baseline before using learned representations

Limitations:

- feature quality depends heavily on manual design
- weaker ability to learn complex high-level visual semantics
- lower performance than pretrained deep representations on this dataset

## HOG + Approximate RBF SVM

Notebook:

```text
notebooks/10_ml_basic_features/10_01_hog_svm.ipynb
```

Feature representation:

- Histogram of Oriented Gradients
- captures local edge and shape structure

Classifier:

- Nyström approximation of an RBF kernel
- LinearSVC classifier

Test result:

- **Test accuracy:** 0.8024
- **Test macro F1:** 0.8040

Artifact:

```text
models/ml_basic_features/hog_svm/run_20260312_095858/model.pkl
```

## LBP + Approximate RBF SVM

Notebook:

```text
notebooks/10_ml_basic_features/10_02_lbp_svm.ipynb
```

Feature representation:

- Local Binary Patterns
- captures local texture statistics

Classifier:

- Nyström approximation of an RBF kernel
- LinearSVC classifier

Test result:

- **Test accuracy:** 0.6432
- **Test macro F1:** 0.6542

Artifact:

```text
models/ml_basic_features/lbp_svm/run_20260311_195915/model.pkl
```

## HSV Histogram + Logistic Regression

Notebook:

```text
notebooks/10_ml_basic_features/10_03_colorhist_lr.ipynb
```

Feature representation:

- HSV color histogram
- captures global color distribution

Classifier:

- Logistic Regression

Test result:

- **Test accuracy:** 0.5115
- **Test macro F1:** 0.5123

Artifact:

```text
models/ml_basic_features/colorhist_lr/run_20260311_194949/model.pkl
```

---

# 2 - Deep Feature Pipelines

These models use a pretrained ResNet50 encoder as a fixed feature extractor. Instead of fine-tuning the neural network end to end, image embeddings are extracted once and then used by classical classifiers.

## Embedding Extraction

Notebook:

```text
notebooks/20_ml_deep_features_fixed_encoder/20_01_extract_embeddings_resnet50.ipynb
```

Encoder:

- ImageNet-pretrained ResNet50
- final classification layer removed
- 2048-dimensional feature vector extracted per image

MLflow records separate embedding extraction throughput metrics. These are useful for pipeline analysis but are not used as final classifier latency numbers in the main benchmark table.

## Logistic Regression on ResNet50 Embeddings

Notebook:

```text
notebooks/20_ml_deep_features_fixed_encoder/20_02_lr_on_embeddings.ipynb
```

Test result:

- **Test accuracy:** 0.9949
- **Test macro F1:** 0.9950

Artifact:

```text
models/ml_deep_features/resnet50_embeddings/run_20260311_211643/classifier.pkl
```

## Approximate RBF SVM on ResNet50 Embeddings

Notebook:

```text
notebooks/20_ml_deep_features_fixed_encoder/20_03_svm_on_embeddings.ipynb
```

Test result:

- **Test accuracy:** 0.9877
- **Test macro F1:** 0.9882

Artifact:

```text
models/ml_deep_features/resnet50_embeddings/run_20260311_210533/classifier.pkl
```

Observation:

The fixed-embedding logistic regression model is extremely strong. This shows that ImageNet-pretrained visual representations already separate this dataset very well, even before end-to-end fine-tuning.

---

# 3 - CNNs Trained From Scratch

The scratch CNN notebooks are the central course-submission component because they directly implement neural networks, optimization, regularization, training loops, evaluation, and inference benchmarking in PyTorch.

Shared training choices:

- optimizer: AdamW
- loss: cross-entropy
- learning-rate scheduling
- dropout
- batch normalization in the deeper model
- gradient clipping
- runtime data augmentation
- MLflow tracking
- checkpointing and curve export

## CustomCNN v1 Architecture

Notebook:

```text
notebooks/30_cnn_scratch_custom/30_01_customcnn_v1.ipynb
```

Architecture:

- small convolutional neural network
- stacked convolution and pooling blocks
- dropout-regularized classifier head
- 3-class output layer

Model size:

```text
127,043 parameters
~0.485 MB
```

Best validation result:

- **Best epoch:** 28
- **Best validation macro F1:** 0.9487

Test result:

- **Test loss:** 0.1442
- **Test accuracy:** 0.9454
- **Test macro F1:** 0.9472

Inference benchmark:

- **Latency per image:** 0.1941 ms
- **Throughput:** 5152.97 images/sec

Artifacts:

```text
models/cnn_scratch/customcnn_v1/run_20260313_095856/
- checkpoint.pt
- config.json
- metrics.json
- loss_curve.png
- accuracy_curve.png
```

## CustomCNN v2 Architecture

Notebook:

```text
notebooks/30_cnn_scratch_custom/30_02_customcnn_v2.ipynb
```

Architecture:

- deeper scratch CNN
- stacked convolutional blocks
- batch normalization
- dropout
- larger classifier head
- 3-class output layer

Model size:

```text
355,491 parameters
~1.360 MB
```

Best validation result:

- **Best epoch:** 27
- **Best validation macro F1:** 0.9746

Test result:

- **Test loss:** 0.0864
- **Test accuracy:** 0.9714
- **Test macro F1:** 0.9722

Inference benchmark:

- **Latency per image:** 0.4496 ms
- **Throughput:** 2224.17 images/sec

Artifacts:

```text
models/cnn_scratch/customcnn_v2/run_20260313_114741/
- checkpoint.pt
- config.json
- metrics.json
- loss_curve.png
- accuracy_curve.png
```

Scratch CNN comparison:

`CustomCNN v2` improves substantially over `CustomCNN v1`. The deeper stacked-convolution design with batch normalization increases parameter count from **127,043** to **355,491**, but raises test macro F1 from **0.9472** to **0.9722**. This is a direct example of how architectural depth, normalization, and regularization can improve generalization.

---

# 4 - Pretrained CNN Transfer Learning

Phase 4 extends the benchmark to end-to-end pretrained CNN classifiers initialized from official ImageNet weights in `torchvision`.

Unlike the fixed-embedding pipelines, these models:

- replace the original ImageNet classification head with a 3-class head
- train the new head first with the backbone frozen
- then partially fine-tune the pretrained backbone
- log checkpoint, metrics, curves, latency, throughput, parameter count, and model size

Shared transfer-learning recipe:

| Parameter | Value |
|------|------|
| Head-only epochs | 5 |
| Partial fine-tuning epochs | 15 |
| Optimizer | AdamW |
| Head learning rate | 1e-3 |
| Backbone learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Seed | 42 |

## Pretrained CNN Results

| Model | Weights | Best Val Macro F1 | Test Accuracy | Test Macro F1 | Latency ms/img | Params | Size MB |
|------|------|------:|------:|------:|------:|------:|------:|
| ResNet18 | IMAGENET1K_V1 | 0.9951 | 0.9947 | 0.9949 | 0.2786 | 11,178,051 | 42.678 |
| MobileNetV3 Large | IMAGENET1K_V2 | 0.9933 | 0.9914 | 0.9918 | 0.4396 | 2,974,835 | 11.442 |
| EfficientNet-B0 | IMAGENET1K_V1 | 0.9945 | 0.9928 | 0.9932 | 0.4424 | 4,011,391 | 15.463 |
| ResNet50 | IMAGENET1K_V2 | 0.9973 | 0.9959 | 0.9961 | 0.5290 | 23,514,179 | 89.903 |
| EfficientNet-B2 | IMAGENET1K_V1 | 0.9949 | 0.9938 | 0.9941 | 0.5408 | 7,705,221 | 29.651 |

Observation:

The pretrained CNN family clearly outperforms scratch CNNs. `ResNet50` is the strongest pretrained CNN, while `ResNet18` is a particularly strong efficiency baseline because it achieves high accuracy with relatively low latency.

---

# 5 - Pretrained Vision Transformer Transfer Learning

Phase 5 extends the benchmark from convolutional ImageNet backbones to pretrained transformer-family image classifiers from `torchvision`.

These models follow the same transfer-learning contract as pretrained CNNs:

- load official ImageNet pretrained weights
- replace the original classification head with a 3-class project head
- train the new head with the backbone frozen
- partially fine-tune the pretrained backbone
- log all artifacts and metrics through MLflow

Shared training recipe:

| Parameter | Value |
|------|------|
| Head-only epochs | 5 |
| Partial fine-tuning epochs | 15 |
| Optimizer | AdamW |
| Head learning rate | 1e-3 |
| Backbone learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Seed | 42 |

## Pretrained ViT Results

| Model | Architecture Family | Best Val Macro F1 | Test Accuracy | Test Macro F1 | Latency ms/img | Params | Size MB |
|------|------|------:|------:|------:|------:|------:|------:|
| ViT-B/16 | Plain ViT | 0.9977 | 0.9968 | 0.9969 | 1.5236 | 85,800,963 | 327.305 |
| Swin-T | Hierarchical shifted-window Transformer | 0.9982 | 0.9973 | 0.9974 | 0.8609 | 27,521,661 | 105.207 |
| Swin V2-S | Second-generation shifted-window Transformer | 0.9978 | 0.9962 | 0.9963 | 2.1057 | 48,970,749 | 187.600 |
| MaxViT-T | Hybrid convolution-attention Transformer | 0.9978 | 0.9974 | 0.9976 | 1.3636 | 30,409,163 | 116.587 |

Artifacts:

```text
models/vit/vit_b_16/run_20260427_102658/
models/vit/swin_t/run_20260427_110425/
models/vit/swin_v2_s/run_20260427_112808/
models/vit/maxvit_t/run_20260429_095644/
```

Observation:

The pretrained transformer-family models are the strongest overall group. `MaxViT-T` achieves the best final result, while `Swin-T` is extremely competitive and more efficient than ViT-B/16.

---

# 6 - Vision Transformers From Scratch

Phase 6 adds educational Vision Transformer models implemented directly in PyTorch.

Unlike the pretrained ViT family, these models start from random initialization and are trained end-to-end on `split_v1`. They are intentionally smaller than ImageNet-pretrained ViT backbones so they remain practical for course-focused experimentation while still exposing the main Transformer building blocks.

Both custom ViTs use:

- patch embedding with 16x16 image patches
- a learnable class token
- learnable positional embeddings
- stacked Transformer encoder blocks
- multi-head self-attention
- MLP feed-forward blocks
- layer normalization and dropout
- a final 3-class classification head

## CustomViT v1

Architecture:

- image size: 224
- patch size: 16
- embedding dimension: 192
- encoder depth: 6
- attention heads: 3
- MLP ratio: 4.0
- dropout: 0.1

Model size:

```text
2,855,811 parameters
~10.89 MB
```

Test result:

- **Test accuracy:** 0.9417
- **Test macro F1:** 0.9449
- **Latency per image:** 0.2587 ms
- **Throughput:** 3865.15 images/sec

## CustomViT v2

Architecture:

- image size: 224
- patch size: 16
- embedding dimension: 256
- encoder depth: 8
- attention heads: 8
- MLP ratio: 4.0
- dropout: 0.1

Model size:

```text
6,566,915 parameters
~25.05 MB
```

Test result:

- **Test accuracy:** 0.9489
- **Test macro F1:** 0.9523
- **Latency per image:** 0.4416 ms
- **Throughput:** 2264.60 images/sec

Artifacts:

```text
models/vit_scratch/customvit_v1/run_20260505_102351/
models/vit_scratch/customvit_v2/run_20260505_111414/
```

Observation:

`CustomViT v2` improves over `CustomViT v1`, showing the benefit of increased Transformer capacity. However, both scratch ViTs remain below `CustomCNN v2`, which illustrates that convolutional inductive bias is still very useful when training from scratch on this dataset.

---

# Experimental Results

| Model | Category | Test Accuracy | Macro F1 | Latency (ms/image) | Throughput (img/s) | Params | Size MB |
|------|------|------:|------:|------:|------:|------:|------:|
| HOG + Approx. RBF SVM | Handcrafted Features | 0.8024 | 0.8040 | - | - | - | - |
| LBP + Approx. RBF SVM | Handcrafted Features | 0.6432 | 0.6542 | - | - | - | - |
| HSV Histogram + Logistic Regression | Handcrafted Features | 0.5115 | 0.5123 | - | - | - | - |
| ResNet50 Embeddings + Logistic Regression | Deep Features | 0.9949 | 0.9950 | - | - | - | - |
| ResNet50 Embeddings + Approx. RBF SVM | Deep Features | 0.9877 | 0.9882 | - | - | - | - |
| CustomCNN v1 | CNN from Scratch | 0.9454 | 0.9472 | 0.1941 | 5152.97 | 127,043 | 0.485 |
| CustomCNN v2 | CNN from Scratch | 0.9714 | 0.9722 | 0.4496 | 2224.17 | 355,491 | 1.360 |
| ResNet18 Pretrained | CNN Transfer Learning | 0.9947 | 0.9949 | 0.2786 | 3588.84 | 11,178,051 | 42.678 |
| MobileNetV3 Large Pretrained | CNN Transfer Learning | 0.9914 | 0.9918 | 0.4396 | 2275.05 | 2,974,835 | 11.442 |
| EfficientNet-B0 Pretrained | CNN Transfer Learning | 0.9928 | 0.9932 | 0.4424 | 2260.58 | 4,011,391 | 15.463 |
| ResNet50 Pretrained | CNN Transfer Learning | 0.9959 | 0.9961 | 0.5290 | 1890.34 | 23,514,179 | 89.903 |
| EfficientNet-B2 Pretrained | CNN Transfer Learning | 0.9938 | 0.9941 | 0.5408 | 1849.21 | 7,705,221 | 29.651 |
| ViT-B/16 Pretrained | ViT Transfer Learning | 0.9968 | 0.9969 | 1.5236 | 656.32 | 85,800,963 | 327.305 |
| Swin-T Pretrained | ViT Transfer Learning | 0.9973 | 0.9974 | 0.8609 | 1161.54 | 27,521,661 | 105.207 |
| Swin V2-S Pretrained | ViT Transfer Learning | 0.9962 | 0.9963 | 2.1057 | 474.91 | 48,970,749 | 187.600 |
| MaxViT-T Pretrained | ViT Transfer Learning | 0.9974 | 0.9976 | 1.3636 | 733.37 | 30,409,163 | 116.587 |
| CustomViT v1 | ViT from Scratch | 0.9417 | 0.9449 | 0.2587 | 3865.15 | 2,855,811 | 10.894 |
| CustomViT v2 | ViT from Scratch | 0.9489 | 0.9523 | 0.4416 | 2264.60 | 6,566,915 | 25.051 |

Notes:

- Classical and fixed-embedding rows do not have directly comparable neural-network parameter counts.
- Latency and throughput were not logged for the classical and fixed-embedding classifier rows.
- Serialized sklearn artifact sizes are available, but they are not the same concept as neural-network parameter size.
- The strongest final model is **MaxViT-T Pretrained**.
- The strongest scratch-trained model is **CustomCNN v2**.

---

# Loss Curve Snapshot

The following six loss curves summarize representative training behavior across moderate, strong, and best-performing neural models.

## Moderate-Performing Models

**CustomCNN v1** reaches a respectable scratch-trained baseline, but the validation curve settles higher than the stronger transfer-learning models. This reflects the limitation of a small CNN trained without external pretraining.

![CustomCNN v1 loss curve](models/cnn_scratch/customcnn_v1/run_20260313_095856/loss_curve.png)

**CustomViT v2** improves over the smaller scratch ViT and learns steadily, but its loss remains above the pretrained families. This highlights how Vision Transformers benefit strongly from large-scale pretraining.

![CustomViT v2 loss curve](models/vit_scratch/customvit_v2/run_20260505_111414/loss_curve.png)

## Strong-Performing Models

**ResNet18 Pretrained** converges quickly and maintains a low validation loss while staying relatively small and fast. It is one of the best efficiency-oriented baselines in the benchmark.

![ResNet18 pretrained loss curve](models/cnn_pretrained/resnet18_pretrained/run_20260403_103808/loss_curve.png)

**ResNet50 Pretrained** achieves lower final loss than the smaller CNN baselines and remains the strongest pretrained CNN result. The curve reflects the value of deeper residual ImageNet features for this dataset.

![ResNet50 pretrained loss curve](models/cnn_pretrained/resnet50_pretrained/run_20260403_114106/loss_curve.png)

## Best-Performing Models

**Swin-T Pretrained** reaches very low loss while remaining lighter than ViT-B/16. The curve supports the benchmark result where Swin-T is both accurate and efficient among transformer-family models.

![Swin-T pretrained loss curve](models/vit/swin_t/run_20260427_110425/loss_curve.png)

**MaxViT-T Pretrained** gives the strongest final benchmark score. Its loss curve stays in the same low-loss regime as the best transformer models, matching its top test macro F1 result.

![MaxViT-T pretrained loss curve](models/vit/maxvit_t/run_20260429_095644/loss_curve.png)

---

# Course Syllabus Relevance

The project connects directly to many topics from the deep learning course:

| Course Topic | Project Connection |
|------|------|
| Weight Decay | AdamW regularization in scratch and transfer-learning models |
| Dropout | Used in scratch CNNs, scratch ViTs, and classifier heads |
| Data Augmentation | Runtime image augmentation for training transforms |
| Learning Rate | Separate learning rates for classifier heads and pretrained backbones |
| Adam / AdamW | Main optimizer for neural model training |
| Batch Normalization | Used in the deeper scratch CNN architecture |
| Transfer Learning and Fine-Tuning | Pretrained CNN and ViT families |
| Residual Networks | ResNet18 and ResNet50 baselines |
| Mobile/Efficient Models | MobileNetV3 and EfficientNet families |
| Vision Transformers | ViT-B/16, Swin, Swin V2, MaxViT |
| Embeddings | ResNet50 fixed-feature pipelines |
| Hyperparameter Tuning | Architecture scaling from CustomCNN v1 to v2 and CustomViT v1 to v2 |
| Training Efficient Models | Latency, throughput, parameter count, and model size comparisons |

The scratch CNN notebooks are the clearest direct application of the course material, while the broader benchmark helps contextualize how course concepts compare with modern pretrained architectures.

---

# Current Stored Figures and Artifacts

The repository contains generated artifacts from preprocessing and training runs, including:

- `reports/figures/sample_augmented_images.png`
- `models/cnn_scratch/customcnn_v1/run_20260313_095856/loss_curve.png`
- `models/cnn_scratch/customcnn_v1/run_20260313_095856/accuracy_curve.png`
- `models/cnn_scratch/customcnn_v2/run_20260313_114741/loss_curve.png`
- `models/cnn_scratch/customcnn_v2/run_20260313_114741/accuracy_curve.png`
- `models/cnn_pretrained/resnet18_pretrained/run_20260403_103808/loss_curve.png`
- `models/cnn_pretrained/resnet18_pretrained/run_20260403_103808/accuracy_curve.png`
- `models/cnn_pretrained/resnet50_pretrained/run_20260403_114106/loss_curve.png`
- `models/cnn_pretrained/resnet50_pretrained/run_20260403_114106/accuracy_curve.png`
- `models/vit/swin_t/run_20260427_110425/loss_curve.png`
- `models/vit/maxvit_t/run_20260429_095644/loss_curve.png`
- `models/vit_scratch/customvit_v1/run_20260505_102351/loss_curve.png`
- `models/vit_scratch/customvit_v2/run_20260505_111414/loss_curve.png`

The main trained model directories are:

```text
models/ml_basic_features/
models/ml_deep_features/
models/cnn_scratch/
models/cnn_pretrained/
models/vit/
models/vit_scratch/
```

---

# Project Structure

```text
AnimalClassification/
|-- configs/
|   `-- transforms_v1.yaml
|-- data/
|   |-- prepared/
|   |-- processed/
|   |   |-- features/
|   |   `-- embeddings/
|   `-- splits/
|       `-- split_v1/
|-- documentation/
|-- mlruns/
|-- models/
|   |-- ml_basic_features/
|   |-- ml_deep_features/
|   |-- cnn_scratch/
|   |-- cnn_pretrained/
|   |-- vit/
|   `-- vit_scratch/
|-- notebooks/
|   |-- 00_project_setup.ipynb
|   |-- 01_data_prep_and_splits.ipynb
|   |-- 02_transforms_and_augmentation.ipynb
|   |-- 10_ml_basic_features/
|   |-- 20_ml_deep_features_fixed_encoder/
|   |-- 30_cnn_scratch_custom/
|   |-- 40_cnn_pretrained/
|   |-- 50_vit/
|   `-- 60_vit_scratch/
|-- reports/
|   |-- figures/
|   `-- metrics/
|-- scripts/
|-- src/
|   |-- data/
|   `-- models/
|       |-- cnn_scratch/
|       |-- cnn_pretrained/
|       |-- vit/
|       `-- vit_scratch/
|-- project_report.md
|-- README.md
`-- requirements.txt
```

---

# Hardware and Environment

Observed training environment in the provided experiment runs:

- CUDA GPU available
- neural model training and inference benchmarking ran on GPU
- batch sizes vary by model family and memory requirements
- runtime artifacts record the selected device, parameter count, model size, latency, and throughput where applicable
- artifacts were migrated between separate Windows and Linux machines using repository-relative paths

Core software versions:

| Package | Version |
|------|------|
| Python | 3.12 |
| torch | 2.10.0 |
| torchvision | 0.25.0 |
| scikit-learn | 1.8.0 |
| scikit-image | 0.26.0 |
| numpy | 2.4.2 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| mlflow | 3.10.0 |
| Pillow | 12.1.1 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |

---

# Discussion

The benchmark shows a clear progression:

1. Handcrafted features provide useful baselines but are limited.
2. Fixed ResNet50 embeddings are already extremely strong.
3. Scratch CNNs demonstrate the core course concepts and improve substantially from v1 to v2.
4. Pretrained CNNs outperform scratch CNNs through transfer learning.
5. Pretrained ViT-family models produce the strongest final results.
6. Scratch ViTs are educationally valuable but do not outperform scratch CNNs on this dataset.

The strongest completed result is **MaxViT-T Pretrained**, with:

- **Test accuracy:** 0.9974
- **Test macro F1:** 0.9976

The strongest scratch-trained model is **CustomCNN v2**, with:

- **Test accuracy:** 0.9714
- **Test macro F1:** 0.9722

This supports an important lesson: modern pretrained representations are extremely powerful, but building and training scratch CNNs remains essential for understanding deep learning fundamentals.

---

# Limitations

The main limitations are:

- the benchmark uses one dataset and one deterministic split
- some very high accuracies may reflect the relative simplicity of the dataset
- classical and fixed-embedding pipelines do not have fully comparable latency logs
- ONNX export was attempted in earlier phases but was not used as a final evaluation artifact
- external validation on a completely separate dataset was not included

These limitations do not invalidate the benchmark, but they should be considered when interpreting near-perfect accuracy values.

---

# Future Work

Possible extensions include:

- a lightweight MLOps simulation with a FastAPI inference endpoint
- synthetic request traffic for monitoring confidence and latency drift
- Prometheus/Grafana-style deployment monitoring
- external validation on a new animal dataset
- more standardized inference timing across CPU and GPU
- additional robustness tests for blurred, cropped, low-light, or out-of-distribution inputs

---

# Conclusion

This project evolved from a course-focused scratch-CNN implementation into a broader animal image classification benchmark. The core deep learning course work is represented by the scratch CNN models, especially the progression from `CustomCNN v1` to `CustomCNN v2`. The expanded experiments provide additional context by comparing those scratch-trained networks against classical ML baselines, fixed deep features, pretrained CNNs, pretrained Vision Transformers, and custom ViTs from scratch.

The final results show that pretrained models dominate this dataset, with `MaxViT-T Pretrained` achieving the highest overall macro F1. At the same time, the scratch CNN experiments remain the most important educational component because they directly demonstrate architecture design, optimization, regularization, training curves, and evaluation from first principles.

Overall, the repository provides both a course-aligned deep learning implementation and a broader reproducible benchmark of image classification strategies.

