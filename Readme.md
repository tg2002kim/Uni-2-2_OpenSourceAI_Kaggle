Dankook University, 2nd Year 2nd Semester (Fall 2024)

# OpenSourceAI - Kaggle Competition Report

This repository contains the final deliverable for the **Kaggle OpenSourceAI Competition**, completed during the Fall 2024 semester for the **Open Source AI Applications** course at Dankook University. The project focuses on binary classification using multimodal data.

---

## Project Overview

- **Objective**: To perform binary classification on multimodal data, leveraging both image and text inputs.
- **Dataset**: A custom dataset with image paths, text descriptions, and binary labels.
- **Goal**: To achieve high classification accuracy and a strong leaderboard score.

---

## Methodology

### Model Architecture
- **Image Encoder**: **EfficientNet-B3** (pretrained)
- **Text Encoder**: **BERT** (bert-base-uncased)
- **Fusion**: Image and text embeddings are concatenated and fed into fully connected layers for classification.

### Training Techniques
- **Data Augmentation**: Techniques used include resizing, random horizontal flips, rotation, and color jitter.
- **Regularization**: **MixUp** was applied to improve model generalization.
- **Optimization**: The model was trained using **SAM (Sharpness-Aware Minimization)**, a **Lookahead + AdamW** optimizer, and a **CosineAnnealingWarmRestarts** learning rate scheduler. **Mixed Precision Training (AMP)** was enabled for faster training.

---

## Experimental Results

| Model Version | Validation Accuracy | Public Score | Private Score |
| :--- | :--- | :--- | :--- |
| Baseline (Simple CNN + Text) | 0.675 | 0.683 | 0.684 |
| BERT + CNN | 0.715 | 0.718 | 0.720 |
| **BERT + EfficientNet + SAM + MixUp** | **0.742** | **0.731** | **0.729** |

The **BERT + EfficientNet + SAM + MixUp** model was the final submitted version.

---

## Repository Structure

OpenSourceAI_Kaggle.ipynb       # Final notebook used for submission
Kaggle_NonOrganized.ipynb       # Earlier experiments and baseline models
reports/                        # Visual results, accuracy curves, confusion matrix
assets/ (optional)              # Model architecture diagrams or other materials


--- 

## Environment & Dependencies

- **Python**: 3.8+
- **Deep Learning Framework**: PyTorch >= 1.10
- **Libraries**: `transformers`, `timm`, `albumentations`, `torchvision`, `pandas`, `numpy`
- **GPU**: CUDA Toolkit is required for training.

---

## Notes

* This repository is part of a project for the 2024 Fall Open Source AI Applications course at Dankook University.
* All experiments were conducted on a GPU with mixed precision training enabled for efficiency.
* The final model was submitted to the official Kaggle OpenSourceAI competition.
