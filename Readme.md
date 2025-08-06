OpenSourceAI - Kaggle Competition Report

This repository contains the final deliverable for the Kaggle OpenSourceAI Competition, conducted during the Fall 2024 semester as part of the Open Source AI Applications course at Dankook University.


Project Overview

Objective
Perform binary classification on multimodal data using both image and text inputs.

Dataset Description

* train.csv: image paths, corresponding text, and ground-truth labels

* test.csv: image paths and text without labels

* sample_submission.csv: format template for submission


Methodology

Model Architecture
* Image encoder: EfficientNet-B3 (pretrained)
* Text encoder: BERT (bert-base-uncased)
* Fusion: Concatenation of image and text embeddings, followed by fully connected layers

Training Techniques
* Data Augmentation: Resize, Random Horizontal Flip, Rotation, Color Jitter
* Regularization: MixUp
* Optimization:
    * SAM (Sharpness-Aware Minimization)
    * Lookahead + AdamW optimizer
    * CosineAnnealingWarmRestarts learning rate scheduler
    * Mixed Precision Training via AMP (torch.cuda.amp)


Experimental Results

Model Version	Validation Accuracy	Public Score	Private Score
Baseline (Simple CNN + Text)	0.675	0.683	0.684
BERT + CNN	0.715	0.718	0.720
BERT + EfficientNet + SAM + MixUp	0.742	0.731	0.729

Final submitted model:
BERT + EfficientNet + SAM + MixUp


Repository Structure

File / Folder	Description
OpenSourceAI_Kaggle.ipynb	Final version used for submission, with best-performing model and configuration
Kaggle_NonOrganized.ipynb	Earlier experiments including baseline, BERT-only, and CNN-only variants. Contains multiple test versions before final convergence
reports/	Contains visual results such as accuracy curves, confusion matrix, and Kaggle submission screenshots
assets/ (optional)	Additional materials (e.g., model architecture diagram)


Visual Results

### Sample Submission Format
![Sample CSV](./reports/solution_sample_overview.png)

### Final Kaggle Leaderboard Results

**Public Leaderboard**  
![Public Leaderboard](./reports/kaggle_public_leaderboard.png)

**Private Leaderboard**  
![Private Leaderboard](./reports/kaggle_private_leaderboard.png)


Environment & Dependencies

Python 3.8+

PyTorch >= 1.10

CUDA Toolkit

transformers (HuggingFace)

timm, albumentations, torchvision, pandas, numpy


Notes
* This repository is based on the project submitted for the 2024 Fall Open Source AI Applications course at Dankook University.
* All experiments were conducted on GPU with mixed precision training enabled.
* The final result was submitted to the official Kaggle OSAI competition.