# DeepFake Detection
A deep learning project that detects whether a face image is **Real** or **Fake (AI-generated / manipulated)** using a fine-tuned **EfficientNet-B0** model built with PyTorch.

## Overview
With the rise of AI-generated media, detecting deepfakes has become a critical task. This project trains a binary image classifier on the [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset to distinguish real human faces from synthetically generated ones.

## Project Structure
DeepFake-Detection/
├── deepfake-detection.ipynb   # Full training pipeline (EDA, training, evaluation)
├── predict.py                 # Inference script for single or multiple images
├── best_model.pth             # Saved best model weights
└── image.png                  # Sample output / result visualization

## Model Architecture
- **Base Model:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Classes:** `fake` | `real`
- **Input Size:** 224 × 224 RGB
- **Framework:** PyTorch + torchvision
- **Hardware:** NVIDIA Tesla T4 GPU (Kaggle)

## Dataset
- **Source:** [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Train:** 100,000 images
- **Validation:** 20,000 images
- **Test:** 20,000 images
- **Classes:** `fake` (GAN-generated faces), `real` (genuine photos)

## Training
The full training pipeline is in `deepfake-detection.ipynb`, which covers:
- Dataset loading and preprocessing
- Data augmentation (random flip, rotation, color jitter)
- Model fine-tuning with EfficientNet-B0
- Best model checkpoint saving based on validation accuracy

Training was done for **5 epochs** with the following progression:

| Epoch | Train Acc | Val Acc  | Val Loss |
|-------|-----------|----------|----------|
| 1     | 92.92%    | 98.21%   | 0.0485   |
| 2     | 97.38%    | 98.95%   | 0.0306   |
| 3     | 98.18%    | 99.37%   | 0.0184   |
| 4     | 98.71%    | 98.74%   | 0.0370   |
| 5     | **98.89%**    | **99.54%** | **0.0134** |

## Results
Evaluated on **20,000 test images**:

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 99.41%  |
| F1 Score    | 99.41%  |
| AUC-ROC     | 0.9999  |

### Per-Class Classification Report
| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Fake    | 0.99      | 1.00   | 0.99     | 10,000  |
| Real    | 1.00      | 0.99   | 0.99     | 10,000  |
| **Avg** | **0.99**  | **0.99** | **0.99** | **20,000** |
