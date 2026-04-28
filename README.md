# Transfer Learning for Brain Tumor MRI Classification

A systematic investigation of transfer learning strategies for brain tumor classification using pretrained convolutional neural networks on MRI scan data.

## Overview

This project evaluates the effectiveness of different layer freezing strategies and training data sizes for classifying brain tumor MRI images across four diagnostic classes: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**.

Two pretrained architectures are compared:
- **ResNet-50** (~25M parameters)
- **EfficientNet-B3** (~12M parameters)

## Experiments

- 2 models × 3 data splits × 5 freeze levels = **30 total runs**
- Freeze levels: 0%, 25%, 50%, 75%, 100%
- Data splits: 25%, 50%, 100% of training data
- All models pretrained on ImageNet

## Results Summary

| Model | 0% Frozen | 25% Frozen | 50% Frozen | 75% Frozen | 100% Frozen |
|---|---|---|---|---|---|
| ResNet-50 | 95.69% | 95.75% | 95.94% | 95.25% | 77.81% |
| EfficientNet-B3 | 95.63% | 95.88% | 95.63% | 95.19% | 79.31% |

Both models exceed **95% validation accuracy** with full fine-tuning and maintain strong performance (~92%+) with only 25% of training data, demonstrating the practical value of transfer learning for medical imaging applications with limited labeled data.

## Dataset

This project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle containing approximately 7,000 MRI brain scans across four diagnostic classes.

Once downloaded, extract to the following structure:
```
data/
└── brain_tumor/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```
## Installation

**1. Install dependencies:**
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn kaggle
```

**2. Download the dataset from Kaggle:**

Option A — Kaggle API:
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/brain_tumor
```

Option B — Manual download:
1. Go to https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click Download
3. Extract to `data/brain_tumor/` following the folder structure above

## Usage

**Run all experiments:**
```bash
python experiment_brain.py
```

The experiment runner includes resume/skip logic — if interrupted, it will automatically skip completed runs and continue from where it left off.

**Generate plots:**
```bash
python plot_brain.py
```

Results are saved to `results_brain/brain_results.csv` and plots to `results_brain/plots/`.

## Project Structure
```
├── config_brain.py        # Hyperparameters and experiment settings
├── data_brain.py          # Dataset loading and preprocessing
├── model_brain.py         # Model loading and layer freezing logic
├── train_brain.py         # Training and validation loops
├── experiment_brain.py    # Main experiment runner
├── plot_brain.py          # Result visualization
└── results_brain/
├── brain_results.csv  # All experiment results
└── plots/             # Generated figures
```
## Key Findings

- Full fine-tuning (0% frozen) achieves highest accuracy but partial fine-tuning (25–50% frozen) offers the best accuracy-efficiency tradeoff
- Both models maintain >92% accuracy with only 25% of training data
- Performance drops sharply at 100% freeze, confirming feature extraction alone is insufficient for medical imaging tasks with significant domain shift from ImageNet
- EfficientNet-B3 marginally outperforms ResNet-50 at full feature extraction (79.31% vs 77.81%), suggesting its compact representations generalize slightly better in dissimilar domains

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU recommended

## Course

CSC 7760 - Deep Learning | Wayne State University | April 2026
