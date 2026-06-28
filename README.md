<img width="1536" height="1024" alt="技术全景" src="https://github.com/user-attachments/assets/8ff6ddf3-721b-4159-b919-1e8a44291ceb" />
<img width="1536" height="1024" alt="技术全景" src="https://github.com/user-attachments/assets/d6e2f820-3f52-4b3d-b46c-49a943614e59" />
# CogPepML - Cognitive Peptide Machine Learning

A LightGBM-based model for predicting bioactive peptides with potential for preventing cognitive impairment, using ESM-2 150M (640-dim) + shallow physicochemical features (426-dim) = 1066-dim total.

## Project Structure

```
├── 00_data/                              # Training data (80/10/10 stratified split)
│   ├── train.csv                         # 6910 positive + negative samples
│   ├── val.csv                           # 864 validation samples
│   └── test.csv                          # 864 test samples (held out)
├── 01_features/
│   ├── feature_extraction_guide.md        # How features are computed + reference code
│   └── precomputed/                       # Pre-extracted 1066-dim features (ready to use)
├── 02_optimal_model/                     # ★ Best model: Test AUC = 0.9405
│   ├── optuna_search.py                  # Optuna 300-trial x 5-fold CV search script
│   ├── model/LightGBM_Optuna_best.pkl    # Trained model file
│   ├── search_logs/                      # Trial log, convergence trend, best value evolution
│   └── evaluation/                       # Reports, 10-fold CV, confusion matrix, etc.
├── 03_calibration_isotonic/              # Isotonic Regression calibration
│   ├── calibrate.py                      # 80% train + 20% calibrate
│   ├── calibration_notes.txt              # Method description and results
│   ├── model/LightGBM_Isotonic_calibrated.pkl
│   ├── cognitive_peptide/                 # External validation: cognitive peptides
│   └── storage_protein/                   # External validation: storage proteins + BBB
├── 04_calibration_platt/                 # Platt Scaling (Sigmoid) calibration
│   ├── calibrate.py                      # 80% train + 20% calibrate
│   ├── calibration_notes.txt
│   ├── model/LightGBM_Platt_calibrated.pkl
│   └── storage_protein/                  # Full 384-row predictions + ≥5aa active list
└── 05_prediction/
    └── predict_peptide.py               # Load model and predict new peptide sequences
```

## Model Performance

| Metric | Value |
|:------:|:-----:|
| **Test AUC** | **0.9405** |
| Test ACC | 0.8588 |
| Test F1 | 0.8697 |
| Test MCC | 0.7167 |

### 10-fold Cross-Validation

| Range | Mean | Std |
|:----:|:----:|:---:|
| 0.9210 ~ 0.9359 | 0.9292 | 0.0053 |

### Confusion Matrix (864 test samples)

```
           Pred Neg  Pred Pos
True Neg     335       72      Specificity = 82.31%
True Pos      50      407      Sensitivity = 89.06%
```

## Quick Start

### Installation

```bash
pip install lightgbm scikit-learn pandas numpy torch fair-esm optuna tqdm openpyxl
```

### Reproduce the full pipeline

```bash
# Step 1: [Optional] Re-extract 1066-dim features (see feature_extraction_guide.md)
# Step 2: Optuna hyperparameter search
cd 02_optimal_model
python optuna_search.py

# Step 3: Isotonic calibration
cd ../03_calibration_isotonic
python calibrate.py

# Step 4: Platt calibration
cd ../04_calibration_platt
python calibrate.py
```

### Predict new peptides

```bash
cd 05_prediction

# Single sequence (defaults to Isotonic calibrated model)
python predict_peptide.py "LLYQQPV"

# Batch prediction from CSV
python predict_peptide.py --csv input.csv --output result.csv
```

## Features

Each peptide is represented as a **1066-dimensional** feature vector:

| Feature | Dim | Description |
|:-------:|:---:|-------------|
| ESM-2 150M | 640 | 30-layer Transformer, pretrained on 250M protein sequences |
| AA Composition | 20 | Frequencies of 20 standard amino acids |
| Dipeptide Composition | 400 | Frequencies of 400 dipeptide pairs |
| Physicochemical | 6 | Net charge, hydrophobicity (KD), MW, pI, aromaticity, flexibility |

All precomputed features are provided in `01_features/precomputed/` and ready for immediate use.

## Hyperparameter Search

| Item | Value |
|:----|:-----:|
| Algorithm | Optuna TPE (Tree-structured Parzen Estimator) |
| Trials | 300 × 5-fold CV = 1500 training runs |
| Search Space | 16 hyperparameters |
| Time | ~280 min (CPU) |

### Top-5 Important Hyperparameters

| Rank | Parameter | Importance | Description |
|:---:|:---------:|:----------:|-------------|
| 1 | **max_delta_step** | **0.5054** | Step size limit per iteration |
| 2 | min_gain_to_split | 0.1769 | Minimum split gain |
| 3 | lambda_l1 | 0.1364 | L1 regularization |
| 4 | extra_trees | 0.0562 | Extra Trees mode |
| 5 | min_child_samples | 0.0242 | Min leaf samples |

## Probability Calibration

Standard LightGBM's 1000-tree voting causes probability saturation (many 0.999+). Calibration is required for publication-ready probability values.

### Calibration Effect (864 test samples)

| Metric | Before | Isotonic | Platt |
|:-----:|:------:|:--------:|:-----:|
| Min | 0.0000 | 0.0000 | **0.0771** |
| Max | 1.0000 | 1.0000 | **0.9078** |
| Mean | 0.5539 | 0.5285 | **0.5271** |

### Which Calibration to Use

| Method | Pros | Cons | Recommendation |
|:------|:----|:-----|:--------------|
| **Platt (Sigmoid)** | Stable, few params, good extrapolation | Sigmoid ceiling on 1.0 | **Paper publication** |
| **Isotonic** | No shape assumption | Overfits small data, no extrapolation | Ranking only |

## Citation

If you use this work, please cite:

```
[To be added — paper under preparation]
```

## Requirements

```
Python 3.8+
lightgbm, scikit-learn, pandas, numpy
torch, fair-esm
optuna, tqdm, openpyxl
```
