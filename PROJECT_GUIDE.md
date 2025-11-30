# NER Project Configuration & User Guide

This document provides a detailed overview of the refactored NER project, explaining the code structure, configuration options, and how to run different models.

## 1. Project Structure Overview

The project is organized into modular components to separate data, models, and training logic.

```
ML-Assignment/
├── data/                  # Data handling
│   ├── dataset.py         # Loads CoNLL-2003 dataset and calculates statistics
│   └── preprocessing.py   # Feature extraction (CRF) and tensor preparation (BiLSTM)
│
├── models/                # Model definitions
│   ├── crf_model.py       # Wrapper for sklearn-crfsuite
│   └── bilstm_crf.py      # PyTorch implementation of BiLSTM-CRF
│
├── trainers/              # Training loops
│   ├── crf_trainer.py     # Manages CRF training and cross-validation
│   └── bilstm_trainer.py  # Manages BiLSTM training epochs and evaluation
│
├── utils/                 # Helper functions
│   ├── metrics.py         # F1, Precision, Recall calculations (using seqeval)
│   └── visualization.py   # Plotting functions (loss curves, confusion matrices)
│
├── config/                # Configuration
│   └── config.py          # Central place for default hyperparameters
│
├── main.py                # Main entry point script
├── requirements.txt       # Python dependencies
└── run_test.bat           # Quick verification script
```

## 2. Configuration Parameters

The project uses a dual configuration system:
1.  **Default Defaults**: Defined in `config/config.py`.
2.  **Command Line Overrides**: You can change any parameter when running `main.py`.

### CRF Parameters
| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| **L1 Regularization** | `--c1` | `0.1` | Controls feature selection sparsity. |
| **L2 Regularization** | `--c2` | `0.1` | Prevents overfitting. |
| **Max Iterations** | `--max_iter` | `100` | Maximum number of optimization steps. |

### BiLSTM-CRF Parameters
| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| **Epochs** | `--epochs` | `20` | Number of training passes over the dataset. |
| **Batch Size** | `--batch_size` | `32` | Number of samples per gradient update. |
| **Learning Rate** | `--lr` | `0.001` | Step size for the Adam optimizer. |
| **Hidden Dimension** | `--hidden_dim` | `256` | Size of the LSTM hidden layer. |
| **Embedding Dim** | *(in config.py)* | `100` | Size of word embeddings. |
| **Dropout** | *(in config.py)* | `0.5` | Dropout probability for regularization. |

## 3. How to Run

### Prerequisite
Ensure dependencies are installed:

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Using Conda (Recommended)**
```bash
# Create a new environment
conda create -n ner_env python=3.8

# Activate the environment
conda activate ner_env

# Install dependencies
pip install -r requirements.txt
```

### Scenario A: Train CRF Model
To train the CRF model with default settings:
```bash
python main.py --model crf --mode train
```

To customize parameters (e.g., more iterations):
```bash
python main.py --model crf --mode train --max_iter 200 --c1 0.2
```

### Scenario B: Train BiLSTM-CRF Model
To train the Deep Learning model:
```bash
python main.py --model bilstm --mode train
```

To train for longer with a smaller batch size:
```bash
python main.py --model bilstm --mode train --epochs 50 --batch_size 16
```

### Scenario C: Cross-Validation (CRF Only)
To run 5-fold cross-validation to check model stability:
```bash
python main.py --model crf --mode cv
```

### Scenario D: Visualization
To generate dataset statistics and plots (Entity Distribution, Sentence Lengths):
```bash
python main.py --mode visualize
```
This will print a summary table to the console and save plots to `outputs/`.

## 4. Output Files

Running the project will create an `outputs/` directory containing:
*   **`crf_model.joblib`**: Saved CRF model.
*   **`best_bilstm_crf.pt`**: Saved BiLSTM model (best weights).
*   **`*.png`**: Visualization plots (training history, entity distribution).

## 5. Code Details

*   **`data/preprocessing.py`**:
    *   `word2features`: Exactly matches your notebook's feature extraction (suffixes, capitalization, POS tags, etc.).
    *   `prepare_sequence`: Converts words to integer IDs for the BiLSTM.
*   **`models/bilstm_crf.py`**:
    *   Contains the `BiLSTM_CRF` class. It uses an Embedding layer, a bidirectional LSTM, a Linear layer, and finally a CRF layer (from `pytorch-crf`).
*   **`utils/metrics.py`**:
    *   Uses `seqeval` to calculate entity-level F1 scores, ensuring strict evaluation standard for NER tasks.
