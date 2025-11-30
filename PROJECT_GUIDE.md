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
│   ├── bilstm_crf.py      # PyTorch implementation of BiLSTM-CRF
│   └── bert_model.py      # BERT for token classification wrapper
│
├── trainers/              # Training loops
│   ├── crf_trainer.py     # Manages CRF training and cross-validation
│   ├── bilstm_trainer.py  # Manages BiLSTM training epochs and evaluation
│   └── bert_trainer.py    # Manages BERT training using Hugging Face Trainer
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

### BERT Parameters
| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| **Model Name** | `config.py` | `bert-base-cased` | Pre-trained BERT model identifier from Hugging Face. |
| **Learning Rate** | `--lr` | `2e-5` | Fine-tuning learning rate (typical for BERT). |
| **Batch Size** | `--batch_size` | `16` | Training batch size per device. |
| **Epochs** | `--epochs` | `3` | Number of fine-tuning epochs. |
| **Weight Decay** | `config.py` | `0.01` | L2 regularization strength. |
| **Warmup Steps** | `config.py` | `500` | Number of warmup steps for learning rate scheduler. |

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

### Scenario A: Generate Data Visualizations
To generate plots for entity distribution and sentence lengths:
```bash
python main.py --mode visualize
```
This will save `entity_distribution.png` and `sentence_length.png` to the `outputs/` directory.

### Scenario B: Train CRF Model
To train the CRF model with default settings:
```bash
python main.py --model crf --mode train
```

To customize parameters (e.g., more iterations):
```bash
python main.py --model crf --mode train --max_iter 200 --c1 0.2
```

### Scenario C: Train BiLSTM-CRF Model
To train the BiLSTM-CRF model:
```bash
python main.py --model bilstm --mode train --epochs 20 --batch_size 32
```

### Scenario D: Train BERT Model
To fine-tune the BERT model:
```bash
python main.py --model bert --mode train --epochs 3 --batch_size 16
```

### Scenario E: Cross-Validation (CRF Only)
To run 5-fold cross-validation to check model stability:
```bash
python main.py --model crf --mode cv
```

## 4. Output Files

Running the project will create an `outputs/` directory containing:
*   **`crf_model.joblib`**: Saved CRF model.
*   **`best_bilstm_crf.pt`**: Saved BiLSTM model (best weights).
*   **`best_bert_model/`**: Saved BERT model directory (includes model weights and tokenizer).
*   **`bert_checkpoints/`**: Training checkpoints for BERT (created during training).
*   **`logs/`**: TensorBoard logs for BERT training.
*   **`*.png`**: Visualization plots (training history, entity distribution).

## 5. Code Details

*   **`data/preprocessing.py`**:
    *   `word2features`: Exactly matches your notebook's feature extraction (suffixes, capitalization, POS tags, etc.).
    *   `prepare_sequence`: Converts words to integer IDs for the BiLSTM.
*   **`models/crf_model.py`**:
    *   Wrapper around `sklearn-crfsuite` for easy training and prediction with feature-based CRF.
*   **`models/bilstm_crf.py`**:
    *   Contains the `BiLSTM_CRF` class. It uses an Embedding layer, a bidirectional LSTM, a Linear layer, and finally a CRF layer (from `pytorch-crf`).
*   **`models/bert_model.py`**:
    *   Wrapper for `BertForTokenClassification` from Hugging Face Transformers.
    *   Automatically loads pre-trained BERT model and tokenizer.
    *   Handles model saving/loading with tokenizer.
*   **`trainers/bert_trainer.py`**:
    *   Uses Hugging Face `Trainer` API for efficient fine-tuning.
    *   Implements custom metrics computation compatible with seqeval.
    *   Supports automatic checkpoint saving and best model selection.
*   **`utils/metrics.py`**:
    *   Uses `seqeval` to calculate entity-level F1 scores, ensuring strict evaluation standard for NER tasks.

## 6. Model Comparison

| Model | Training Time | Performance | Memory Usage | Best For |
|-------|---------------|-------------|--------------|----------|
| **CRF** | Fast (~minutes) | Good baseline | Low | Feature engineering, interpretability |
| **BiLSTM-CRF** | Medium (~1 hour) | Better accuracy | Medium | Learning contextual patterns |
| **BERT** | Slow (~hours) | State-of-the-art | High (GPU recommended) | Maximum accuracy, transfer learning |

## 7. Dependencies

Key libraries used in this project:
- **`transformers`**: Hugging Face library for BERT and other transformer models
- **`torch`**: PyTorch for BiLSTM-CRF and BERT
- **`sklearn-crfsuite`**: CRF implementation
- **`pytorch-crf`**: CRF layer for BiLSTM-CRF
- **`seqeval`**: NER-specific evaluation metrics
- **`datasets`**: Hugging Face datasets library for loading CoNLL-2003
