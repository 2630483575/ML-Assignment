# NER Project Refactoring

This project implements Named Entity Recognition (NER) using the CoNLL-2003 dataset. It provides a modular Python structure for training and evaluating CRF, BiLSTM-CRF, and BERT-based models.

## Project Structure

```
ML-Assignment/
├── data/               # Data loading and preprocessing
├── models/             # Model implementations (CRF, BiLSTM-CRF, BERT)
├── utils/              # Metrics and visualization
├── config/             # Configuration management
├── trainers/           # Training logic (CRF, BiLSTM, BERT)
├── main.py             # Main entry point
├── requirements.txt    # Dependencies
└── outputs/            # Saved models and visualizations
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Train CRF Model
```bash
python main.py --model crf --mode train
```

### Train BiLSTM-CRF Model
```bash
python main.py --model bilstm --mode train --epochs 20
```

### Train BERT Model
```bash
python main.py --model bert --mode train --epochs 3 --batch_size 16
```

### Generate Visualizations
```bash
python main.py --mode visualize
```

### Cross-Validation (CRF)
```bash
python main.py --model crf --mode cv
```

## Configuration

### CRF Parameters
- `--c1`, `--c2`: L1/L2 regularization parameters (default: 0.1)
- `--max_iter`: Maximum optimization iterations (default: 100)

### BiLSTM-CRF / BERT Parameters
- `--epochs`: Number of training epochs (default: 20 for BiLSTM, 3 for BERT)
- `--batch_size`: Batch size (default: 32 for BiLSTM, 16 for BERT)
- `--lr`: Learning rate (default: 0.001 for BiLSTM, 2e-5 for BERT)
- `--hidden_dim`: LSTM hidden dimension (default: 256)

### Advanced BERT Parameters
Defined in `config/config.py`:
- `model_name`: Pre-trained model (default: "bert-base-cased")
- `weight_decay`: 0.01
- `warmup_steps`: 500

### General Parameters
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Output directory (default: "outputs")

## Outputs
Models and plots are saved in the `outputs/` directory:
- `crf_model.joblib`: Trained CRF model
- `best_bilstm_crf.pt`: Trained BiLSTM-CRF model
- `best_bert_model/`: Trained BERT model and tokenizer
- `*.png`: Visualization plots