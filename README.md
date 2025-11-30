# NER Project Refactoring

This project implements Named Entity Recognition (NER) using the CoNLL-2003 dataset. It provides a modular Python structure for training and evaluating CRF and BiLSTM-CRF models.

## Project Structure

```
ML-Assignment/
├── data/               # Data loading and preprocessing
├── models/             # Model implementations (CRF, BiLSTM-CRF)
├── utils/              # Metrics and visualization
├── config/             # Configuration management
├── trainers/           # Training logic
python main.py --model crf --mode train
```

### Train BiLSTM-CRF Model
```bash
python main.py --model bilstm --mode train --epochs 20
```

### Cross-Validation (CRF)
```bash
python main.py --model crf --mode cv
```

## Configuration
You can adjust hyperparameters via command line arguments:
- `--c1`, `--c2`: CRF regularization parameters
- `--epochs`, `--batch_size`, `--lr`: BiLSTM training parameters
- `--seed`: Random seed for reproducibility

## Outputs
Models and plots are saved in the `outputs/` directory.