# NER Project

Modular Named Entity Recognition (NER) pipeline for the CoNLL-2003 benchmark. The codebase trains and evaluates CRF, BiLSTM-CRF, BERT, and RoBERTa models, runs ablation studies, generates visual reports, and performs dimensionality-reduction diagnostics.

## Project Structure

```
ML-Assignment/
├── config/
│   └── config.py           # Configuration classes (ProjectConfig, ModelConfig)
├── data/
│   ├── dataset.py          # CoNLL-2003 loader
│   ├── preprocessing.py    # Feature extraction & tokenization
│   └── data_analysis.py    # EDA & statistics
├── models/
│   ├── crf_model.py        # sklearn-crfsuite wrapper
│   ├── bilstm_crf.py       # PyTorch BiLSTM-CRF
│   ├── bert_model.py       # Hugging Face BERT wrapper
│   └── roberta_model.py    # Hugging Face RoBERTa wrapper
├── trainers/
│   ├── crf_trainer.py      # CRF training logic
│   ├── bilstm_trainer.py   # BiLSTM training loop
│   ├── bert_trainer.py     # BERT fine-tuning (HF Trainer)
│   └── roberta_trainer.py  # RoBERTa fine-tuning
├── scripts/
│   ├── run_data_analysis.py      # Generate EDA report
│   ├── generate_visualizations.py # Create plots from results
│   ├── run_cross_validation.py   # K-Fold CV
│   ├── compare_features.py       # Feature ablation
│   └── analyze_embeddings.py     # PCA/t-SNE analysis
├── utils/
│   ├── metrics.py          # Seqeval metrics (F1, Precision, Recall)
│   └── visualization.py    # Plotting utilities
├── outputs/                # Artifacts (models, logs, plots)
├── main.py                 # CLI Entry point
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Environment Setup

1. **Python version**: 3.9–3.11 works (project tested on 3.10).
2. **Create a virtual environment (PowerShell)**
	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```
3. **Install dependencies**
	```powershell
	pip install -r requirements.txt
	```
4. **Verify optional dimensionality-reduction stack**
	```powershell
	python scripts/verify_setup.py
	```
	The script checks `numpy`, `matplotlib`, `scikit-learn`, and optional `umap`. Install any missing package and rerun.

All runtime settings (batch size, epochs, learning rates, etc.) can be edited through CLI flags or directly inside `config/config.py`.

## Dataset Preparation

| Step | Instruction |
|------|-------------|
| Source | CoNLL-2003 (English) on Hugging Face Datasets: https://huggingface.co/datasets/conll2003 (mirrors the original https://www.clips.uantwerpen.be/conll2003/ner/ release). |
| Download | Run once to cache every split locally:<br>`python -c "from datasets import load_dataset; load_dataset('conll2003', revision='refs/convert/parquet')"` |
| Default location | Hugging Face stores the parquet shards under `%USERPROFILE%\.cache\huggingface\datasets\conll2003\`. Leave them there; `data/dataset.py` loads from this cache automatically. |
| Optional project-local copy | If you must keep raw `.txt` files inside the repo, mirror the structure below (these files are not required unless you customize `load_conll2003()`):<br>
```
ML-Assignment/
└── data/
	 └── raw/
		  └── conll2003/
				├── train.txt
				├── valid.txt
				└── test.txt
``` |
| Preprocess / sanity check | Generate the exploratory report and POS/NER statistics:<br>`python scripts/run_data_analysis.py --output_dir outputs/data_analysis`<br>This produces `outputs/data_analysis/*.png` plus `my_analysis/data_analysis_report.md`. |

> **Note**: No manual preprocessing is needed beyond the command above—the Hugging Face loader yields tokenized sentences and NER tags. Feature extractors in `data/preprocessing.py` operate directly on that format.

## Running the Pipeline

All commands are Windows PowerShell-friendly. Add `--output_dir <path>` to redirect artifacts if needed.

### 1. Health checks & analytics

```powershell
python scripts/verify_setup.py                              # dependency sanity check
python scripts/run_data_analysis.py --output_dir my_analysis # Generate data analysis report
```

### 2. Train Models (Generates artifacts for visualization)

Run these commands to train models and generate the necessary `.json` files (history and predictions) for the visualization suite.

```powershell
# 1. CRF Baseline
python main.py --model crf --mode train

# 2. BiLSTM-CRF
python main.py --model bilstm --mode train

# 3. BERT Fine-tuning
python main.py --model bert --mode train

# 4. RoBERTa Fine-tuning
python main.py --model roberta --mode train
```

### 3. Generate Visualizations

After training the models, run this script to generate comprehensive plots (Confusion Matrices, Training Curves, Model Comparison).

```powershell
python scripts/generate_visualizations.py
```
The visualizations will be saved in `outputs/visualizations/`.

### 4. Advanced Experiments (Optional)

```powershell
# Feature Ablation Study (CRF)
python scripts/compare_features.py --quick

# Cross-Validation
python scripts/run_cross_validation.py --model crf --n_splits 5
python scripts/run_cross_validation.py --model bilstm --n_splits 3 --quick

# Dimensionality Reduction Analysis (PCA/t-SNE/UMAP)
# Visualizes how models cluster different entity types in the embedding space
python scripts/analyze_embeddings.py --model both --max_samples 1000
```

Each command emits progress to the console and writes artifacts into `outputs/`, including:

- `best_bilstm_crf.pt`, `best_bert_model/`, `best_roberta_model/`, `crf_model.joblib`
- `*_results.json`, `*_cv_results.json`, `feature_comparison_results.json`
- `outputs/visualizations/*.png` (confusion matrices, per-label performance, training curves)
- `outputs/analysis/*.png` (PCA/t-SNE/UMAP + clustering)
- `my_plots/index.html` gallery for quick inspection

## Configuration Reference

- **Global**: `--seed`, `--output_dir`
- **CRF**: `--c1`, `--c2`, `--max_iter`
- **BiLSTM**: `--epochs`, `--batch_size`, `--lr`, `--hidden_dim`
- **BERT/RoBERTa**: share the `--epochs`, `--batch_size`, `--lr` flags (defaults overridden to 2e-5 for BERT/RoBERTa inside `main.py` if you keep `0.001`). Extra knobs—model name, warmup, weight decay—live under `config/config.py`.

Adjusting these flags plus the scripts above covers every experiment reproduced in `Experiment_Report.md`.

## Troubleshooting

- **Datasets download errors**: ensure `pip install datasets` succeeded. Behind a proxy, set `HF_ENDPOINT` or download manually from the URLs listed above and keep the tarball under `%USERPROFILE%\.cache\huggingface\datasets`.
- **CUDA Out Of Memory**: lower `--batch_size` for BERT/RoBERTa or enable gradient accumulation inside `config/bert`.
- **Missing UMAP**: install via `pip install umap-learn` or skip UMAP plots (script handles the absence gracefully).