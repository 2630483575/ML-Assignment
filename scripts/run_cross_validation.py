"""
Unified script to run cross-validation for all models.

This script provides comprehensive cross-validation functionality including:
- Standard K-fold cross-validation for all models
- Grid search hyperparameter tuning
- Results comparison across models
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_conll2003, get_label_mappings
from config.config import ProjectConfig
from trainers.crf_trainer import CRFTrainer
from utils.cross_validation import compare_cv_results
import json


def run_crf_cv(config, dataset, n_splits=5, with_grid_search=False):
    """Run cross-validation for CRF model."""
    print("\n" + "="*80)
    print("CRF CROSS-VALIDATION")
    print("="*80)
    
    label_list, _, _ = get_label_mappings(dataset)
    trainer = CRFTrainer(config, label_list)
    
    if with_grid_search:
        print("\nRunning grid search with cross-validation...")
        param_grid = {
            'c1': [0.01, 0.1, 0.5],
            'c2': [0.01, 0.1, 0.5],
            'max_iterations': [100]
        }
        best_params, all_results = trainer.grid_search_cv(
            dataset,
            param_grid=param_grid,
            n_splits=n_splits
        )
        
        # Extract best CV results
        best_result = next(r for r in all_results if r['params'] == best_params)
        cv_results = best_result['cv_results']
        
        print(f"\nBest Parameters: {best_params}")
        print(f"Best F1 Score: {cv_results['mean']['f1']:.4f} ± {cv_results['std']['f1']:.4f}")
        
        return cv_results
    else:
        trainer.cross_validate(dataset, n_splits=n_splits)
        return None


def run_bilstm_cv(config, dataset, n_splits=5, quick_mode=False):
    """Run cross-validation for BiLSTM model."""
    print("\n" + "="*80)
    print("BiLSTM CROSS-VALIDATION")
    print("="*80)
    
    from trainers.bilstm_trainer import BiLSTMTrainer
    from data.preprocessing import build_vocab, NERDataset, collate_fn
    from utils.cross_validation import CrossValidator
    from models.bilstm_crf import BiLSTM_CRF
    import torch
    from torch.utils.data import DataLoader
    from utils.metrics import evaluate_model
    from data.preprocessing import prepare_sequence, sent2labels
    
    label_list, _, _ = get_label_mappings(dataset)
    
    # Build vocabulary
    print("Building vocabulary...")
    word2idx = build_vocab(dataset['train'])
    tag2idx = {tag: idx for idx, tag in enumerate(label_list)}
    
    # Prepare full training data
    print("Preparing training data...")
    train_data = []
    for example in dataset['train']:
        train_data.append(example)
    
    # Quick mode: reduce epochs for faster testing
    epochs = 5 if quick_mode else config.bilstm.epochs
    
    # Define train and eval functions
    def train_fn(data, params):
        # Prepare data for DataLoader
        processed_data = [prepare_sequence(s, word2idx, tag2idx, label_list) for s in data]
        dataset = NERDataset(processed_data)
        dataloader = DataLoader(dataset, batch_size=config.bilstm.batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Initialize trainer
        trainer = BiLSTMTrainer(config, label_list)
        trainer.word2idx = word2idx
        trainer.tag2idx = tag2idx
        trainer.idx2tag = {i: t for t, i in tag2idx.items()}
        
        # Update config with params if needed
        original_lr = config.bilstm.learning_rate
        original_hidden = config.bilstm.hidden_dim
        
        if 'lr' in params: config.bilstm.learning_rate = params['lr']
        if 'hidden_dim' in params: config.bilstm.hidden_dim = params['hidden_dim']
        
        # Setup model
        trainer.setup_model(len(word2idx), tag2idx)
        
        # Train
        for epoch in range(epochs):
            trainer.train_epoch(dataloader)
            
        # Restore config
        config.bilstm.learning_rate = original_lr
        config.bilstm.hidden_dim = original_hidden
        
        return trainer.model
    
    def eval_fn(model, data):
        # Prepare data
        processed_data = [prepare_sequence(s, word2idx, tag2idx, label_list) for s in data]
        dataset = NERDataset(processed_data)
        dataloader = DataLoader(dataset, batch_size=config.bilstm.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Use trainer for evaluation
        trainer = BiLSTMTrainer(config, label_list)
        trainer.model = model
        trainer.word2idx = word2idx
        trainer.tag2idx = tag2idx
        trainer.idx2tag = {i: t for t, i in tag2idx.items()}
        
        return trainer.evaluate(dataloader)
    
    # Run cross-validation
    cv = CrossValidator(n_splits=n_splits, random_state=config.seed)
    
    params = {
        'embedding_dim': config.bilstm.embedding_dim,
        'hidden_dim': config.bilstm.hidden_dim,
        'lr': config.bilstm.learning_rate
    }
    
    cv_results = cv.cross_validate(
        train_fn=train_fn,
        eval_fn=eval_fn,
        dataset=train_data,
        params=params,
        verbose=True
    )
    
    # Manually set cv_results for plotting since we didn't use grid_search
    cv.cv_results = [{
        'params': params,
        'mean_score': cv_results['mean']['f1'],
        'std_score': cv_results['std']['f1'],
        'cv_results': cv_results
    }]
    
    # Save results
    results_path = os.path.join(config.output_dir, "bilstm_cv_results.json")
    cv.save_results(results_path)
    
    # Plot results
    plot_path = os.path.join(config.output_dir, "bilstm_cv_plots.png")
    cv.plot_cv_results(metric_name='f1', save_path=plot_path)
    
    return cv_results


def run_bert_cv(config, dataset, n_splits=3, quick_mode=True):
    """
    Run cross-validation for BERT model.
    
    Note: BERT CV is very slow, so we use fewer folds by default.
    Quick mode uses 1 epoch per fold for faster testing.
    """
    print("\n" + "="*80)
    print("BERT CROSS-VALIDATION (This will take a while...)")
    print("="*80)
    
    from models.bert_model import BERTModel
    from utils.cross_validation import CrossValidator
    from data.preprocessing import tokenize_and_align_labels
    from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
    from utils.metrics import evaluate_model
    import torch
    
    label_list, label2id, id2label = get_label_mappings(dataset)
    
    # Prepare training data
    print("Preparing data...")
    train_data = list(dataset['train'])
    
    # Quick mode: 1 epoch, otherwise 2 epochs
    epochs_per_fold = 1 if quick_mode else 2
    
    def train_fn(data, params):
        # Create temporary dataset
        from datasets import Dataset as HFDataset
        
        temp_dataset = HFDataset.from_dict({
            'id': [str(i) for i in range(len(data))],
            'tokens': [example['tokens'] for example in data],
            'ner_tags': [example['ner_tags'] for example in data],
            'pos_tags': [example.get('pos_tags', [0]*len(example['tokens'])) for example in data],
            'chunk_tags': [example.get('chunk_tags', [0]*len(example['tokens'])) for example in data]
        })
        
        # Initialize model
        model_wrapper = BERTModel(config, len(label_list), id2label, label2id)
        
        # Tokenize
        def tokenize_fn(examples):
            return tokenize_and_align_labels(examples, model_wrapper.tokenizer)
        
        tokenized_data = temp_dataset.map(tokenize_fn, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(config.output_dir, "bert_cv_temp"),
            num_train_epochs=epochs_per_fold,
            per_device_train_batch_size=config.bert.batch_size,
            save_strategy="no",
            logging_steps=50,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=model_wrapper.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model_wrapper.model,
            args=training_args,
            train_dataset=tokenized_data,
            data_collator=data_collator
        )
        
        # Train
        trainer.train()
        
        return model_wrapper
    
    def eval_fn(model_wrapper, data):
        # Similar to train_fn, create temp dataset
        from datasets import Dataset as HFDataset
        
        temp_dataset = HFDataset.from_dict({
            'id': [str(i) for i in range(len(data))],
            'tokens': [example['tokens'] for example in data],
            'ner_tags': [example['ner_tags'] for example in data],
            'pos_tags': [example.get('pos_tags', [0]*len(example['tokens'])) for example in data],
            'chunk_tags': [example.get('chunk_tags', [0]*len(example['tokens'])) for example in data]
        })
        
        def tokenize_fn(examples):
            return tokenize_and_align_labels(examples, model_wrapper.tokenizer)
        
        tokenized_data = temp_dataset.map(tokenize_fn, batched=True)
        
        # Make predictions
        model_wrapper.model.eval()
        
        y_true_all = []
        y_pred_all = []
        
        from torch.utils.data import DataLoader
        from transformers import DataCollatorForTokenClassification
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=model_wrapper.tokenizer,
            padding=True
        )
        
        dataloader = DataLoader(tokenized_data, batch_size=16, collate_fn=data_collator)
        
        with torch.no_grad():
            for idx, example in enumerate(data):
                # Get predictions for this example
                inputs = model_wrapper.tokenizer(
                    example['tokens'],
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Get word_ids before converting to dict
                word_ids = inputs.word_ids(batch_index=0)
                
                # Move inputs to device
                inputs = {k: v.to(model_wrapper.device) for k, v in inputs.items()}
                
                outputs = model_wrapper.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
                
                # Align predictions with tokens
                pred_labels = []
                previous_word_idx = None
                
                for i, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue
                    elif word_idx != previous_word_idx:
                        # Only take the label of the first token of each word
                        pred_labels.append(id2label[predictions[i]])
                        previous_word_idx = word_idx
                
                true_labels = [label_list[tag] for tag in example['ner_tags']]
                
                # Trim to same length
                min_len = min(len(pred_labels), len(true_labels))
                y_pred_all.append(pred_labels[:min_len])
                y_true_all.append(true_labels[:min_len])
        
        metrics = evaluate_model(y_true_all, y_pred_all)
        return metrics
    
    # Run cross-validation with fewer folds for BERT
    cv = CrossValidator(n_splits=n_splits, random_state=config.seed)
    
    params = {}
    
    print(f"\nRunning {n_splits}-fold CV with {epochs_per_fold} epoch(s) per fold...")
    print("This may take 30-60 minutes depending on your hardware.\n")
    
    cv_results = cv.cross_validate(
        train_fn=train_fn,
        eval_fn=eval_fn,
        dataset=train_data,
        params=params,
        verbose=True
    )
    
    # Manually set cv_results for plotting
    cv.cv_results = [{
        'params': params,
        'mean_score': cv_results['mean']['f1'],
        'std_score': cv_results['std']['f1'],
        'cv_results': cv_results
    }]
    
    # Save results
    results_path = os.path.join(config.output_dir, "bert_cv_results.json")
    cv.save_results(results_path)
    
    # Plot results
    plot_path = os.path.join(config.output_dir, "bert_cv_plots.png")
    cv.plot_cv_results(metric_name='f1', save_path=plot_path)
    
    return cv_results


def run_all_cv(config, dataset, n_splits=5):
    """Run cross-validation for all models and compare."""
    results = {}
    
    # CRF (fast)
    try:
        print("\n>>> Running CRF Cross-Validation...")
        crf_results = run_crf_cv(config, dataset, n_splits=n_splits, with_grid_search=False)
        if crf_results:
            results['CRF'] = crf_results
    except Exception as e:
        print(f"CRF CV failed: {e}")
    
    # BiLSTM (medium speed - use quick mode)
    try:
        print("\n>>> Running BiLSTM Cross-Validation...")
        bilstm_results = run_bilstm_cv(config, dataset, n_splits=n_splits, quick_mode=True)
        results['BiLSTM'] = bilstm_results
    except Exception as e:
        print(f"BiLSTM CV failed: {e}")
        import traceback
        traceback.print_exc()
    
    # BERT (very slow - skip by default or use n_splits=2)
    try:
        print("\n>>> Running BERT Cross-Validation...")
        # Use fewer splits for BERT if n_splits is large, unless user really wants it
        bert_splits = min(n_splits, 3)
        bert_results = run_bert_cv(config, dataset, n_splits=bert_splits, quick_mode=True)
        results['BERT'] = bert_results
    except Exception as e:
        print(f"BERT CV failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare results
    if len(results) > 1:
        print("\n" + "="*80)
        print("CROSS-VALIDATION COMPARISON")
        print("="*80)
        
        comparison_path = os.path.join(config.output_dir, "cv_comparison.png")
        compare_cv_results(results, metric='f1', save_path=comparison_path)
        
        # Save comparison results
        comparison_json_path = os.path.join(config.output_dir, "cv_comparison.json")
        comparison_data = {}
        for model_name, cv_result in results.items():
            comparison_data[model_name] = {
                'mean_f1': float(cv_result['mean']['f1']),
                'std_f1': float(cv_result['std']['f1']),
                'mean_precision': float(cv_result['mean']['precision']),
                'mean_recall': float(cv_result['mean']['recall'])
            }
        
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nComparison results saved to {comparison_json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Validation for NER Models")
    parser.add_argument("--model", type=str, default="crf",
                       choices=["crf", "bilstm", "bert", "all"],
                       help="Which model to run CV for")
    parser.add_argument("--n_splits", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--grid_search", action="store_true",
                       help="Run grid search (CRF only)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer epochs for testing")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    config = ProjectConfig()
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading CoNLL-2003 dataset...")
    dataset = load_conll2003()
    
    # Run cross-validation
    if args.model == "all":
        run_all_cv(config, dataset, n_splits=args.n_splits)
    elif args.model == "crf":
        run_crf_cv(config, dataset, n_splits=args.n_splits, with_grid_search=args.grid_search)
    elif args.model == "bilstm":
        run_bilstm_cv(config, dataset, n_splits=args.n_splits, quick_mode=args.quick)
    elif args.model == "bert":
        run_bert_cv(config, dataset, n_splits=min(args.n_splits, 3), quick_mode=args.quick)
    
    print("\n" + "="*80)
    print("Cross-validation complete!")
    print(f"Results saved to {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
