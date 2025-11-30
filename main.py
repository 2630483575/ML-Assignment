"""
Main entry point for NER project.
"""
import argparse
import os
import torch
import numpy as np
import random
from config.config import ProjectConfig
from data.dataset import load_conll2003, get_label_mappings
from trainers.crf_trainer import CRFTrainer
from trainers.bilstm_trainer import BiLSTMTrainer
from trainers.bert_trainer import BERTTrainer
from models.bert_model import BERTModel

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="NER Project for CoNLL-2003")
    
    # Mode selection
    parser.add_argument("--model", type=str, default="crf", choices=["crf", "bilstm", "bert"], help="Model type")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "cv", "visualize"], help="Operation mode")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # CRF arguments
    parser.add_argument("--c1", type=float, default=0.1, help="CRF L1 regularization")
    parser.add_argument("--c2", type=float, default=0.1, help="CRF L2 regularization")
    parser.add_argument("--max_iter", type=int, default=100, help="CRF max iterations")
    
    # BiLSTM arguments
    parser.add_argument("--epochs", type=int, default=20, help="BiLSTM/BERT epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="BiLSTM/BERT batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="BiLSTM/BERT learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="BiLSTM hidden dimension")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ProjectConfig()
    config.model_type = args.model
    config.mode = args.mode
    config.output_dir = args.output_dir
    config.seed = args.seed
    
    # Update sub-configs based on args
    config.crf.c1 = args.c1
    config.crf.c2 = args.c2
    config.crf.max_iterations = args.max_iter
    
    config.bilstm.epochs = args.epochs
    config.bilstm.batch_size = args.batch_size
    config.bilstm.learning_rate = args.lr
    config.bilstm.hidden_dim = args.hidden_dim

    # Update BERT config if model is BERT
    if config.model_type == 'bert':
        config.bert.epochs = args.epochs
        config.bert.batch_size = args.batch_size
        # Use default BERT LR if user didn't change the default arg (which is 0.001, too high for BERT)
        if args.lr == 0.001:
             config.bert.learning_rate = 2e-5
        else:
             config.bert.learning_rate = args.lr
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Starting NER Project with {config.model_type.upper()} model in {config.mode.upper()} mode")

    # Load dataset
    dataset = load_conll2003()
    label_list, label2id, id2label = get_label_mappings(dataset)

    # Visualization mode
    if config.mode == "visualize":
        from utils.visualization import plot_entity_distribution, plot_sentence_length_distribution, print_data_summary_table
        from data.dataset import extract_statistics
        
        print("Generating dataset visualizations...")
        
        # Extract stats
        train_stats = extract_statistics(dataset['train'])
        val_stats = extract_statistics(dataset['validation'])
        test_stats = extract_statistics(dataset['test'])
        
        stats = {
            'Train': train_stats,
            'Validation': val_stats,
            'Test': test_stats
        }
        
        # 1. Entity Distribution Plot
        save_path = os.path.join(config.output_dir, "entity_distribution.png")
        plot_entity_distribution(stats, save_path=save_path)
        
        # 2. Sentence Length Distribution Plot
        save_path_len = os.path.join(config.output_dir, "sentence_length.png")
        plot_sentence_length_distribution(train_stats['sentence_lengths'], save_path=save_path_len)
        
        # 3. Data Summary Table
        print_data_summary_table(train_stats, val_stats, len(dataset['train']), len(dataset['validation']))
        
        print(f"\nDone! Plots saved to {config.output_dir}")
        return

    # Training/Evaluation Logic
    if config.model_type == "crf":
        trainer = CRFTrainer(config, label_list)
        
        if config.mode == "train":
            trainer.train(dataset)
        elif config.mode == "cv":
            trainer.cross_validate(dataset)
        elif config.mode == "evaluate":
            print("Evaluation mode for CRF not fully implemented separately, running training + eval")
            trainer.train(dataset)

    elif config.model_type == "bilstm":
        trainer = BiLSTMTrainer(config, label_list)
        
        if config.mode == "train":
            trainer.train(dataset)
        elif config.mode == "evaluate":
            print("Evaluation mode for BiLSTM not fully implemented separately")
            # trainer.evaluate(dataset) # Need to implement load model first

    elif config.model_type == "bert":
        # Initialize BERT model wrapper
        model_wrapper = BERTModel(config, len(label_list), id2label, label2id)
        
        # Prepare data for BERT
        from data.preprocessing import tokenize_and_align_labels
        
        def tokenize_function(examples):
            return tokenize_and_align_labels(examples, model_wrapper.tokenizer)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        trainer = BERTTrainer(
            config, 
            model_wrapper, 
            train_dataset=tokenized_datasets['train'], 
            eval_dataset=tokenized_datasets['validation']
        )
        
        if config.mode == "train":
            trainer.train()
        elif config.mode == "evaluate":
            trainer.evaluate()

if __name__ == "__main__":
    main()
