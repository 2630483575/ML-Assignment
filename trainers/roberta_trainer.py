"""
Trainer for RoBERTa NER model.

This trainer uses the Hugging Face Trainer API for efficient training,
similar to the BERT trainer but optimized for RoBERTa.
"""

import os
import json
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from utils.metrics import compute_metrics_transformers
from utils.visualization import plot_training_history


class RoBERTaTrainer:
    """
    Trainer class for RoBERTa NER model using Hugging Face Trainer.
    """
    
    def __init__(self, config, model_wrapper, train_dataset=None, eval_dataset=None):
        """
        Initialize RoBERTa trainer.
        
        Args:
            config: Project configuration
            model_wrapper: RoBERTaModel instance
            train_dataset: Training dataset (tokenized)
            eval_dataset: Evaluation dataset (tokenized)
        """
        self.config = config
        self.model_wrapper = model_wrapper
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Data collator for dynamic padding
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=model_wrapper.tokenizer,
            padding=True
        )
        
        # Setup training arguments
        self.training_args = self._setup_training_args()
        
        # Create trainer
        self.trainer = self._create_trainer()
    
    def _setup_training_args(self):
        """Setup training arguments for Hugging Face Trainer."""
        output_dir = os.path.join(self.config.output_dir, "roberta_checkpoints")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Training hyperparameters
            num_train_epochs=self.config.bert.epochs,  # Reuse BERT config
            per_device_train_batch_size=self.config.bert.batch_size,
            per_device_eval_batch_size=self.config.bert.batch_size,
            learning_rate=self.config.bert.learning_rate,
            weight_decay=self.config.bert.weight_decay,
            warmup_steps=self.config.bert.warmup_steps,
            
            # Evaluation strategy
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            
            # Logging
            logging_dir=os.path.join(self.config.output_dir, "roberta_logs"),
            logging_steps=100,
            logging_first_step=True,
            
            # Other settings
            save_total_limit=2,
            seed=self.config.seed,
            report_to="none",  # Disable wandb/tensorboard by default
            
            # Performance optimizations
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            dataloader_num_workers=2,
        )
        
        return training_args
    
    def _create_trainer(self):
        """Create Hugging Face Trainer instance."""
        # Create a partial function to pass id2label to compute_metrics
        from functools import partial
        compute_metrics_fn = partial(compute_metrics_transformers, id2label=self.model_wrapper.id2label)
        
        trainer = Trainer(
            model=self.model_wrapper.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.model_wrapper.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        return trainer
    
    def train(self):
        """
        Train the RoBERTa model.
        
        Returns:
            Training result
        """
        print("\n" + "="*80)
        print("Training RoBERTa Model")
        print("="*80)
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Evaluation samples: {len(self.eval_dataset)}")
        print(f"Epochs: {self.config.bert.epochs}")
        print(f"Batch size: {self.config.bert.batch_size}")
        print(f"Learning rate: {self.config.bert.learning_rate}")
        print("="*80)
        
        # Train
        train_result = self.trainer.train()
        
        # Log final metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # Save best model
        best_model_path = os.path.join(self.config.output_dir, "best_roberta_model")
        self.model_wrapper.save(best_model_path)
        
        print(f"\nBest model saved to: {best_model_path}")
        
        # Extract and plot history
        history = {'train_loss': [], 'val_f1': []}
        
        # Iterate through log history to extract metrics
        for log in self.trainer.state.log_history:
            if 'loss' in log:
                history['train_loss'].append(log['loss'])
            if 'eval_f1' in log:
                history['val_f1'].append(log['eval_f1'])
        
        # Aggregate training loss by epoch
        epoch_losses = {}
        for log in self.trainer.state.log_history:
            if 'loss' in log and 'epoch' in log:
                epoch = int(log['epoch'])
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(log['loss'])
        
        avg_train_loss = [float(np.mean(epoch_losses[e])) for e in sorted(epoch_losses.keys())]
        
        # Re-structure history for the plotter
        clean_history = {
            'train_loss': avg_train_loss,
            'val_f1': history['val_f1']
        }
        
        plot_path = os.path.join(self.config.output_dir, "roberta_training_history.png")
        plot_training_history(clean_history, save_path=plot_path)
        
        # Save history to JSON
        history_path = os.path.join(self.config.output_dir, "roberta_history.json")
        with open(history_path, 'w') as f:
            json.dump(clean_history, f, indent=2)
            
        # Generate predictions for confusion matrix
        print("Generating predictions for confusion matrix...")
        predictions_output = self.trainer.predict(self.eval_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=2)
        labels = predictions_output.label_ids
        
        # Convert to list of lists of labels (strings)
        y_pred = []
        y_true = []
        
        id2label = self.model_wrapper.id2label
        
        for i in range(len(predictions)):
            pred_seq = []
            true_seq = []
            for j in range(len(predictions[i])):
                label_id = labels[i][j]
                if label_id != -100:  # Ignore special tokens
                    pred_seq.append(id2label[predictions[i][j]])
                    true_seq.append(id2label[label_id])
            y_pred.append(pred_seq)
            y_true.append(true_seq)
            
        predictions_data = {
            'y_true': y_true,
            'y_pred': y_pred
        }
        predictions_path = os.path.join(self.config.output_dir, "roberta_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        return train_result
    
    def evaluate(self, eval_dataset=None):
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Optional evaluation dataset (uses self.eval_dataset if None)
            
        Returns:
            Evaluation metrics
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if dataset is None:
            raise ValueError("No evaluation dataset provided")
        
        print("\n" + "="*80)
        print("Evaluating RoBERTa Model")
        print("="*80)
        
        metrics = self.trainer.evaluate(eval_dataset=dataset)
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def predict(self, test_dataset):
        """
        Make predictions on test dataset.
        
        Args:
            test_dataset: Test dataset (tokenized)
            
        Returns:
            Predictions and metrics
        """
        print("\n" + "="*80)
        print("Making Predictions with RoBERTa Model")
        print("="*80)
        
        predictions = self.trainer.predict(test_dataset)
        
        return predictions


# Import torch at module level
import torch
