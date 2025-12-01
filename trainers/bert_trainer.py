import numpy as np
import torch
import json
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from utils.metrics import compute_bert_metrics
from utils.visualization import plot_training_history
import os

class BERTTrainer:
    def __init__(self, config, model_wrapper, train_dataset=None, eval_dataset=None):
        self.config = config
        self.model_wrapper = model_wrapper
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
    def train(self):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, 'bert_checkpoints'),
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.config.bert.learning_rate,
            per_device_train_batch_size=self.config.bert.batch_size,
            per_device_eval_batch_size=self.config.bert.batch_size * 2,
            num_train_epochs=self.config.bert.epochs,
            weight_decay=self.config.bert.weight_decay,
            logging_dir=os.path.join(self.config.output_dir, 'logs'),
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            warmup_steps=self.config.bert.warmup_steps,
            use_cpu=not torch.cuda.is_available(),
        )
        
        data_collator = DataCollatorForTokenClassification(self.model_wrapper.tokenizer)
        
        def compute_metrics_wrapper(eval_pred):
            return compute_bert_metrics(eval_pred, self.model_wrapper.id2label)

        trainer = Trainer(
            model=self.model_wrapper.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model_wrapper.tokenizer,
            compute_metrics=compute_metrics_wrapper
        )
        
        print("Starting BERT training...")
        train_result = trainer.train()
        print("BERT training complete.")
        
        # Save best model
        save_path = os.path.join(self.config.output_dir, "best_bert_model")
        self.model_wrapper.save(save_path)

        # Extract and plot history
        history = {'train_loss': [], 'val_f1': []}
        
        # Iterate through log history to extract metrics
        for log in trainer.state.log_history:
            if 'loss' in log:
                history['train_loss'].append(log['loss'])
            if 'eval_f1' in log:
                history['val_f1'].append(log['eval_f1'])
        
        # Handle case where lengths might mismatch due to logging steps vs eval steps
        # For simple plotting, we might just plot what we have or align them by epoch if available
        # Here we just pass the raw lists; the plotter might need adjustment if they are vastly different lengths
        # But typically plot_training_history expects epoch-aligned data. 
        # HF Trainer logs 'loss' every logging_steps, and 'eval_f1' every epoch.
        # So we might need to aggregate training loss per epoch to match the visualization utility.
        
        # Aggregate training loss by epoch
        epoch_losses = {}
        for log in trainer.state.log_history:
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
        
        plot_path = os.path.join(self.config.output_dir, "bert_training_history.png")
        plot_training_history(clean_history, save_path=plot_path)
        
        # Save history to JSON
        history_path = os.path.join(self.config.output_dir, "bert_history.json")
        with open(history_path, 'w') as f:
            json.dump(clean_history, f, indent=2)
            
        # Generate predictions for confusion matrix
        print("Generating predictions for confusion matrix...")
        predictions_output = trainer.predict(self.eval_dataset)
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
        predictions_path = os.path.join(self.config.output_dir, "bert_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
    def evaluate(self):
        # Evaluation logic if needed separately, but Trainer handles it
        pass
