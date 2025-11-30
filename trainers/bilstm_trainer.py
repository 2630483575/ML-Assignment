"""
Trainer for BiLSTM-CRF model.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.bilstm_crf import BiLSTM_CRF
from data.preprocessing import prepare_datasets, NERDataset, collate_fn
from utils.metrics import evaluate_model
from utils.visualization import plot_training_history

class BiLSTMTrainer:
    def __init__(self, config, label_list):
        """
        Initialize BiLSTM trainer.
        
        Args:
            config (ProjectConfig): Project configuration
            label_list (list): List of label names
        """
        self.config = config
        self.label_list = label_list
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.word2idx = None
        self.tag2idx = None
        self.idx2tag = None
        
    def setup_model(self, vocab_size, tag2idx):
        """Initialize model and optimizer."""
        self.model = BiLSTM_CRF(
            vocab_size=vocab_size,
            tag_to_ix=tag2idx,
            embedding_dim=self.config.bilstm.embedding_dim,
            hidden_dim=self.config.bilstm.hidden_dim,
            dropout=self.config.bilstm.dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.bilstm.learning_rate,
            weight_decay=self.config.bilstm.weight_decay
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for words_batch, tags_batch, masks_batch in dataloader:
            words_batch = words_batch.to(self.device)
            tags_batch = tags_batch.to(self.device)
            masks_batch = masks_batch.to(self.device)
            
            self.model.zero_grad()
            
            loss = self.model.neg_log_likelihood(words_batch, tags_batch, masks_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for words_batch, tags_batch, masks_batch in dataloader:
                words_batch = words_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)
                
                batch_preds = self.model(words_batch, masks_batch)
                
                for i, pred_path in enumerate(batch_preds):
                    valid_len = sum(masks_batch[i]).item()
                    true_tags = tags_batch[i][:valid_len].tolist()
                    
                    pred_label = [self.idx2tag[p] for p in pred_path]
                    true_label = [self.idx2tag[t] for t in true_tags]
                    
                    all_preds.append(pred_label)
                    all_trues.append(true_label)
                    
        return evaluate_model(all_trues, all_preds)

    def train(self, dataset):
        """
        Full training loop.
        
        Args:
            dataset: HuggingFace DatasetDict
        """
        # Prepare data
        train_data, val_data, test_data, self.word2idx, self.tag2idx, self.idx2tag = \
            prepare_datasets(dataset, self.label_list)
            
        train_loader = DataLoader(
            NERDataset(train_data),
            batch_size=self.config.bilstm.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            NERDataset(val_data),
            batch_size=self.config.bilstm.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Setup model
        self.setup_model(len(self.word2idx), self.tag2idx)
        
        # Training loop
        best_f1 = 0.0
        history = {'train_loss': [], 'val_f1': []}
        patience_counter = 0
        
        print(f"\nStarting training for {self.config.bilstm.epochs} epochs...")
        
        for epoch in range(self.config.bilstm.epochs):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_f1 = metrics['f1']
            
            history['train_loss'].append(train_loss)
            history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch+1}/{self.config.bilstm.epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # Save best model
                save_path = os.path.join(self.config.output_dir, "best_bilstm_crf.pt")
                os.makedirs(self.config.output_dir, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved! (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.bilstm.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        print(f"\nTraining completed. Best Validation F1: {best_f1:.4f}")
        
        # Plot history
        plot_path = os.path.join(self.config.output_dir, "bilstm_training_history.png")
        plot_training_history(history, save_path=plot_path)
        
        # Final evaluation on test set
        print("\nEvaluating on test set with best model...")
        self.model.load_state_dict(torch.load(os.path.join(self.config.output_dir, "best_bilstm_crf.pt")))
        
        test_loader = DataLoader(
            NERDataset(test_data),
            batch_size=self.config.bilstm.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_metrics = self.evaluate(test_loader)
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        print("\nDetailed Report:")
        print(test_metrics['report'])
