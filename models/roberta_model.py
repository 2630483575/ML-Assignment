"""
RoBERTa Model for NER.

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a retraining of BERT
with improved training methodology. It often achieves better performance than BERT
on downstream tasks including NER.
"""

from transformers import RobertaForTokenClassification, RobertaTokenizerFast
import torch
import os


class RoBERTaModel:
    """
    Wrapper for RoBERTa-based NER model using Hugging Face transformers.
    """
    
    def __init__(self, config, num_labels, id2label, label2id, 
                 model_name="roberta-base"):
        """
        Initialize RoBERTa model for token classification.
        
        Args:
            config: Project configuration
            num_labels: Number of NER labels
            id2label: Mapping from label ID to label name
            label2id: Mapping from label name to label ID
            model_name: Pre-trained RoBERTa model name
        """
        self.config = config
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.model_name = model_name
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name,
            add_prefix_space=True  # Important for RoBERTa tokenization
        )
        
        # Initialize model
        self.model = RobertaForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"RoBERTa model initialized: {model_name}")
        print(f"Device: {self.device}")
        print(f"Number of labels: {num_labels}")
    
    def save(self, save_directory):
        """
        Save model and tokenizer.
        
        Args:
            save_directory: Directory to save model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        print(f"RoBERTa model saved to {save_directory}")
    
    def load(self, load_directory):
        """
        Load model and tokenizer from directory.
        
        Args:
            load_directory: Directory containing saved model
        """
        self.model = RobertaForTokenClassification.from_pretrained(load_directory)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(load_directory)
        
        self.model.to(self.device)
        
        print(f"RoBERTa model loaded from {load_directory}")
    
    def get_model(self):
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
