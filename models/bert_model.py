import torch
from transformers import BertForTokenClassification, BertTokenizerFast
import os

class BERTModel:
    def __init__(self, config, num_labels, id2label, label2id):
        self.config = config
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading BERT model: {config.bert.model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(config.bert.model_name)
        self.model = BertForTokenClassification.from_pretrained(
            config.bert.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        self.model.to(self.device)
        
    def save(self, path):
        """Save model and tokenizer."""
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"BERT model saved to {path}")
        
    def load(self, path):
        """Load model and tokenizer."""
        self.model = BertForTokenClassification.from_pretrained(path)
        self.tokenizer = BertTokenizerFast.from_pretrained(path)
        self.model.to(self.device)
        print(f"BERT model loaded from {path}")
