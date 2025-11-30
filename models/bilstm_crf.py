"""
BiLSTM-CRF Model implementation using PyTorch.
"""
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=100, hidden_dim=256, dropout=0.5):
        """
        Initialize BiLSTM-CRF model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            tag_to_ix (dict): Mapping from tag to index
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of hidden layer
            dropout (float): Dropout probability
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentence, mask):
        """
        Forward pass for prediction (decoding).
        
        Args:
            sentence (tensor): Input sentence indices [batch_size, seq_len]
            mask (tensor): Mask tensor [batch_size, seq_len]
            
        Returns:
            list: List of best path indices for each sequence in batch
        """
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)

        emissions = self.hidden2tag(lstm_out)

        predictions = self.crf.decode(emissions, mask=mask)
        return predictions

    def neg_log_likelihood(self, sentence, tags, mask):
        """
        Calculate negative log likelihood loss.
        
        Args:
            sentence (tensor): Input sentence indices [batch_size, seq_len]
            tags (tensor): Target tag indices [batch_size, seq_len]
            mask (tensor): Mask tensor [batch_size, seq_len]
            
        Returns:
            tensor: Loss value
        """
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)

        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss
