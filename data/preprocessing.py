"""
Data preprocessing and feature extraction utilities for NER.
"""
import torch
from torch.nn.utils.rnn import pad_sequence


# ============================================================================
# CRF Feature Extraction
# ============================================================================

def word2features(sent, i):
    """
    Extract features for a single word in a sentence for CRF.
    
    Args:
        sent: Dictionary with 'tokens' and 'pos_tags' keys
        i: Index of the word
        
    Returns:
        dict: Feature dictionary
    """
    word = sent['tokens'][i]
    postag = sent['pos_tags'][i]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': str(postag),
    }
    
    # Features for previous word
    if i > 0:
        word1 = sent['tokens'][i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': str(sent['pos_tags'][i-1]),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    
    # Features for next word
    if i < len(sent['tokens']) - 1:
        word1 = sent['tokens'][i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': str(sent['pos_tags'][i+1]),
        })
    else:
        features['EOS'] = True  # End of sentence
    
    return features


def sent2features(sent):
    """
    Convert a sentence to a list of feature dictionaries for CRF.
    
    Args:
        sent: Dictionary with 'tokens' and 'pos_tags' keys
        
    Returns:
        list: List of feature dictionaries
    """
    return [word2features(sent, i) for i in range(len(sent['tokens']))]


def sent2labels(sent, label_list):
    """
    Extract labels from a sentence.
    
    Args:
        sent: Dictionary with 'ner_tags' key
        label_list: List of label names
        
    Returns:
        list: List of label strings
    """
    return [label_list[tag] for tag in sent['ner_tags']]


# ============================================================================
# BiLSTM Vocabulary and Data Preparation
# ============================================================================

def build_vocab(dataset_split):
    """
    Build vocabulary from training data.
    
    Args:
        dataset_split: Training dataset split
        
    Returns:
        dict: word2idx mapping
    """
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    for example in dataset_split:
        for word in example['tokens']:
            word_lower = word.lower()
            if word_lower not in word2idx:
                word2idx[word_lower] = len(word2idx)
    
    print(f"Vocabulary size: {len(word2idx):,}")
    return word2idx


def prepare_sequence(sent, word2idx, tag2idx, label_list):
    """
    Convert a sentence to indices for BiLSTM.
    
    Args:
        sent: Dictionary with 'tokens' and 'ner_tags'
        word2idx: Word to index mapping
        tag2idx: Tag to index mapping
        label_list: List of label names
        
    Returns:
        tuple: (word_indices, tag_indices)
    """
    word_indices = [word2idx.get(w.lower(), word2idx['<UNK>']) for w in sent['tokens']]
    tag_indices = [tag2idx[label_list[t]] for t in sent['ner_tags']]
    return word_indices, tag_indices


# ============================================================================
# PyTorch Dataset and DataLoader utilities
# ============================================================================

class NERDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for NER."""
    
    def __init__(self, data):
        """
        Args:
            data: List of (word_indices, tag_indices) tuples
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        words, tags = self.data[idx]
        return torch.tensor(words, dtype=torch.long), torch.tensor(tags, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function for BiLSTM DataLoader.
    """
    words_batch = [item[0] for item in batch]
    tags_batch = [item[1] for item in batch]

    words_padded = pad_sequence(words_batch, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags_batch, batch_first=True, padding_value=0)

    masks = (words_padded != 0).byte()

    return words_padded, tags_padded, masks


def prepare_datasets(dataset, label_list):
    """
    Prepare train, validation, and test datasets for BiLSTM.
    
    Args:
        dataset: HuggingFace DatasetDict
        label_list: List of label names
        
    Returns:
        tuple: (train_data, val_data, test_data, word2idx, tag2idx, idx2tag)
    """
    
    # Build vocabulary
    word2idx = build_vocab(dataset['train'])
    
    # Create tag mappings
    tag2idx = {tag: i for i, tag in enumerate(label_list)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    
    # Prepare data
    print("\nConverting data to indices...")
    train_data = [prepare_sequence(s, word2idx, tag2idx, label_list) for s in dataset['train']]
    val_data = [prepare_sequence(s, word2idx, tag2idx, label_list) for s in dataset['validation']]
    test_data = [prepare_sequence(s, word2idx, tag2idx, label_list) for s in dataset['test']]
    
    print(f"Prepared {len(train_data)} training sequences")
    print(f"Prepared {len(val_data)} validation sequences")
    print(f"Prepared {len(test_data)} test sequences")
    
    return train_data, val_data, test_data, word2idx, tag2idx, idx2tag


# ============================================================================
# BERT Data Preparation
# ============================================================================

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """
    Tokenize and align labels for BERT.
    
    Args:
        examples: Dataset examples
        tokenizer: BERT tokenizer
        label_all_tokens: Whether to label all tokens or just the first one
        
    Returns:
        tokenized_inputs: Tokenized inputs with labels
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
