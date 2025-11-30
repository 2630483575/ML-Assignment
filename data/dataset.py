"""
Dataset loading and management utilities for NER project.
"""
from datasets import load_dataset
from collections import Counter
import numpy as np


def load_conll2003():
    """
    Load the CoNLL-2003 dataset.
    
    Returns:
        DatasetDict: Dictionary with 'train', 'validation', and 'test' splits
    """
    print("Loading CoNLL-2003 dataset...")
    try:
        dataset = load_dataset("conll2003", revision="refs/convert/parquet")
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        print("Attempting to load from local cache or alternative source if available...")
        # Fallback or re-raise
        raise e
    
    print(f"\nDataset Structure:")
    print(dataset)
    print(f"\nTraining set: {len(dataset['train'])} examples")
    print(f"Validation set: {len(dataset['validation'])} examples")
    print(f"Test set: {len(dataset['test'])} examples")
    
    return dataset


def get_label_mappings(dataset):
    """
    Extract label mappings from dataset.
    
    Args:
        dataset: HuggingFace dataset
        
    Returns:
        tuple: (label_list, label2id, id2label)
    """
    label_list = dataset['train'].features['ner_tags'].feature.names
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    print("\nNER Label Categories:")
    for i, label in enumerate(label_list):
        print(f"{i}: {label}")
    
    return label_list, label2id, id2label


def extract_statistics(dataset_split):
    """
    Extract statistics from a dataset split.
    
    Args:
        dataset_split: A split of the dataset (train/val/test)
        
    Returns:
        dict: Statistics including sentence lengths, entity counts, etc.
    """
    # Get label list from features if available, otherwise assume standard CoNLL labels
    if hasattr(dataset_split, 'features'):
        label_list = dataset_split.features['ner_tags'].feature.names
    else:
        # Fallback if features not available directly on split
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    stats = {
        'sentence_lengths': [],
        'entity_counts': Counter(),
        'label_counts': Counter(),
        'vocabulary': set()
    }
    
    for example in dataset_split:
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        # Sentence length
        stats['sentence_lengths'].append(len(tokens))
        
        # Vocabulary
        stats['vocabulary'].update([t.lower() for t in tokens])
        
        # Label counts
        for tag_id in ner_tags:
            if tag_id < len(label_list):
                stats['label_counts'][label_list[tag_id]] += 1
        
        # Entity counts (count B- tags)
        for tag_id in ner_tags:
            if tag_id < len(label_list):
                tag = label_list[tag_id]
                if tag.startswith('B-'):
                    entity_type = tag[2:]  # Remove 'B-' prefix
                    stats['entity_counts'][entity_type] += 1
    
    return stats


def print_dataset_summary(dataset):
    """
    Print a comprehensive summary of the dataset.
    
    Args:
        dataset: HuggingFace DatasetDict
    """
    train_stats = extract_statistics(dataset['train'])
    val_stats = extract_statistics(dataset['validation'])
    
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"{'Metric':<25} {'Train':<15} {'Validation':<15}")
    print("-"*50)
    print(f"{'Sentences':<25} {len(dataset['train']):<15} {len(dataset['validation']):<15}")
    print(f"{'Vocabulary Size':<25} {len(train_stats['vocabulary']):<15} {len(val_stats['vocabulary']):<15}")
    print(f"{'Avg Sentence Length':<25} {np.mean(train_stats['sentence_lengths']):<15.1f} {np.mean(val_stats['sentence_lengths']):<15.1f}")
    print(f"{'Max Sentence Length':<25} {max(train_stats['sentence_lengths']):<15} {max(val_stats['sentence_lengths']):<15}")
    
    print("\nEntity Distribution (Train):")
    for entity_type, count in train_stats['entity_counts'].most_common():
        print(f"  {entity_type:<10} {count:>6}")
    print("="*50)
