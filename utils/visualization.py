"""
Visualization utilities for NER project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def plot_entity_distribution(stats_dict, save_path=None):
    """
    Plot entity distribution for train/val/test sets matching notebook style.
    
    Args:
        stats_dict (dict): Dictionary mapping split names to stats objects
        save_path (str, optional): Path to save the plot
    """
    n_splits = len(stats_dict)
    fig, axes = plt.subplots(1, n_splits, figsize=(6*n_splits, 6))
    
    if n_splits == 1:
        axes = [axes]
        
    # Colors from notebook
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
    for idx, (name, stats) in enumerate(stats_dict.items()):
        entity_types = list(stats['entity_counts'].keys())
        entity_counts = list(stats['entity_counts'].values())
        
        if not entity_counts:
            continue
            
        max_count = max(entity_counts)

        axes[idx].bar(entity_types, entity_counts, color=colors[:len(entity_types)])
        axes[idx].set_title(f'{name} - Entity distribution', fontsize=14, fontweight='bold', pad=15)
        axes[idx].set_xlabel('Entity Type')
        axes[idx].set_ylabel('quantity')
        axes[idx].set_ylim(0, max_count * 1.15)

        for i, v in enumerate(entity_counts):
            axes[idx].text(i, v + (max_count*0.02), str(v), ha='center', fontsize=10)
            
        # Print most/least common like in notebook
        if name == 'Train':
            print(f"- Most common entity: {stats['entity_counts'].most_common(1)[0]}")
            if stats['entity_counts']:
                print(f"- Least common entity: {stats['entity_counts'].most_common()[-1]}")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Entity distribution plot saved to {save_path}")

def plot_sentence_length_distribution(lengths, save_path=None):
    """
    Plot sentence length distribution histogram.
    
    Args:
        lengths (list): List of sentence lengths
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    plt.hist(lengths, bins=50, alpha=0.7, color='#FF6B6B', edgecolor='black')
    plt.xlabel('Sentence length(tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Sentence length distribution', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sentence length plot saved to {save_path}")
        
    print(f"\nAverage sentence length: {np.mean(lengths):.2f} tokens")
    print(f"The longest sentence: {np.max(lengths)} tokens")

def print_data_summary_table(train_stats, val_stats, train_size, val_size):
    """
    Print data summary table using pandas.
    
    Args:
        train_stats (dict): Training set statistics
        val_stats (dict): Validation set statistics
        train_size (int): Number of training sentences
        val_size (int): Number of validation sentences
    """
    summary = pd.DataFrame({
        'Indicator': ['Number of sentences', 'Vocabulary', 'Average sentence length', 'PER', 'ORG', 'LOC', 'MISC'],
        'Training set': [
            train_size,
            len(train_stats['vocabulary']),
            f"{np.mean(train_stats['sentence_lengths']):.1f}",
            train_stats['entity_counts']['PER'],
            train_stats['entity_counts']['ORG'],
            train_stats['entity_counts']['LOC'],
            train_stats['entity_counts']['MISC']
        ],
        'Validation set': [
            val_size,
            len(val_stats['vocabulary']),
            f"{np.mean(val_stats['sentence_lengths']):.1f}",
            val_stats['entity_counts']['PER'],
            val_stats['entity_counts']['ORG'],
            val_stats['entity_counts']['LOC'],
            val_stats['entity_counts']['MISC']
        ]
    })
    
    print("\nDataset Summary:")
    print(summary.to_string(index=False))

def plot_training_history(history, save_path=None):
    """
    Plot training loss and validation metrics.
    
    Args:
        history (dict): Dictionary with 'train_loss' and 'val_f1' lists
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#FF6B6B', marker='o', markersize=3)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Validation F1', color='#4ECDC4', marker='o', markersize=3)
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

def plot_cv_results(cv_scores, save_path=None):
    """
    Plot cross-validation results.
    
    Args:
        cv_scores (list): List of F1 scores from CV
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))

    plt.plot(range(1, len(cv_scores) + 1), cv_scores,
             marker='o',
             linewidth=3,
             markersize=12,
             color='#FF6B35',
             markerfacecolor='#FFA500',
             markeredgecolor='#FF4500',
             markeredgewidth=2)

    mean_score = np.mean(cv_scores)
    plt.axhline(y=mean_score,
                color='#00CED1',
                linestyle='--',
                linewidth=2.5,
                label=f'mean: {mean_score:.4f}')

    for i, score in enumerate(cv_scores, 1):
        plt.text(i, score + 0.0002, f'{score:.4f}',
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold',
                 color='#333333')

    plt.xlabel('Fold numbers', fontsize=13, fontweight='bold')
    plt.ylabel('F1 score', fontsize=13, fontweight='bold')
    plt.title('Cross-validation results', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(alpha=0.3, linestyle='--')
    
    # Adjust ylim with some padding
    y_range = max(cv_scores) - min(cv_scores)
    padding = y_range * 0.1 if y_range > 0 else 0.01
    plt.ylim(min(cv_scores) - padding, max(cv_scores) + padding)
    
    plt.xticks(range(1, len(cv_scores) + 1), fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CV results plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        y_true (list): List of true labels (flattened)
        y_pred (list): List of predicted labels (flattened)
        labels (list, optional): List of label names to include
        save_path (str, optional): Path to save the plot
        title (str): Plot title
    """
    from sklearn.metrics import confusion_matrix
    
    # Flatten lists if they are lists of lists
    if isinstance(y_true[0], list):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    
    # Get unique labels if not provided
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
        # Move 'O' to the end if present for better visualization of entities
        if 'O' in labels:
            labels.remove('O')
            labels.append('O')
            
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title, fontsize=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()
