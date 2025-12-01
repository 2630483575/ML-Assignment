"""
Enhanced Visualization Suite for NER Project.

This module provides comprehensive visualization capabilities including:
- Model performance comparisons
- Confusion matrices
- Training history analysis
- Error analysis visualizations
- Attention weight visualization (BERT/RoBERTa)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Dict, Tuple, Optional
import os


# Set professional style
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_model_comparison(results_dict: Dict[str, Dict[str, float]], 
                          metrics: List[str] = ['f1', 'precision', 'recall'],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 5)):
    """
    Create comprehensive model comparison visualization.
    
    Args:
        results_dict: Dict mapping model names to metric dictionaries
        metrics: List of metric names to compare
        save_path: Path to save the figure
        figsize: Figure size
    """
    model_names = list(results_dict.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(model_names))
    width = 0.25
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, metric in enumerate(metrics):
        scores = [results_dict[name].get(metric, 0) for name in model_names]
        offset = (i - len(metrics)/2 + 0.5) * width
        ax1.bar(x + offset, scores, width, label=metric.capitalize(), 
               color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Plot 2: Radar chart (for F1 scores)
    ax2 = axes[1]
    f1_scores = [results_dict[name].get('f1', 0) for name in model_names]
    
    # Create horizontal bar chart sorted by performance
    sorted_indices = np.argsort(f1_scores)
    sorted_models = [model_names[i] for i in sorted_indices]
    sorted_scores = [f1_scores[i] for i in sorted_indices]
    
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_models)))
    bars = ax2.barh(sorted_models, sorted_scores, color=colors_bar, alpha=0.8)
    
    # Highlight best model
    bars[-1].set_color('gold')
    bars[-1].set_edgecolor('darkgoldenrod')
    bars[-1].set_linewidth(2)
    
    ax2.set_xlabel('F1 Score', fontweight='bold', fontsize=12)
    ax2.set_title('F1 Score Ranking', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax2.text(score + 0.02, i, f'{score:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrices(y_true_dict: Dict[str, List[List[str]]], 
                            y_pred_dict: Dict[str, List[List[str]]],
                            label_list: List[str],
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 4)):
    """
    Plot confusion matrices for multiple models side by side.
    
    Args:
        y_true_dict: Dict mapping model names to true labels
        y_pred_dict: Dict mapping model names to predicted labels
        label_list: List of label names
        save_path: Path to save the figure
        figsize: Figure size
    """
    n_models = len(y_true_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, y_true) in zip(axes, y_true_dict.items()):
        y_pred = y_pred_dict[model_name]
        
        # Flatten lists
        y_true_flat = [label for sent in y_true for label in sent]
        y_pred_flat = [label for sent in y_pred for label in sent]
        
        # Filter out 'O' tags for better visualization (optional)
        # Keep only entity labels
        entity_labels = [l for l in label_list if l != 'O']
        
        # Map labels to indices
        label_to_idx = {label: i for i, label in enumerate(entity_labels)}
        
        # Filter and convert to indices
        y_true_idx = []
        y_pred_idx = []
        for yt, yp in zip(y_true_flat, y_pred_flat):
            if yt in entity_labels and yp in entity_labels:
                y_true_idx.append(label_to_idx[yt])
                y_pred_idx.append(label_to_idx[yp])
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(entity_labels)))
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        # Set ticks
        tick_marks = np.arange(len(entity_labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(entity_labels, rotation=45, ha='right')
        ax.set_yticklabels(entity_labels)
        
        # Add text annotations
        thresh = cm_norm.max() / 2.
        for i in range(len(entity_labels)):
            for j in range(len(entity_labels)):
                ax.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center",
                       color="white" if cm_norm[i, j] > thresh else "black",
                       fontsize=8)
        
        ax.set_ylabel('True label', fontsize=10)
        ax.set_xlabel('Predicted label', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()


def plot_training_curves(history_dict: Dict[str, Dict[str, List[float]]],
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 10)):
    """
    Plot training and validation curves for multiple models.
    
    Args:
        history_dict: Dict mapping model names to history dictionaries
                     Each history dict should have 'train_loss', 'val_loss', 'val_f1', etc.
        save_path: Path to save the figure
        figsize: Figure size
    """
    n_models = len(history_dict)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot 1: Training Loss
    ax = axes[0]
    for (model_name, history), color in zip(history_dict.items(), colors):
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], marker='o', label=model_name, 
                   color=color, linewidth=2)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[1]
    for (model_name, history), color in zip(history_dict.items(), colors):
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], marker='s', label=model_name, 
                   color=color, linewidth=2)
    ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation F1
    ax = axes[2]
    for (model_name, history), color in zip(history_dict.items(), colors):
        if 'val_f1' in history:
            epochs = range(1, len(history['val_f1']) + 1)
            ax.plot(epochs, history['val_f1'], marker='^', label=model_name, 
                   color=color, linewidth=2)
    ax.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate Schedule (if available)
    ax = axes[3]
    for (model_name, history), color in zip(history_dict.items(), colors):
        if 'learning_rate' in history:
            epochs = range(1, len(history['learning_rate']) + 1)
            ax.plot(epochs, history['learning_rate'], marker='D', label=model_name, 
                   color=color, linewidth=2)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_error_analysis(errors: List[Dict],
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (14, 8)):
    """
    Visualize error analysis patterns.
    
    Args:
        errors: List of error dictionaries with keys:
               'true_label', 'pred_label', 'word', 'sentence'
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Count error types
    error_types = {}
    for error in errors:
        key = (error['true_label'], error['pred_label'])
        error_types[key] = error_types.get(key, 0) + 1
    
    # Sort by frequency
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Error type distribution
    ax = axes[0]
    labels = [f"{true}→{pred}" for (true, pred), count in sorted_errors]
    counts = [count for _, count in sorted_errors]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(labels)))
    bars = ax.barh(labels, counts, color=colors)
    ax.set_xlabel('Frequency', fontweight='bold')
    ax.set_title('Top Error Types', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
               str(count), va='center')
    
    # Plot 2: Error distribution by true label
    ax = axes[1]
    true_label_errors = {}
    for error in errors:
        label = error['true_label']
        true_label_errors[label] = true_label_errors.get(label, 0) + 1
    
    labels = list(true_label_errors.keys())
    values = list(true_label_errors.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Errors by True Label', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Error analysis plot saved to {save_path}")
    
    plt.show()


def plot_per_label_performance(metrics_per_label: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)):
    """
    Plot performance metrics for each label.
    
    Args:
        metrics_per_label: Dict mapping labels to metric dictionaries
        save_path: Path to save the figure
        figsize: Figure size
    """
    labels = list(metrics_per_label.keys())
    precision = [metrics_per_label[l].get('precision', 0) for l in labels]
    recall = [metrics_per_label[l].get('recall', 0) for l in labels]
    f1 = [metrics_per_label[l].get('f1', 0) for l in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Entity Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Performance by Entity Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Per-label performance plot saved to {save_path}")
    
    plt.show()


def create_summary_report(results_dict: Dict[str, Dict],
                         cv_results: Optional[Dict] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (16, 10)):
    """
    Create a comprehensive summary report with multiple subplots.
    
    Args:
        results_dict: Dict mapping model names to results
        cv_results: Optional cross-validation results
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('NER Model Performance Summary Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: F1 Score Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    models = list(results_dict.keys())
    f1_scores = [results_dict[m]['f1'] for m in models]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax1.bar(models, f1_scores, color=colors, alpha=0.8)
    best_idx = np.argmax(f1_scores)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)
    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('F1 Score Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Precision-Recall
    ax2 = fig.add_subplot(gs[0, 2])
    precisions = [results_dict[m]['precision'] for m in models]
    recalls = [results_dict[m]['recall'] for m in models]
    for i, model in enumerate(models):
        ax2.scatter(recalls[i], precisions[i], s=150, label=model, 
                   color=colors[i], alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Metrics Radar (for best model)
    ax3 = fig.add_subplot(gs[1, :], projection='polar')
    best_model = models[best_idx]
    best_results = results_dict[best_model]
    
    categories = ['F1', 'Precision', 'Recall']
    values = [best_results['f1'], best_results['precision'], best_results['recall']]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax3.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax3.fill(angles, values, alpha=0.25, color='#3498db')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title(f'Best Model: {best_model}', fontweight='bold', pad=20)
    ax3.grid(True)
    
    # Plot 4: Model Comparison Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for model in models:
        row = [
            model,
            f"{results_dict[model]['f1']:.4f}",
            f"{results_dict[model]['precision']:.4f}",
            f"{results_dict[model]['recall']:.4f}"
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'F1', 'Precision', 'Recall'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best model row
    for i in range(4):    
        table[(best_idx + 1, i)].set_facecolor('#ffffcc')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Summary report saved to {save_path}")
    
    plt.show()
