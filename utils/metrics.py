"""
Evaluation metrics utilities for NER.
"""
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from sklearn.model_selection import KFold

def evaluate_model(y_true, y_pred):
    """
    Compute evaluation metrics using seqeval.
    
    Args:
        y_true (list): List of true label sequences
        y_pred (list): List of predicted label sequences
        
    Returns:
        dict: Dictionary containing f1, precision, recall, and report
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def format_results(metrics, prefix=""):
    """
    Format metrics for display.
    
    Args:
        metrics (dict): Metrics dictionary
        prefix (str): Prefix for metric names
        
    Returns:
        str: Formatted string
    """
    p = f"{prefix} " if prefix else ""
    output = f"\n{p}Evaluation Results:\n"
    output += f"{'-'*30}\n"
    output += f"F1 Score:  {metrics['f1']:.4f}\n"
    output += f"Precision: {metrics['precision']:.4f}\n"
    output += f"Recall:    {metrics['recall']:.4f}\n"
    output += f"\nDetailed Report:\n{metrics['report']}\n"
    return output


def compute_bert_metrics(eval_pred, id2label):
    """
    Compute metrics for BERT model.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        id2label: Mapping from index to label
        
    Returns:
        dict: Metrics dictionary
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = evaluate_model(true_labels, true_predictions)
    # Remove report for trainer metrics to avoid clutter, or keep it if needed
    # For HF Trainer, we usually want scalar values
    return {k: v for k, v in results.items() if k != 'report'}
