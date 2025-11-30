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
