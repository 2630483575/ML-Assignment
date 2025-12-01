"""
Unified Cross-Validation Framework for NER Models.

This module provides:
- K-fold cross-validation for CRF, BiLSTM, and BERT models
- Hyperparameter tuning with Grid Search and Random Search
- Result aggregation and statistical analysis
- Visualization of cross-validation results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os


class CrossValidator:
    """
    Unified cross-validation framework for all NER models.
    
    Supports:
    - K-fold cross-validation
    - Stratified splits (if needed)
    - Hyperparameter grid search
    - Random search
    - Result aggregation and visualization
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # Store results
        self.cv_results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def cross_validate(self, 
                      train_fn: Callable,
                      eval_fn: Callable,
                      dataset: Any,
                      params: Dict[str, Any],
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Perform K-fold cross-validation.
        
        Args:
            train_fn: Function to train model, signature: train_fn(train_data, params) -> model
            eval_fn: Function to evaluate model, signature: eval_fn(model, test_data) -> metrics_dict
            dataset: Dataset to split (should support indexing/slicing)
            params: Hyperparameters for the model
            verbose: Whether to print progress
            
        Returns:
            Dictionary with aggregated results (mean, std for each metric)
        """
        fold_results = []
        
        # Convert dataset to list for easier indexing
        data_list = list(dataset)
        indices = np.arange(len(data_list))
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.kfold.split(indices)):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Fold {fold_idx + 1}/{self.n_splits}")
                print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
                print(f"{'='*60}")
            
            # Create train and validation splits
            train_data = [data_list[i] for i in train_indices]
            val_data = [data_list[i] for i in val_indices]
            
            # Train model
            if verbose:
                print(f"Training on fold {fold_idx + 1}...")
            model = train_fn(train_data, params)
            
            # Evaluate model
            if verbose:
                print(f"Evaluating on fold {fold_idx + 1}...")
            metrics = eval_fn(model, val_data)
            
            # Store results
            metrics['fold'] = fold_idx + 1
            fold_results.append(metrics)
            
            if verbose:
                print(f"Fold {fold_idx + 1} Results:")
                for metric_name, value in metrics.items():
                    if metric_name == 'fold' or metric_name == 'report':
                        continue
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
        
        # Aggregate results
        aggregated = self._aggregate_results(fold_results)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Cross-Validation Results (Mean ± Std)")
            print(f"{'='*60}")
            for metric_name in aggregated['mean'].keys():
                mean_val = aggregated['mean'][metric_name]
                std_val = aggregated['std'][metric_name]
                print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        return {
            'mean': aggregated['mean'],
            'std': aggregated['std'],
            'fold_results': fold_results,
            'params': params
        }
    
    def grid_search(self,
                   train_fn: Callable,
                   eval_fn: Callable,
                   dataset: Any,
                   param_grid: Dict[str, List[Any]],
                   scoring_metric: str = 'f1',
                   verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform grid search with cross-validation.
        
        Args:
            train_fn: Training function
            eval_fn: Evaluation function
            dataset: Dataset
            param_grid: Dictionary of parameter lists to try
            scoring_metric: Metric to use for selecting best params
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_params, all_results)
        """
        all_results = []
        best_score = -np.inf
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Grid Search: Testing {len(param_combinations)} parameter combinations")
            print(f"{'='*70}")
        
        for idx, params in enumerate(param_combinations):
            if verbose:
                print(f"\n[{idx + 1}/{len(param_combinations)}] Testing params: {params}")
            
            # Run cross-validation for this parameter set
            cv_results = self.cross_validate(
                train_fn=train_fn,
                eval_fn=eval_fn,
                dataset=dataset,
                params=params,
                verbose=False  # Don't print fold details in grid search
            )
            
            # Get mean score for the scoring metric
            mean_score = cv_results['mean'][scoring_metric]
            std_score = cv_results['std'][scoring_metric]
            
            if verbose:
                print(f"  {scoring_metric}: {mean_score:.4f} ± {std_score:.4f}")
            
            # Store results
            result_entry = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_results': cv_results
            }
            all_results.append(result_entry)
            
            # Update best if this is better
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                if verbose:
                    print(f"  ⭐ New best {scoring_metric}: {mean_score:.4f}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Grid Search Complete!")
            print(f"Best {scoring_metric}: {best_score:.4f}")
            print(f"Best params: {best_params}")
            print(f"{'='*70}")
        
        self.best_params = best_params
        self.best_score = best_score
        self.cv_results = all_results
        
        return best_params, all_results
    
    def random_search(self,
                     train_fn: Callable,
                     eval_fn: Callable,
                     dataset: Any,
                     param_distributions: Dict[str, List[Any]],
                     n_iter: int = 10,
                     scoring_metric: str = 'f1',
                     verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform random search with cross-validation.
        
        Args:
            train_fn: Training function
            eval_fn: Evaluation function
            dataset: Dataset
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of random combinations to try
            scoring_metric: Metric to use for selecting best params
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_params, all_results)
        """
        all_results = []
        best_score = -np.inf
        best_params = None
        
        np.random.seed(self.random_state)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Random Search: Testing {n_iter} random parameter combinations")
            print(f"{'='*70}")
        
        for idx in range(n_iter):
            # Randomly sample parameters
            params = {key: np.random.choice(values) 
                     for key, values in param_distributions.items()}
            
            if verbose:
                print(f"\n[{idx + 1}/{n_iter}] Testing params: {params}")
            
            # Run cross-validation
            cv_results = self.cross_validate(
                train_fn=train_fn,
                eval_fn=eval_fn,
                dataset=dataset,
                params=params,
                verbose=False
            )
            
            mean_score = cv_results['mean'][scoring_metric]
            std_score = cv_results['std'][scoring_metric]
            
            if verbose:
                print(f"  {scoring_metric}: {mean_score:.4f} ± {std_score:.4f}")
            
            result_entry = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_results': cv_results
            }
            all_results.append(result_entry)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                if verbose:
                    print(f"  ⭐ New best {scoring_metric}: {mean_score:.4f}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Random Search Complete!")
            print(f"Best {scoring_metric}: {best_score:.4f}")
            print(f"Best params: {best_params}")
            print(f"{'='*70}")
        
        self.best_params = best_params
        self.best_score = best_score
        self.cv_results = all_results
        
        return best_params, all_results
    
    def _aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate results across folds."""
        # Collect all metric names (excluding 'fold')
        metric_names = [key for key in fold_results[0].keys() if key != 'fold']
        
        aggregated = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {}
        }
        
        for metric_name in metric_names:
            values = [result[metric_name] for result in fold_results]
            
            # Skip non-numeric metrics (like classification report string)
            if not values or not isinstance(values[0], (int, float, np.number)):
                continue
                
            aggregated['mean'][metric_name] = np.mean(values)
            aggregated['std'][metric_name] = np.std(values)
            aggregated['min'][metric_name] = np.min(values)
            aggregated['max'][metric_name] = np.max(values)
        
        return aggregated
    
    def plot_cv_results(self, 
                       metric_name: str = 'f1',
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 6)):
        """
        Plot cross-validation results.
        
        Args:
            metric_name: Metric to visualize
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.cv_results:
            print("No cross-validation results to plot. Run cross_validate first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Score by fold for each parameter set
        ax1 = axes[0]
        for idx, result_entry in enumerate(self.cv_results):
            fold_scores = [fold[metric_name] 
                          for fold in result_entry['cv_results']['fold_results']]
            folds = list(range(1, len(fold_scores) + 1))
            
            label = f"Params {idx + 1}"
            ax1.plot(folds, fold_scores, marker='o', label=label, alpha=0.7)
        
        ax1.set_xlabel('Fold')
        ax1.set_ylabel(metric_name.upper())
        ax1.set_title(f'{metric_name.upper()} by Fold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean score with error bars
        ax2 = axes[1]
        param_labels = [f"P{i+1}" for i in range(len(self.cv_results))]
        mean_scores = [r['mean_score'] for r in self.cv_results]
        std_scores = [r['std_score'] for r in self.cv_results]
        
        x_pos = np.arange(len(param_labels))
        bars = ax2.bar(x_pos, mean_scores, yerr=std_scores, 
                      capsize=5, alpha=0.7, color='steelblue')
        
        # Highlight best
        best_idx = np.argmax(mean_scores)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.9)
        
        ax2.set_xlabel('Parameter Set')
        ax2.set_ylabel(f'Mean {metric_name.upper()}')
        ax2.set_title(f'Mean {metric_name.upper()} with Std Dev')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(param_labels, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """Save cross-validation results to JSON."""
        results_dict = {
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score != -np.inf else None,
            'n_splits': self.n_splits,
            'all_results': []
        }
        
        for result_entry in self.cv_results:
            entry = {
                'params': result_entry['params'],
                'mean_score': float(result_entry['mean_score']),
                'std_score': float(result_entry['std_score']),
                'mean_metrics': {k: float(v) for k, v in result_entry['cv_results']['mean'].items()},
                'std_metrics': {k: float(v) for k, v in result_entry['cv_results']['std'].items()}
            }
            results_dict['all_results'].append(entry)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load cross-validation results from JSON."""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.best_params = results_dict['best_params']
        self.best_score = results_dict['best_score'] if results_dict['best_score'] is not None else -np.inf
        self.n_splits = results_dict['n_splits']
        
        # Reconstruct cv_results
        self.cv_results = []
        for entry in results_dict['all_results']:
            self.cv_results.append({
                'params': entry['params'],
                'mean_score': entry['mean_score'],
                'std_score': entry['std_score'],
                'cv_results': {
                    'mean': entry['mean_metrics'],
                    'std': entry['std_metrics']
                }
            })
        
        print(f"Results loaded from {filepath}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best params: {self.best_params}")


def compare_cv_results(results_dict: Dict[str, Dict], 
                      metric: str = 'f1',
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 6)):
    """
    Compare cross-validation results from different models.
    
    Args:
        results_dict: Dictionary mapping model names to their CV results
        metric: Metric to compare
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    model_names = list(results_dict.keys())
    means = [results_dict[name]['mean'][metric] for name in model_names]
    stds = [results_dict[name]['std'][metric] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    
    bars = plt.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=['steelblue', 'coral', 'lightgreen'][:len(model_names)])
    
    # Highlight best
    best_idx = np.argmax(means)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)
    
    plt.ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title(f'Cross-Validation Comparison: {metric.upper()} Score', 
             fontsize=14, fontweight='bold')
    plt.xticks(x_pos, model_names)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
