"""
Trainer for CRF model.
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from models.crf_model import CRFModel
from data.preprocessing import sent2features, sent2labels
from utils.metrics import evaluate_model, format_results
from utils.visualization import plot_cv_results

class CRFTrainer:
    def __init__(self, config, label_list):
        """
        Initialize CRF trainer.
        
        Args:
            config (ProjectConfig): Project configuration
            label_list (list): List of label names
        """
        self.config = config
        self.label_list = label_list
        self.model = CRFModel(
            algorithm=config.crf.algorithm,
            c1=config.crf.c1,
            c2=config.crf.c2,
            max_iterations=config.crf.max_iterations,
            all_possible_transitions=config.crf.all_possible_transitions
        )
        
    def prepare_data(self, dataset):
        """
        Prepare data for CRF training (feature extraction).
        
        Args:
            dataset: HuggingFace DatasetDict
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("Extracting features for CRF...")
        X_train = [sent2features(s) for s in dataset['train']]
        y_train = [sent2labels(s, self.label_list) for s in dataset['train']]
        
        X_val = [sent2features(s) for s in dataset['validation']]
        y_val = [sent2labels(s, self.label_list) for s in dataset['validation']]
        
        X_test = [sent2features(s) for s in dataset['test']]
        y_test = [sent2labels(s, self.label_list) for s in dataset['test']]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, dataset):
        """
        Train the model.
        
        Args:
            dataset: HuggingFace DatasetDict
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(dataset)
        
        print(f"Training CRF model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        y_pred_val = self.model.predict(X_val)
        metrics_val = evaluate_model(y_val, y_pred_val)
        print(format_results(metrics_val, prefix="Validation"))
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_pred_test = self.model.predict(X_test)
        metrics_test = evaluate_model(y_test, y_pred_test)
        print(format_results(metrics_test, prefix="Test"))
        
        # Save model
        save_path = os.path.join(self.config.output_dir, "crf_model.joblib")
        self.model.save(save_path)
        
    def cross_validate(self, dataset, n_splits=5):
        """
        Perform cross-validation.
        
        Args:
            dataset: HuggingFace DatasetDict
            n_splits (int): Number of folds
        """
        X_train, y_train, _, _, _, _ = self.prepare_data(dataset)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)
        cv_scores = []
        
        print(f"\nStarting {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            print(f"Fold {fold}/{n_splits}...")
            
            X_tr = [X_train[i] for i in train_idx]
            y_tr = [y_train[i] for i in train_idx]
            X_vl = [X_train[i] for i in val_idx]
            y_vl = [y_train[i] for i in val_idx]
            
            # Create fresh model for each fold
            fold_model = CRFModel(
                algorithm=self.config.crf.algorithm,
                c1=self.config.crf.c1,
                c2=self.config.crf.c2,
                max_iterations=self.config.crf.max_iterations,
                all_possible_transitions=self.config.crf.all_possible_transitions
            )
            
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_vl)
            metrics = evaluate_model(y_vl, y_pred)
            
            cv_scores.append(metrics['f1'])
            print(f"   F1 score: {metrics['f1']:.4f}")
            
        mean_f1 = np.mean(cv_scores)
        std_f1 = np.std(cv_scores)
        print(f"\nAverage F1: {mean_f1:.4f} ± {std_f1:.4f}")
        
        # Plot results
        plot_path = os.path.join(self.config.output_dir, "crf_cv_results.png")
        plot_cv_results(cv_scores, save_path=plot_path)
