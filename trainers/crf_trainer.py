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

        # Save predictions for visualization
        import json
        predictions_path = os.path.join(self.config.output_dir, "crf_predictions.json")
        
        # Helper to ensure JSON serializability
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump({
                'y_true': [to_serializable(x) for x in y_test],
                'y_pred': [to_serializable(x) for x in y_pred_test]
            }, f)
        print(f"Saved predictions to {predictions_path}")
        
    def cross_validate(self, dataset, n_splits=5):
        """
        Perform cross-validation (legacy method).
        
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
    
    def grid_search_cv(self, dataset, param_grid=None, n_splits=5):
        """
        Perform grid search with cross-validation using the unified framework.
        
        Args:
            dataset: HuggingFace DatasetDict
            param_grid: Dictionary of parameters to search (if None, use defaults)
            n_splits: Number of folds
            
        Returns:
            Best parameters and all results
        """
        from utils.cross_validation import CrossValidator
        
        X_train, y_train, _, _, _, _ = self.prepare_data(dataset)
        
        # Create features-labels pairs for cross-validation
        train_data = list(zip(X_train, y_train))
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'c1': [0.01, 0.1, 0.5, 1.0],
                'c2': [0.01, 0.1, 0.5, 1.0],
                'max_iterations': [100, 200]
            }
        
        # Define train and eval functions
        def train_fn(data, params):
            X = [x for x, y in data]
            y = [y for x, y in data]
            
            model = CRFModel(
                algorithm=self.config.crf.algorithm,
                c1=params.get('c1', self.config.crf.c1),
                c2=params.get('c2', self.config.crf.c2),
                max_iterations=params.get('max_iterations', self.config.crf.max_iterations),
                all_possible_transitions=self.config.crf.all_possible_transitions
            )
            model.fit(X, y)
            return model
        
        def eval_fn(model, data):
            X = [x for x, y in data]
            y_true = [y for x, y in data]
            y_pred = model.predict(X)
            metrics = evaluate_model(y_true, y_pred)
            return metrics
        
        # Run grid search
        cv = CrossValidator(n_splits=n_splits, random_state=self.config.seed)
        best_params, all_results = cv.grid_search(
            train_fn=train_fn,
            eval_fn=eval_fn,
            dataset=train_data,
            param_grid=param_grid,
            scoring_metric='f1',
            verbose=True
        )
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "crf_grid_search_results.json")
        cv.save_results(results_path)
        
        # Plot results
        plot_path = os.path.join(self.config.output_dir, "crf_grid_search_cv.png")
        cv.plot_cv_results(metric_name='f1', save_path=plot_path)
        
        return best_params, all_results
