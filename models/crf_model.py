"""
CRF Model implementation using sklearn-crfsuite.
"""
import sklearn_crfsuite
import joblib
import os

class CRFModel:
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=False):
        """
        Initialize CRF model.
        
        Args:
            algorithm (str): Optimization algorithm
            c1 (float): L1 regularization coefficient
            c2 (float): L2 regularization coefficient
            max_iterations (int): Maximum number of iterations
            all_possible_transitions (bool): Whether to include all possible transitions
            verbose (bool): Whether to print verbose output
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
            verbose=verbose
        )
        
    def fit(self, X_train, y_train):
        """
        Train the CRF model.
        
        Args:
            X_train (list): List of feature sequences
            y_train (list): List of label sequences
        """
        print(f"Training CRF model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("Training completed.")
        
    def predict(self, X):
        """
        Predict labels for sequences.
        
        Args:
            X (list): List of feature sequences
            
        Returns:
            list: List of predicted label sequences
        """
        return self.model.predict(X)
        
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        
    def load(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found at {filepath}")
