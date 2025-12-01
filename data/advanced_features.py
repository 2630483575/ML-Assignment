"""
Advanced Feature Engineering for NER.

This module provides enhanced feature extraction methods including:
- Pre-trained word embeddings (GloVe, FastText)
- Character-level features
- Extended context window features
- Morphological features
- Gazetteer features (entity dictionaries)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import string
import re


class AdvancedFeatureExtractor:
    """
    Enhanced feature extractor for NER with multiple feature types.
    """
    
    def __init__(self, 
                 use_char_features: bool = True,
                 use_word_embeddings: bool = False,
                 use_gazetteer: bool = False,
                 context_window: int = 2,
                 word_embeddings: Optional[Dict[str, np.ndarray]] = None,
                 gazetteer: Optional[Set[str]] = None):
        """
        Initialize advanced feature extractor.
        
        Args:
            use_char_features: Whether to use character-level features
            use_word_embeddings: Whether to use pre-trained embeddings
            use_gazetteer: Whether to use entity gazetteer
            context_window: Size of context window (±N words)
            word_embeddings: Dictionary mapping words to embedding vectors
            gazetteer: Set of known entity words
        """
        self.use_char_features = use_char_features
        self.use_word_embeddings = use_word_embeddings
        self.use_gazetteer = use_gazetteer
        self.context_window = context_window
        self.word_embeddings = word_embeddings or {}
        self.gazetteer = gazetteer or set()
    
    def extract_char_features(self, word: str) -> Dict[str, any]:
        """
        Extract character-level features from a word.
        
        Features:
        - Character n-grams (prefix/suffix)
        - Character types (all caps, digits, etc.)
        - Special characters
        """
        features = {}
        
        # Prefix and suffix n-grams (2-4 characters)
        for n in range(2, min(5, len(word) + 1)):
            features[f'prefix-{n}'] = word[:n].lower()
            features[f'suffix-{n}'] = word[-n:].lower()
        
        # Character type patterns
        features['word.length'] = len(word)
        features['char.digits'] = sum(c.isdigit() for c in word)
        features['char.letters'] = sum(c.isalpha() for c in word)
        features['char.upper'] = sum(c.isupper() for c in word)
        features['char.lower'] = sum(c.islower() for c in word)
        
        # Special character indicators
        features['has.hyphen'] = '-' in word
        features['has.apostrophe'] = "'" in word
        features['has.period'] = '.' in word
        
        # Pattern features
        features['all.caps'] = word.isupper()
        features['all.lower'] = word.islower()
        features['title.case'] = word.istitle()
        features['mixed.case'] = any(c.isupper() for c in word) and any(c.islower() for c in word)
        
        # Digit patterns
        if any(c.isdigit() for c in word):
            features['has.digit'] = True
            features['all.digits'] = word.isdigit()
            features['starts.digit'] = word[0].isdigit()
            features['ends.digit'] = word[-1].isdigit()
        
        # Shape features (character-level shape)
        shape = ''.join(['X' if c.isupper() else 'x' if c.islower() 
                        else 'd' if c.isdigit() else c for c in word])
        features['word.shape'] = shape[:20]  # Truncate very long shapes
        
        # Short shape (consecutive chars collapsed)
        short_shape = re.sub(r'X+', 'X', shape)
        short_shape = re.sub(r'x+', 'x', short_shape)
        short_shape = re.sub(r'd+', 'd', short_shape)
        features['word.short_shape'] = short_shape[:10]
        
        return features
    
    def extract_morphological_features(self, word: str) -> Dict[str, any]:
        """
        Extract morphological features.
        
        Features:
        - Common prefixes/suffixes
        - Word formation patterns
        """
        features = {}
        
        # Common English prefixes
        common_prefixes = ['un', 'in', 'im', 'dis', 're', 'pre', 'post', 
                          'anti', 'de', 'over', 'sub', 'inter', 'trans']
        for prefix in common_prefixes:
            if word.lower().startswith(prefix):
                features[f'prefix.{prefix}'] = True
        
        # Common English suffixes
        common_suffixes = ['ing', 'ed', 'ly', 'er', 'est', 'tion', 'ation',
                          'ness', 'ment', 'ity', 'able', 'ible', 'al', 'ial',
                          'ous', 'ious', 'ful', 'less', 'ish']
        for suffix in common_suffixes:
            if word.lower().endswith(suffix):
                features[f'suffix.{suffix}'] = True
        
        return features
    
    def get_embedding_features(self, word: str) -> Dict[str, float]:
        """
        Get pre-trained word embedding features.
        
        Returns embedding values as features (binned for CRF).
        """
        features = {}
        
        if not self.use_word_embeddings or not self.word_embeddings:
            return features
        
        # Look up embedding (try lowercase if not found)
        embedding = self.word_embeddings.get(word)
        if embedding is None:
            embedding = self.word_embeddings.get(word.lower())
        
        if embedding is not None:
            # Bin embedding dimensions for CRF (cannot use continuous values directly)
            # Use quantized embedding values
            for i in range(min(50, len(embedding))):  # Use first 50 dims
                val = embedding[i]
                # Bin into ranges
                if val < -0.5:
                    bin_val = 'very_low'
                elif val < -0.1:
                    bin_val = 'low'
                elif val < 0.1:
                    bin_val = 'neutral'
                elif val < 0.5:
                    bin_val = 'high'
                else:
                    bin_val = 'very_high'
                
                features[f'emb[{i}]'] = bin_val
        else:
            features['emb.unknown'] = True
        
        return features
    
    def get_gazetteer_features(self, word: str) -> Dict[str, bool]:
        """
        Get gazetteer (entity dictionary) features.
        """
        features = {}
        
        if not self.use_gazetteer or not self.gazetteer:
            return features
        
        # Check if word is in gazetteer
        features['in.gazetteer'] = word in self.gazetteer or word.lower() in self.gazetteer
        
        return features
    
    def get_context_features(self, 
                            words: List[str], 
                            index: int,
                            base_feature_func) -> Dict[str, any]:
        """
        Get features from context window.
        
        Args:
            words: List of words in sentence
            index: Current word index
            base_feature_func: Function to extract base features from a word
        """
        features = {}
        
        # Get features from surrounding words
        for offset in range(-self.context_window, self.context_window + 1):
            if offset == 0:
                continue
            
            ctx_idx = index + offset
            if 0 <= ctx_idx < len(words):
                ctx_word = words[ctx_idx]
                
                # Add basic features from context word
                ctx_features = base_feature_func(ctx_word)
                for key, value in ctx_features.items():
                    # Prefix with offset
                    if offset < 0:
                        features[f'{offset}:{key}'] = value
                    else:
                        features[f'+{offset}:{key}'] = value
        
        return features
    
    def extract_all_features(self, 
                            words: List[str],
                            index: int,
                            pos_tags: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Extract all configured features for a word.
        
        Args:
            words: List of words in sentence
            index: Index of current word
            pos_tags: Optional POS tags
            
        Returns:
            Dictionary of features
        """
        word = words[index]
        features = {}
        
        # Basic word features
        features['word.lower'] = word.lower()
        features['word.isupper'] = word.isupper()
        features['word.istitle'] = word.istitle()
        features['word.isdigit'] = word.isdigit()
        
        # Character-level features
        if self.use_char_features:
            char_features = self.extract_char_features(word)
            features.update(char_features)
        
        # Morphological features
        morph_features = self.extract_morphological_features(word)
        features.update(morph_features)
        
        # Word embedding features
        if self.use_word_embeddings:
            emb_features = self.get_embedding_features(word)
            features.update(emb_features)
        
        # Gazetteer features
        if self.use_gazetteer:
            gaz_features = self.get_gazetteer_features(word)
            features.update(gaz_features)
        
        # POS tag features
        if pos_tags and index < len(pos_tags):
            features['pos'] = pos_tags[index]
            # POS tag bigrams
            if index > 0:
                features['pos_bigram'] = f"{pos_tags[index-1]}_{pos_tags[index]}"
        
        # Position in sentence
        features['is_first'] = index == 0
        features['is_last'] = index == len(words) - 1
        
        return features


def load_glove_embeddings(file_path: str, vocab: Set[str] = None, 
                         limit: int = None) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from file.
    
    Args:
        file_path: Path to GloVe file (e.g., glove.6B.100d.txt)
        vocab: Optional vocabulary set to filter embeddings
        limit: Optional limit on number of embeddings to load
        
    Returns:
        Dictionary mapping words to embedding vectors
    """
    embeddings = {}
    
    print(f"Loading GloVe embeddings from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            parts = line.strip().split()
            word = parts[0]
            
            # Skip if vocab provided and word not in vocab
            if vocab and word not in vocab and word.lower() not in vocab:
                continue
            
            try:
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                continue
            
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1} embeddings...")
    
    print(f"Loaded {len(embeddings)} GloVe embeddings")
    return embeddings


def create_entity_gazetteer(dataset) -> Set[str]:
    """
    Create a gazetteer (entity dictionary) from the training data.
    
    Args:
        dataset: Training dataset
        
    Returns:
        Set of unique entity words
    """
    gazetteer = set()
    
    for example in dataset:
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        for token, tag in zip(tokens, ner_tags):
            if tag != 0:  # Not 'O' tag
                gazetteer.add(token)
                gazetteer.add(token.lower())
    
    print(f"Created gazetteer with {len(gazetteer)} unique entity words")
    return gazetteer


def compare_feature_sets(dataset, feature_configs: List[Tuple[str, dict]], 
                        output_dir: str = "outputs"):
    """
    Compare different feature configurations.
    
    Args:
        dataset: Dataset to use
        feature_configs: List of (name, config_dict) tuples
        output_dir: Output directory for results
    """
    from trainers.crf_trainer import CRFTrainer
    from config.config import ProjectConfig
    from data.dataset import get_label_mappings
    import os
    
    config = ProjectConfig()
    config.output_dir = output_dir
    label_list, _, _ = get_label_mappings(dataset)
    
    results = {}
    
    for name, feature_config in feature_configs:
        print(f"\n{'='*60}")
        print(f"Testing feature configuration: {name}")
        print(f"{'='*60}")
        
        # Train with this configuration
        # (Implementation would require modifying CRF trainer to accept feature extractor)
        # For now, this is a placeholder
        
        # Store results
        results[name] = {
            'f1': 0.0,  # Placeholder
            'precision': 0.0,
            'recall': 0.0
        }
    
    return results


def visualize_feature_importance(model, feature_names: List[str], 
                                 top_k: int = 20, save_path: str = None):
    """
    Visualize feature importance from a trained CRF model.
    
    Args:
        model: Trained CRF model
        feature_names: List of feature names
        top_k: Number of top features to show
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Get feature weights (this is model-specific)
    # Placeholder for actual implementation
    print("Feature importance visualization would be implemented here")
    
    # For CRF, we could extract transition and state weights
    # and show which features are most predictive for each label
