"""
Configuration classes for NER project.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "conll2003"
    data_dir: str = "data"
    max_seq_length: int = 128
    
@dataclass
class CRFConfig:
    """CRF Model configuration."""
    algorithm: str = 'lbfgs'
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 100
    all_possible_transitions: bool = True
    
@dataclass
class BiLSTMConfig:
    """BiLSTM Model configuration."""
    embedding_dim: int = 100
    hidden_dim: int = 256
    dropout: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    patience: int = 5  # For early stopping
    
@dataclass
class BERTConfig:
    model_name: str = "bert-base-cased"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 500

@dataclass
class ProjectConfig:
    """Main project configuration."""
    model_type: str = "crf"  # 'crf', 'bilstm', or 'bert'
    mode: str = "train"      # 'train', 'evaluate', 'cv', or 'visualize'
    output_dir: str = "outputs"
    seed: int = 42
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    crf: CRFConfig = field(default_factory=CRFConfig)
    bilstm: BiLSTMConfig = field(default_factory=BiLSTMConfig)
    bert: BERTConfig = field(default_factory=BERTConfig)

    def save(self, path: str):
        """Save config to file (placeholder for YAML/JSON serialization)."""
        pass
        
    @classmethod
    def load(cls, path: str):
        """Load config from file."""
        return cls()
