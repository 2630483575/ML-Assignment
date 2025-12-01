"""
Dimensionality Reduction and Clustering Utilities for NER Project.

This module provides functions for:
- t-SNE visualization of word embeddings and BERT representations
- PCA analysis for dimensionality reduction
- UMAP visualization (modern alternative to t-SNE)
- K-means clustering of entity embeddings
- Visualization of clustering results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, but don't fail if not available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")


def extract_embeddings_from_model(model, word2idx: Dict[str, int], 
                                   words: List[str]) -> np.ndarray:
    """
    Extract word embeddings from a trained BiLSTM model.
    
    Args:
        model: Trained BiLSTM model with embedding layer
        word2idx: Word to index mapping
        words: List of words to extract embeddings for
        
    Returns:
        numpy array of shape (num_words, embedding_dim)
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for word in words:
            # Try exact match first, then lowercase
            if word in word2idx:
                idx = torch.tensor([word2idx[word]])
                emb = model.word_embeds(idx).cpu().numpy()
                embeddings.append(emb[0])
            elif word.lower() in word2idx:
                idx = torch.tensor([word2idx[word.lower()]])
                emb = model.word_embeds(idx).cpu().numpy()
                embeddings.append(emb[0])
            else:
                # Use UNK token if available, otherwise zero vector
                if '<UNK>' in word2idx:
                    idx = torch.tensor([word2idx['<UNK>']])
                    emb = model.word_embeds(idx).cpu().numpy()
                    embeddings.append(emb[0])
                else:
                    # Use a zero vector for unknown words
                    embeddings.append(np.zeros(model.embedding_dim))
    
    return np.array(embeddings)


def extract_bert_embeddings(model, tokenizer, words: List[str], 
                            layer: int = -1) -> np.ndarray:
    """
    Extract BERT embeddings for a list of words.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        words: List of words to extract embeddings for
        layer: Which layer to extract embeddings from (-1 = last layer)
        
    Returns:
        numpy array of shape (num_words, hidden_size)
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for word in words:
            # Tokenize the word
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            
            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
            # Average over tokens (in case word is split into subwords)
            word_embedding = hidden_states.mean(dim=1).cpu().numpy()[0]
            embeddings.append(word_embedding)
    
    return np.array(embeddings)


def perform_tsne(embeddings: np.ndarray, n_components: int = 2, 
                 perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to (2 or 3)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings of shape (n_samples, n_components)
    """
    print(f"Performing t-SNE reduction: {embeddings.shape} -> ({len(embeddings)}, {n_components})")
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_iter=1000, verbose=1)
    
    # Standardize features before t-SNE
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    reduced = tsne.fit_transform(embeddings_scaled)
    print(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return reduced


def perform_pca(embeddings: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Perform PCA dimensionality reduction.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        n_components: Number of principal components
        
    Returns:
        Tuple of (reduced embeddings, fitted PCA object)
    """
    print(f"Performing PCA reduction: {embeddings.shape} -> ({len(embeddings)}, {n_components})")
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    return reduced, pca


def perform_umap(embeddings: np.ndarray, n_components: int = 2, 
                 n_neighbors: int = 15, min_dist: float = 0.1, 
                 random_state: int = 42) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings of shape (n_samples, n_components)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    print(f"Performing UMAP reduction: {embeddings.shape} -> ({len(embeddings)}, {n_components})")
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                      min_dist=min_dist, random_state=random_state, verbose=True)
    reduced = umap_model.fit_transform(embeddings_scaled)
    
    print("UMAP completed.")
    return reduced


def perform_kmeans(embeddings: np.ndarray, n_clusters: int, 
                   random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering on embeddings.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster labels, fitted KMeans object)
    """
    print(f"Performing K-means clustering with {n_clusters} clusters")
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings_scaled)
    
    print(f"Clustering completed. Inertia: {kmeans.inertia_:.4f}")
    
    return labels, kmeans


def plot_2d_embeddings(embeddings_2d: np.ndarray, labels: List[str], 
                       colors: Optional[List[str]] = None,
                       title: str = "2D Embedding Visualization",
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8)):
    """
    Plot 2D embeddings with labels.
    
    Args:
        embeddings_2d: 2D embeddings of shape (n_samples, 2)
        labels: List of labels for each point
        colors: Optional list of colors for each point
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if colors is None:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
    else:
        # Create a color map for unique labels
        unique_labels = list(set(colors))
        color_map = plt.cm.get_cmap('tab10')
        label_to_color = {label: color_map(i / len(unique_labels)) 
                         for i, label in enumerate(unique_labels)}
        
        for label in unique_labels:
            mask = np.array(colors) == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=label, alpha=0.6, s=50, 
                       color=label_to_color[label])
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate some points (not all to avoid clutter)
    step = max(1, len(labels) // 20)  # Show max 20 labels
    for i in range(0, len(labels), step):
        plt.annotate(labels[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_clustering_results(embeddings_2d: np.ndarray, cluster_labels: np.ndarray,
                           words: List[str], entity_types: Optional[List[str]] = None,
                           title: str = "K-means Clustering Results",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 8)):
    """
    Plot clustering results with cluster assignments.
    
    Args:
        embeddings_2d: 2D embeddings of shape (n_samples, 2)
        cluster_labels: Cluster assignment for each point
        words: List of words
        entity_types: Optional true entity types for comparison
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    n_clusters = len(np.unique(cluster_labels))
    
    if entity_types is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: K-means clusters
        ax1 = axes[0]
        scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=cluster_labels, cmap='tab10', alpha=0.6, s=50)
        ax1.set_title(f'K-means Clusters (k={n_clusters})', fontweight='bold')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Plot 2: True entity types
        ax2 = axes[1]
        unique_types = list(set(entity_types))
        color_map = plt.cm.get_cmap('tab10')
        type_to_color = {t: color_map(i / len(unique_types)) 
                        for i, t in enumerate(unique_types)}
        
        for entity_type in unique_types:
            mask = np.array(entity_types) == entity_type
            ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=entity_type, alpha=0.6, s=50,
                       color=type_to_color[entity_type])
        
        ax2.set_title('True Entity Types', fontweight='bold')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
    else:
        plt.figure(figsize=figsize)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.6, s=50)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clustering plot saved to {save_path}")
    
    plt.show()


def plot_pca_variance(pca: PCA, save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6)):
    """
    Plot PCA explained variance ratio.
    
    Args:
        pca: Fitted PCA object
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Variance ratio per component
    ax1 = axes[0]
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each Component', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative variance
    ax2 = axes[1]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
    ax2.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA variance plot saved to {save_path}")
    
    plt.show()


def compare_reduction_methods(embeddings: np.ndarray, labels: List[str],
                              colors: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (18, 5)):
    """
    Compare different dimensionality reduction methods side by side.
    
    Args:
        embeddings: Original embeddings
        labels: Labels for each embedding
        colors: Optional colors for grouping points
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Perform all reductions
    pca_2d, pca_obj = perform_pca(embeddings, n_components=2)
    tsne_2d = perform_tsne(embeddings, n_components=2, perplexity=min(30, len(embeddings) - 1))
    
    methods = [
        ('PCA', pca_2d),
        ('t-SNE', tsne_2d)
    ]
    
    # Add UMAP if available
    if UMAP_AVAILABLE:
        try:
            umap_2d = perform_umap(embeddings, n_components=2)
            methods.append(('UMAP', umap_2d))
        except Exception as e:
            print(f"UMAP failed: {e}")
    
    fig, axes = plt.subplots(1, len(methods), figsize=figsize)
    if len(methods) == 1:
        axes = [axes]
    
    for ax, (method_name, reduced) in zip(axes, methods):
        if colors is None:
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
        else:
            unique_colors = list(set(colors))
            color_map = plt.cm.get_cmap('tab10')
            color_to_rgb = {c: color_map(i / len(unique_colors)) 
                           for i, c in enumerate(unique_colors)}
            
            for color in unique_colors:
                mask = np.array(colors) == color
                ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                          label=color, alpha=0.6, s=50,
                          color=color_to_rgb[color])
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        ax.set_title(f'{method_name}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Dimensionality Reduction Methods', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
