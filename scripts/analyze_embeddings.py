"""
Script to perform dimensionality reduction and clustering analysis on NER embeddings.

This script:
1. Loads trained models (BiLSTM, BERT)
2. Extracts embeddings for sample entities
3. Performs dimensionality reduction (t-SNE, PCA, UMAP)
4. Performs K-means clustering
5. Creates visualizations
"""

import os
import sys
import argparse
import numpy as np
import torch
import joblib
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_conll2003, get_label_mappings
from data.preprocessing import build_vocab
from models.bilstm_crf import BiLSTM_CRF
from models.bert_model import BERTModel
from config.config import ProjectConfig
from utils.dimensionality_reduction import (
    extract_embeddings_from_model,
    extract_bert_embeddings,
    perform_tsne,
    perform_pca,
    perform_umap,
    perform_kmeans,
    plot_2d_embeddings,
    plot_clustering_results,
    plot_pca_variance,
    compare_reduction_methods,
    UMAP_AVAILABLE
)


def collect_entity_samples(dataset, max_samples_per_type=100):
    """
    Collect sample entities from the dataset for analysis.
    
    Returns:
        Dict with entity_type -> list of (word, entity_type) tuples
    """
    print("Collecting entity samples from dataset...")
    
    entity_samples = defaultdict(list)
    entity_counts = defaultdict(int)
    
    for split in ['train', 'validation', 'test']:
        for example in dataset[split]:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            for word, tag_id in zip(tokens, ner_tags):
                # Map tag_id to tag name
                tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 
                           'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
                tag = tag_names[tag_id]
                
                # Only keep B- tags (beginning of entities)
                if tag.startswith('B-'):
                    entity_type = tag[2:]  # Remove 'B-' prefix
                    
                    if entity_counts[entity_type] < max_samples_per_type:
                        entity_samples[entity_type].append((word, entity_type))
                        entity_counts[entity_type] += 1
    
    # Also add some non-entity words (tagged as 'O')
    non_entity_words = []
    # Use select to get a Dataset object that we can iterate over, instead of slicing which returns a dict
    for example in dataset['train'].select(range(50)):  # First 50 sentences
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        for word, tag_id in zip(tokens, ner_tags):
            if tag_id == 0 and len(non_entity_words) < max_samples_per_type:  # O tag
                non_entity_words.append((word, 'O'))
    
    entity_samples['O'] = non_entity_words
    
    print(f"Collected samples:")
    for entity_type, samples in entity_samples.items():
        print(f"  {entity_type}: {len(samples)} samples")
    
    return entity_samples


def analyze_bilstm_embeddings(config, entity_samples, output_dir):
    """Analyze BiLSTM embeddings."""
    print("\n" + "="*80)
    print("ANALYZING BiLSTM EMBEDDINGS")
    print("="*80)
    
    model_path = os.path.join(config.output_dir, "best_bilstm_crf.pt")
    
    if not os.path.exists(model_path):
        print(f"BiLSTM model not found at {model_path}. Skipping BiLSTM analysis.")
        return
    
    # Load dataset and build vocab
    dataset = load_conll2003()
    label_list, _, _ = get_label_mappings(dataset)
    
    # Build vocabulary using the same logic as training
    word2idx = build_vocab(dataset['train'])
    
    # Load model
    print(f"Loading BiLSTM model from {model_path}...")
    vocab_size = len(word2idx)
    model = BiLSTM_CRF(
        vocab_size=vocab_size,
        tag_to_ix={tag: i for i, tag in enumerate(label_list)},
        embedding_dim=config.bilstm.embedding_dim,
        hidden_dim=config.bilstm.hidden_dim
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    # Handle case where checkpoint is just the state dict or a full dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Collect all words and entity types
    all_words = []
    all_entity_types = []
    
    for entity_type, samples in entity_samples.items():
        for word, etype in samples:
            all_words.append(word)
            all_entity_types.append(etype)
    
    print(f"Extracting embeddings for {len(all_words)} words...")
    embeddings = extract_embeddings_from_model(model, word2idx, all_words)
    
    # Perform dimensionality reduction
    print("\n--- PCA Analysis ---")
    pca_2d, pca = perform_pca(embeddings, n_components=2)
    plot_pca_variance(pca, save_path=os.path.join(output_dir, "bilstm_pca_variance.png"))
    
    print("\n--- t-SNE Analysis ---")
    tsne_2d = perform_tsne(embeddings, n_components=2)
    
    if UMAP_AVAILABLE:
        print("\n--- UMAP Analysis ---")
        try:
            umap_2d = perform_umap(embeddings, n_components=2)
        except Exception as e:
            print(f"UMAP failed: {e}")
            umap_2d = None
    else:
        umap_2d = None
    
    # Plot individual methods
    plot_2d_embeddings(
        pca_2d, all_words, colors=all_entity_types,
        title="BiLSTM Embeddings - PCA",
        save_path=os.path.join(output_dir, "bilstm_pca.png")
    )
    
    plot_2d_embeddings(
        tsne_2d, all_words, colors=all_entity_types,
        title="BiLSTM Embeddings - t-SNE",
        save_path=os.path.join(output_dir, "bilstm_tsne.png")
    )
    
    # Compare all methods
    compare_reduction_methods(
        embeddings, all_words, colors=all_entity_types,
        save_path=os.path.join(output_dir, "bilstm_comparison.png")
    )
    
    # K-means clustering
    print("\n--- K-means Clustering ---")
    n_clusters = len(entity_samples)  # One cluster per entity type
    cluster_labels, kmeans = perform_kmeans(embeddings, n_clusters=n_clusters)
    
    plot_clustering_results(
        tsne_2d, cluster_labels, all_words, entity_types=all_entity_types,
        title=f"BiLSTM Embeddings - K-means (k={n_clusters})",
        save_path=os.path.join(output_dir, "bilstm_clustering.png")
    )
    
    print("\nBiLSTM analysis completed!")


def analyze_bert_embeddings(config, entity_samples, output_dir):
    """Analyze BERT embeddings."""
    print("\n" + "="*80)
    print("ANALYZING BERT EMBEDDINGS")
    print("="*80)
    
    model_path = os.path.join(config.output_dir, "best_bert_model")
    
    if not os.path.exists(model_path):
        print(f"BERT model not found at {model_path}. Skipping BERT analysis.")
        return
    
    # Load dataset
    dataset = load_conll2003()
    label_list, label2id, id2label = get_label_mappings(dataset)
    
    # Load BERT model
    print(f"Loading BERT model from {model_path}...")
    model_wrapper = BERTModel(config, len(label_list), id2label, label2id)
    model_wrapper.load(model_path)
    
    # Collect all words and entity types
    all_words = []
    all_entity_types = []
    
    for entity_type, samples in entity_samples.items():
        for word, etype in samples[:50]:  # Limit to 50 per type for speed
            all_words.append(word)
            all_entity_types.append(etype)
    
    print(f"Extracting BERT embeddings for {len(all_words)} words...")
    embeddings = extract_bert_embeddings(
        model_wrapper.model, 
        model_wrapper.tokenizer, 
        all_words
    )
    
    # Perform dimensionality reduction
    print("\n--- PCA Analysis ---")
    pca_2d, pca = perform_pca(embeddings, n_components=2)
    plot_pca_variance(pca, save_path=os.path.join(output_dir, "bert_pca_variance.png"))
    
    print("\n--- t-SNE Analysis ---")
    tsne_2d = perform_tsne(embeddings, n_components=2)
    
    if UMAP_AVAILABLE:
        print("\n--- UMAP Analysis ---")
        try:
            umap_2d = perform_umap(embeddings, n_components=2)
        except Exception as e:
            print(f"UMAP failed: {e}")
            umap_2d = None
    else:
        umap_2d = None
    
    # Plot individual methods
    plot_2d_embeddings(
        pca_2d, all_words, colors=all_entity_types,
        title="BERT Embeddings - PCA",
        save_path=os.path.join(output_dir, "bert_pca.png")
    )
    
    plot_2d_embeddings(
        tsne_2d, all_words, colors=all_entity_types,
        title="BERT Embeddings - t-SNE",
        save_path=os.path.join(output_dir, "bert_tsne.png")
    )
    
    # Compare all methods
    compare_reduction_methods(
        embeddings, all_words, colors=all_entity_types,
        save_path=os.path.join(output_dir, "bert_comparison.png")
    )
    
    # K-means clustering
    print("\n--- K-means Clustering ---")
    n_clusters = len(entity_samples)  # One cluster per entity type
    cluster_labels, kmeans = perform_kmeans(embeddings, n_clusters=n_clusters)
    
    plot_clustering_results(
        tsne_2d, cluster_labels, all_words, entity_types=all_entity_types,
        title=f"BERT Embeddings - K-means (k={n_clusters})",
        save_path=os.path.join(output_dir, "bert_clustering.png")
    )
    
    print("\nBERT analysis completed!")


def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings with dimensionality reduction")
    parser.add_argument("--model", type=str, default="both", 
                       choices=["bilstm", "bert", "both"],
                       help="Which model to analyze")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for models and plots")
    parser.add_argument("--analysis_dir", type=str, default="outputs/analysis",
                       help="Directory to save analysis results")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Max samples per entity type")
    
    args = parser.parse_args()
    
    # Setup
    config = ProjectConfig()
    config.output_dir = args.output_dir
    
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    # Load dataset and collect samples
    dataset = load_conll2003()
    entity_samples = collect_entity_samples(dataset, max_samples_per_type=args.max_samples)
    
    # Perform analysis
    if args.model in ["bilstm", "both"]:
        try:
            analyze_bilstm_embeddings(config, entity_samples, args.analysis_dir)
        except Exception as e:
            print(f"BiLSTM analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    if args.model in ["bert", "both"]:
        try:
            analyze_bert_embeddings(config, entity_samples, args.analysis_dir)
        except Exception as e:
            print(f"BERT analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {args.analysis_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
