"""
Script to compare different feature configurations for NER.

This script trains CRF models with various feature sets and compares their performance.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data.dataset import load_conll2003, get_label_mappings
from data.advanced_features import AdvancedFeatureExtractor, create_entity_gazetteer
from models.crf_model import CRFModel
from utils.metrics import evaluate_model
from config.config import ProjectConfig


def sent2features_enhanced(sent, feature_extractor: AdvancedFeatureExtractor, label_list):
    """
    Extract features from sentence using advanced feature extractor.
    """
    words = sent['tokens']
    pos_tags = sent.get('pos_tags', None)
    
    # Convert POS tag IDs to names if available
    if pos_tags:
        pos_tag_names = ['""', '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 
                        'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 
                        'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                        'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 
                        'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
        pos_tags = [pos_tag_names[tag] if tag < len(pos_tag_names) else 'UNK' 
                   for tag in pos_tags]
    
    features = []
    for i in range(len(words)):
        word_features = feature_extractor.extract_all_features(
            words, i, pos_tags=pos_tags
        )
        features.append(word_features)
    
    return features


def sent2labels_simple(sent, label_list):
    """Convert sentence tags to labels."""
    return [label_list[tag_id] for tag_id in sent['ner_tags']]


def train_and_evaluate(dataset, feature_config, label_list, config):
    """
    Train CRF model with specified feature configuration and evaluate.
    
    Returns metrics dictionary.
    """
    print(f"Training with features: {feature_config['name']}")
    
    # Create feature extractor
    feature_extractor = AdvancedFeatureExtractor(**feature_config['params'])
    
    # Extract features
    print("  Extracting features...")
    X_train = [sent2features_enhanced(s, feature_extractor, label_list) 
               for s in dataset['train']]
    y_train = [sent2labels_simple(s, label_list) for s in dataset['train']]
    
    X_test = [sent2features_enhanced(s, feature_extractor, label_list) 
              for s in dataset['test']]
    y_test = [sent2labels_simple(s, label_list) for s in dataset['test']]
    
    # Train model
    print("  Training model...")
    model = CRFModel(
        c1=config.crf.c1,
        c2=config.crf.c2,
        max_iterations=config.crf.max_iterations
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("  Evaluating...")
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    return metrics, model


def create_feature_configurations(dataset):
    """
    Create different feature configurations to compare.
    """
    # Create gazetteer from training data
    gazetteer = create_entity_gazetteer(dataset['train'])
    
    configs = [
        {
            'name': 'Baseline',
            'params': {
                'use_char_features': False,
                'use_word_embeddings': False,
                'use_gazetteer': False,
                'context_window': 0
            }
        },
        {
            'name': 'Char Features',
            'params': {
                'use_char_features': True,
                'use_word_embeddings': False,
                'use_gazetteer': False,
                'context_window': 0
            }
        },
        {
            'name': 'Char + Context',
            'params': {
                'use_char_features': True,
                'use_word_embeddings': False,
                'use_gazetteer': False,
                'context_window': 2
            }
        },
        {
            'name': 'Char + Gazetteer',
            'params': {
                'use_char_features': True,
                'use_word_embeddings': False,
                'use_gazetteer': True,
                'context_window': 0,
                'gazetteer': gazetteer
            }
        },
        {
            'name': 'All Features',
            'params': {
                'use_char_features': True,
                'use_word_embeddings': False,
                'use_gazetteer': True,
                'context_window': 2,
                'gazetteer': gazetteer
            }
        }
    ]
    
    return configs


def plot_feature_comparison(results, save_path=None):
    """
    Create visualization comparing different feature configurations.
    """
    # Prepare data
    names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in names]
    precisions = [results[name]['precision'] for name in names]
    recalls = [results[name]['recall'] for name in names]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Grouped bar chart
    ax1 = axes[0]
    x = range(len(names))
    width = 0.25
    
    ax1.bar([i - width for i in x], f1_scores, width, label='F1', alpha=0.8, color='steelblue')
    ax1.bar([i for i in x], precisions, width, label='Precision', alpha=0.8, color='coral')
    ax1.bar([i + width for i in x], recalls, width, label='Recall', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Feature Configuration', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Feature Engineering Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.7, 1.0)
    
    # Plot 2: F1 score focus
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax2.barh(names, f1_scores, color=colors, alpha=0.8)
    
    # Highlight best
    best_idx = f1_scores.index(max(f1_scores))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)
    
    ax2.set_xlabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score by Feature Set', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0.7, 1.0)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax2.text(score + 0.005, i, f'{score:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_feature_ablation_table(results, save_path=None):
    """
    Create a detailed table showing feature ablation results.
    """
    # Create DataFrame
    data = []
    for name, metrics in results.items():
        data.append({
            'Feature Set': name,
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })
    
    df = pd.DataFrame(data)
    
    # Calculate improvements over baseline
    baseline_f1 = results['Baseline']['f1']
    df['F1 Improvement'] = ((df['F1'] - baseline_f1) / baseline_f1 * 100).round(2)
    
    # Sort by F1 score
    df = df.sort_values('F1', ascending=False)
    
    # Format as markdown table
    print("\n" + "="*80)
    print("FEATURE ABLATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to file
    if save_path:
        with open(save_path, 'w') as f:
            f.write("# Feature Engineering Ablation Study\n\n")
            f.write(df.to_markdown(index=False))
            f.write(f"\n\nBaseline F1: {baseline_f1:.4f}\n")
        print(f"\nTable saved to {save_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare NER feature configurations")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: reduce iterations for testing")
    
    args = parser.parse_args()
    
    # Setup
    config = ProjectConfig()
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)
    
    if args.quick:
        config.crf.max_iterations = 50
    
    # Load dataset
    print("Loading CoNLL-2003 dataset...")
    dataset = load_conll2003()
    label_list, _, _ = get_label_mappings(dataset)
    
    # Create feature configurations
    print("\nCreating feature configurations...")
    feature_configs = create_feature_configurations(dataset)
    
    # Train and evaluate each configuration
    results = {}
    for feature_config in feature_configs:
        print(f"\n{'='*80}")
        metrics, model = train_and_evaluate(dataset, feature_config, label_list, config)
        results[feature_config['name']] = metrics
        
        # Save model
        model_path = os.path.join(
            config.output_dir, 
            f"crf_{feature_config['name'].lower().replace(' ', '_')}.joblib"
        )
        model.save(model_path)
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    plot_path = os.path.join(config.output_dir, "feature_comparison.png")
    plot_feature_comparison(results, save_path=plot_path)
    
    table_path = os.path.join(config.output_dir, "feature_ablation.md")
    df = create_feature_ablation_table(results, save_path=table_path)
    
    # Save results to JSON
    results_json_path = os.path.join(config.output_dir, "feature_comparison_results.json")
    with open(results_json_path, 'w') as f:
        json.dump({name: {k: float(v) for k, v in metrics.items() if k != 'report'} 
                  for name, metrics in results.items()}, f, indent=2)
    print(f"Results saved to {results_json_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    best_config = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"Best Configuration: {best_config[0]}")
    print(f"Best F1 Score: {best_config[1]['f1']:.4f}")
    print(f"Improvement over baseline: {(best_config[1]['f1'] - results['Baseline']['f1']) / results['Baseline']['f1'] * 100:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    import numpy as np  # Import needed for bar chart colors
    main()
