"""
Comprehensive Data Analysis Module for NER Dataset.

This module provides in-depth analysis of the CoNLL-2003 dataset including:
- Entity distribution and statistics
- Text complexity metrics
- Data quality assessment
- Label imbalance analysis
- Entity co-occurrence patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import os


class NERDataAnalyzer:
    """Comprehensive analyzer for NER datasets."""
    
    def __init__(self, dataset):
        """
        Initialize analyzer with dataset.
        
        Args:
            dataset: HuggingFace Dataset object
        """
        self.dataset = dataset
        self.label_names = None
        self._stats_cache = {}
    
    def set_label_names(self, label_names: List[str]):
        """Set label names for the dataset."""
        self.label_names = label_names
    
    def analyze_entity_distribution(self, split='train') -> Dict:
        """
        Analyze entity distribution in detail.
        
        Returns:
            Dictionary with entity statistics
        """
        entity_counts = Counter()
        entity_lengths = defaultdict(list)
        entity_positions = defaultdict(list)  # position in sentence (beginning, middle, end)
        
        for example in self.dataset[split]:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            current_entity = None
            entity_start = 0
            
            for i, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
                if self.label_names:
                    tag = self.label_names[tag_id]
                else:
                    tag = str(tag_id)
                
                # Skip 'O' tags
                if tag == 'O':
                    if current_entity:
                        # End of entity
                        entity_length = i - entity_start
                        entity_lengths[current_entity].append(entity_length)
                        
                        # Calculate position
                        position = entity_start / len(tokens)
                        if position < 0.33:
                            entity_positions[current_entity].append('beginning')
                        elif position < 0.67:
                            entity_positions[current_entity].append('middle')
                        else:
                            entity_positions[current_entity].append('end')
                        
                        current_entity = None
                    continue
                
                # Extract entity type (remove B- or I- prefix)
                if tag.startswith('B-') or tag.startswith('I-'):
                    entity_type = tag[2:]
                else:
                    entity_type = tag
                
                # Count entities
                if tag.startswith('B-') or (current_entity is None):
                    entity_counts[entity_type] += 1
                    current_entity = entity_type
                    entity_start = i
        
        return {
            'counts': dict(entity_counts),
            'lengths': {k: np.mean(v) if v else 0 for k, v in entity_lengths.items()},
            'positions': entity_positions
        }
    
    def analyze_text_complexity(self, split='train') -> Dict:
        """
        Analyze text complexity metrics.
        
        Returns:
            Dictionary with text statistics
        """
        sentence_lengths = []
        word_lengths = []
        unique_words = set()
        total_words = 0
        
        for example in self.dataset[split]:
            tokens = example['tokens']
            sentence_lengths.append(len(tokens))
            
            for token in tokens:
                word_lengths.append(len(token))
                unique_words.add(token.lower())
                total_words += 1
        
        return {
            'avg_sentence_length': np.mean(sentence_lengths),
            'median_sentence_length': np.median(sentence_lengths),
            'std_sentence_length': np.std(sentence_lengths),
            'max_sentence_length': np.max(sentence_lengths),
            'min_sentence_length': np.min(sentence_lengths),
            'avg_word_length': np.mean(word_lengths),
            'vocabulary_size': len(unique_words),
            'total_words': total_words,
            'type_token_ratio': len(unique_words) / total_words if total_words > 0 else 0
        }
    
    def analyze_label_imbalance(self, split='train') -> Dict:
        """
        Analyze label distribution and imbalance.
        
        Returns:
            Dictionary with imbalance metrics
        """
        label_counts = Counter()
        
        for example in self.dataset[split]:
            for tag_id in example['ner_tags']:
                if self.label_names:
                    label = self.label_names[tag_id]
                else:
                    label = str(tag_id)
                label_counts[label] += 1
        
        total = sum(label_counts.values())
        label_ratios = {k: v/total for k, v in label_counts.items()}
        
        # Calculate imbalance ratio
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return {
            'label_counts': dict(label_counts),
            'label_ratios': label_ratios,
            'imbalance_ratio': imbalance_ratio,
            'most_common': label_counts.most_common(1)[0],
            'least_common': label_counts.most_common()[-1]
        }
    
    def analyze_entity_cooccurrence(self, split='train') -> pd.DataFrame:
        """
        Analyze which entity types co-occur in the same sentence.
        
        Returns:
            DataFrame with co-occurrence matrix
        """
        # Get unique entity types
        entity_types = set()
        for example in self.dataset[split]:
            for tag_id in example['ner_tags']:
                if self.label_names:
                    tag = self.label_names[tag_id]
                else:
                    tag = str(tag_id)
                
                if tag != 'O':
                    if tag.startswith('B-') or tag.startswith('I-'):
                        entity_types.add(tag[2:])
                    else:
                        entity_types.add(tag)
        
        entity_types = sorted(list(entity_types))
        
        # Count co-occurrences
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for example in self.dataset[split]:
            # Get entities in this sentence
            sentence_entities = set()
            for tag_id in example['ner_tags']:
                if self.label_names:
                    tag = self.label_names[tag_id]
                else:
                    tag = str(tag_id)
                
                if tag != 'O':
                    if tag.startswith('B-') or tag.startswith('I-'):
                        sentence_entities.add(tag[2:])
                    else:
                        sentence_entities.add(tag)
            
            # Count pairs
            sentence_entities_list = list(sentence_entities)
            for i, entity1 in enumerate(sentence_entities_list):
                for entity2 in sentence_entities_list[i:]:
                    cooccurrence[entity1][entity2] += 1
                    if entity1 != entity2:
                        cooccurrence[entity2][entity1] += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(0, index=entity_types, columns=entity_types)
        for e1 in entity_types:
            for e2 in entity_types:
                df.loc[e1, e2] = cooccurrence[e1][e2]
        
        return df
    
    def analyze_data_quality(self, split='train') -> Dict:
        """
        Assess data quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        empty_sentences = 0
        single_token_sentences = 0
        long_sentences = 0
        special_char_ratio = []
        digit_ratio = []
        no_entity_sentences = 0
        
        for example in self.dataset[split]:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Check sentence length
            if len(tokens) == 0:
                empty_sentences += 1
            elif len(tokens) == 1:
                single_token_sentences += 1
            elif len(tokens) > 100:
                long_sentences += 1
            
            # Check for entities
            has_entity = any(tag != 0 for tag in ner_tags)  # Assuming 0 is 'O'
            if not has_entity:
                no_entity_sentences += 1
            
            # Character statistics
            all_text = ' '.join(tokens)
            special_chars = sum(1 for c in all_text if not c.isalnum() and not c.isspace())
            digits = sum(1 for c in all_text if c.isdigit())
            
            if len(all_text) > 0:
                special_char_ratio.append(special_chars / len(all_text))
                digit_ratio.append(digits / len(all_text))
        
        total_sentences = len(self.dataset[split])
        
        return {
            'total_sentences': total_sentences,
            'empty_sentences': empty_sentences,
            'single_token_sentences': single_token_sentences,
            'long_sentences': long_sentences,
            'no_entity_sentences': no_entity_sentences,
            'avg_special_char_ratio': np.mean(special_char_ratio),
            'avg_digit_ratio': np.mean(digit_ratio),
            'sentences_with_entities': total_sentences - no_entity_sentences,
            'entity_sentence_ratio': (total_sentences - no_entity_sentences) / total_sentences
        }
    
    def generate_comprehensive_report(self, output_dir='outputs/data_analysis'):
        """
        Generate comprehensive data analysis report with visualizations.
        
        Args:
            output_dir: Directory to save report and figures
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report = []
        report.append("# Comprehensive NER Data Analysis Report\n")
        report.append(f"Dataset: CoNLL-2003\n\n")
        
        # Analyze each split
        for split in ['train', 'validation', 'test']:
            if split not in self.dataset:
                continue
            
            report.append(f"## {split.upper()} Set Analysis\n\n")
            
            # Entity distribution
            report.append("### Entity Distribution\n\n")
            entity_stats = self.analyze_entity_distribution(split)
            report.append("**Entity Counts:**\n")
            for entity, count in sorted(entity_stats['counts'].items(), key=lambda x: -x[1]):
                report.append(f"- {entity}: {count}\n")
            
            report.append("\n**Average Entity Lengths (tokens):**\n")
            for entity, length in sorted(entity_stats['lengths'].items()):
                report.append(f"- {entity}: {length:.2f}\n")
            
            # Text complexity
            report.append("\n### Text Complexity\n\n")
            complexity = self.analyze_text_complexity(split)
            report.append(f"- Average sentence length: {complexity['avg_sentence_length']:.2f} tokens\n")
            report.append(f"- Median sentence length: {complexity['median_sentence_length']:.1f} tokens\n")
            report.append(f"- Std deviation: {complexity['std_sentence_length']:.2f}\n")
            report.append(f"- Range: {complexity['min_sentence_length']}-{complexity['max_sentence_length']} tokens\n")
            report.append(f"- Vocabulary size: {complexity['vocabulary_size']:,}\n")
            report.append(f"- Total words: {complexity['total_words']:,}\n")
            report.append(f"- Type-Token Ratio: {complexity['type_token_ratio']:.4f}\n")
            
            # Label imbalance
            report.append("\n### Label Imbalance\n\n")
            imbalance = self.analyze_label_imbalance(split)
            report.append(f"- Imbalance ratio: {imbalance['imbalance_ratio']:.2f}\n")
            report.append(f"- Most common label: {imbalance['most_common'][0]} ({imbalance['most_common'][1]:,} occurrences)\n")
            report.append(f"- Least common label: {imbalance['least_common'][0]} ({imbalance['least_common'][1]:,} occurrences)\n")
            
            # Data quality
            report.append("\n### Data Quality\n\n")
            quality = self.analyze_data_quality(split)
            report.append(f"- Total sentences: {quality['total_sentences']:,}\n")
            report.append(f"- Sentences with entities: {quality['sentences_with_entities']:,} ({quality['entity_sentence_ratio']*100:.1f}%)\n")
            report.append(f"- Empty sentences: {quality['empty_sentences']}\n")
            report.append(f"- Single-token sentences: {quality['single_token_sentences']}\n")
            report.append(f"- Long sentences (>100 tokens): {quality['long_sentences']}\n")
            
            report.append("\n" + "="*80 + "\n\n")
        
        # Save report
        report_path = os.path.join(output_dir, 'data_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"Data analysis report saved to: {report_path}")
        
        # Generate visualizations
        self._create_visualizations(output_dir)
        
        return report_path
    
    def _create_visualizations(self, output_dir):
        """Create visualization plots for the report."""
        # 1. Entity distribution across splits
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, split in enumerate(['train', 'validation', 'test']):
            if split not in self.dataset:
                continue
            
            entity_stats = self.analyze_entity_distribution(split)
            entities = list(entity_stats['counts'].keys())
            counts = list(entity_stats['counts'].values())
            
            axes[idx].bar(entities, counts, color=plt.cm.Set3(range(len(entities))))
            axes[idx].set_title(f'{split.capitalize()} Set', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Entity Type')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, (entity, count) in enumerate(zip(entities, counts)):
                axes[idx].text(i, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entity_distribution_across_splits.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Label imbalance visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        imbalance = self.analyze_label_imbalance('train')
        labels = list(imbalance['label_ratios'].keys())
        ratios = [imbalance['label_ratios'][l] * 100 for l in labels]
        
        colors = ['red' if r < 1 else 'orange' if r < 5 else 'green' for r in ratios]
        bars = ax.barh(labels, ratios, color=colors, alpha=0.7)
        
        ax.set_xlabel('Percentage (%)', fontweight='bold')
        ax.set_title('Label Distribution (Train Set)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_imbalance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Entity co-occurrence heatmap
        cooccurrence_df = self.analyze_entity_cooccurrence('train')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cooccurrence_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('Entity Co-occurrence Matrix (Train Set)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entity_cooccurrence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
