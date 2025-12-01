"""
Script to run comprehensive data analysis on the NER dataset.

Usage:
    python scripts/run_data_analysis.py --output_dir outputs/data_analysis
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from data.dataset import load_conll2003, get_label_mappings
from data.data_analysis import NERDataAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive NER data analysis")
    parser.add_argument("--output_dir", type=str, default="outputs/data_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("NER DATASET COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load dataset
    print("\nLoading CoNLL-2003 dataset...")
    dataset = load_conll2003()
    label_list, label2id, id2label = get_label_mappings(dataset)
    
    print(f"Dataset loaded successfully!")
    print(f"  Train: {len(dataset['train'])} sentences")
    print(f"  Validation: {len(dataset['validation'])} sentences")
    print(f"  Test: {len(dataset['test'])} sentences")
    print(f"  Label types: {len(label_list)}")
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = NERDataAnalyzer(dataset)
    analyzer.set_label_names(label_list)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive analysis report...")
    report_path = analyzer.generate_comprehensive_report(args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nReport saved to: {report_path}")
    print(f"Visualizations saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - data_analysis_report.md")
    print(f"  - entity_distribution_across_splits.png")
    print(f"  - label_imbalance.png")
    print(f"  - entity_cooccurrence.png")
    print("="*80)


if __name__ == "__main__":
    main()
