import os
import json
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import evaluate_model

def generate_results_json(output_dir='outputs'):
    models = ['crf', 'bilstm', 'bert', 'roberta']
    
    for model in models:
        pred_file = os.path.join(output_dir, f'{model}_predictions.json')
        result_file = os.path.join(output_dir, f'{model}_results.json')
        
        if os.path.exists(pred_file):
            print(f"Processing {model}...")
            try:
                with open(pred_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    y_true = data['y_true']
                    y_pred = data['y_pred']
                    
                metrics = evaluate_model(y_true, y_pred)
                
                # Save only summary metrics to results.json
                summary = {
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                }
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved {result_file}")
                
            except Exception as e:
                print(f"Error processing {model}: {e}")
        else:
            print(f"Predictions file not found for {model}: {pred_file}")

if __name__ == "__main__":
    generate_results_json()
