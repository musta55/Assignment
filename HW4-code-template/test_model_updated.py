#!/usr/bin/env python
# coding: utf-8
'''Quick test script for the trained model with updated parameters'''

import torch
from torch import nn
import json
from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from LSTM import LSTMModel
import numpy as np

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Model parameters (UPDATED to match GloVe 200d training)
    input_len = 150
    embedding_dim = 200  # GloVe 200d
    hidden_dim = 128     # Increased from 50
    output_size = 1
    n_layers = 2         # Increased from 1
    Batch_size = 256
    
    # Load vocabulary
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)
    print(f'Vocabulary size: {vocab_size}')
    
    # Load test data
    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size, 
                               shuffle=False, num_workers=1)
    print(f'Test samples: {len(test_set)}')
    
    # Create model with updated parameters
    model = LSTMModel(vocab_size, output_size, embedding_dim, None,
                     hidden_dim, n_layers, input_len, pretrain=False)
    
    # Load checkpoint
    checkpoint_path = 'cpt/sentiment_model.pt'
    print(f'\nLoading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    
    # Test the model
    print('\n' + '='*80)
    print('TESTING MODEL')
    print('='*80)
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_labels) in enumerate(test_generator):
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            
            # Get predictions
            y_out = model(x_batch)
            
            # Round predictions to 0 or 1
            y_pred = torch.round(y_out)
            
            # Calculate accuracy
            correct += (y_pred == y_labels).sum().item()
            total += y_labels.size(0)
            
            # Store for confusion matrix
            all_predictions.extend(y_pred.cpu().numpy())
            all_labels.extend(y_labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1}/{len(test_generator)} batches...', end='\r')
    
    print('\n')
    
    # Calculate metrics
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Correct: {correct}/{total}')
    
    # Confusion matrix
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    true_positive = np.sum((all_predictions == 1) & (all_labels == 1))
    true_negative = np.sum((all_predictions == 0) & (all_labels == 0))
    false_positive = np.sum((all_predictions == 1) & (all_labels == 0))
    false_negative = np.sum((all_predictions == 0) & (all_labels == 1))
    
    print('\n' + '='*80)
    print('CONFUSION MATRIX')
    print('='*80)
    print(f'True Positive (Correctly predicted Positive):  {true_positive}')
    print(f'True Negative (Correctly predicted Negative):  {true_negative}')
    print(f'False Positive (Incorrectly predicted Positive): {false_positive}')
    print(f'False Negative (Incorrectly predicted Negative): {false_negative}')
    
    # Calculate precision, recall, F1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print('\n' + '='*80)
    print('PERFORMANCE METRICS')
    print('='*80)
    print(f'Precision: {precision:.4f} ({precision*100:.2f}%)')
    print(f'Recall:    {recall:.4f} ({recall*100:.2f}%)')
    print(f'F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)')
    print('='*80)
    
    # Determine grade
    print('\n' + '='*80)
    print('ASSIGNMENT GRADING')
    print('='*80)
    if accuracy >= 80:
        print(f'✅ EXCELLENT! Accuracy {accuracy:.2f}% >= 80% - Full Credit (100%)')
    elif accuracy >= 75:
        print(f'✅ GOOD! Accuracy {accuracy:.2f}% >= 75% - Partial Credit (90%)')
    elif accuracy >= 70:
        print(f'✅ OK! Accuracy {accuracy:.2f}% >= 70% - Partial Credit (85%)')
    elif accuracy >= 65:
        print(f'⚠️  PASSING! Accuracy {accuracy:.2f}% >= 65% - Partial Credit (80%)')
    else:
        print(f'❌ LOW! Accuracy {accuracy:.2f}% < 65% - Needs Improvement')
    print('='*80)
    
    # Model configuration summary
    print('\n' + '='*80)
    print('MODEL CONFIGURATION')
    print('='*80)
    print(f'Architecture: LSTM')
    print(f'LSTM Layers: {n_layers}')
    print(f'Hidden Dimension: {hidden_dim}')
    print(f'Embedding Dimension: {embedding_dim} (GloVe 200d)')
    print(f'Batch Size: {Batch_size}')
    print(f'Epochs Trained: {checkpoint["epoch"] + 1}')
    print(f'Vocabulary Size: {vocab_size}')
    print(f'Sequence Length: {input_len}')
    print('='*80)

if __name__ == '__main__':
    main()
