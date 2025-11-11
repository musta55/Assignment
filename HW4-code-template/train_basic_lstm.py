#!/usr/bin/env python
# coding: utf-8
'''Basic LSTM without pre-trained embeddings - for baseline comparison'''

import torch
from torch import nn
import pandas as pd 
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from LSTM import LSTMModel
import time
import numpy as np

def main():
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    
    print('device: ', device)

    ## Basic LSTM settings (NO pre-trained embeddings)
    mode = 'train'
    Batch_size = 256
    n_layers = 2
    input_len = 150
    embedding_dim = 100  # Random initialization
    hidden_dim = 128
    output_size = 1
    num_epoches = 15
    learning_rate = 0.002
    clip = 5
    ckp_path = 'cpt/sentiment_model_basic.pt'
    pretrain = False  # <<< NO pre-trained embeddings
    
    print('='*70)
    print('BASIC LSTM (Random Embeddings - No Pre-training)')
    print('='*70)

    ## Load data
    training_set = MovieDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size,
                                    shuffle=True, num_workers=1)
    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size,
                                shuffle=False, num_workers=1)

    ## Load vocabulary
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)
    
    embedding_matrix = None  # No pre-trained embeddings

    ## Initialize model
    model = LSTMModel(vocab_size, output_size, embedding_dim, embedding_matrix,
                      hidden_dim, n_layers, input_len, pretrain)
    model.to(device)

    ## Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = nn.BCELoss()
    
    start_epoch = 0

    ## Training
    print('\nTraining Basic LSTM...\n')
    
    loss_history = []
    
    if mode == 'train':
        model.train()
        for epoches in range(start_epoch, num_epoches):
            epoch_loss = 0.0
            num_batches = 0
            for x_batch, y_labels in training_generator:
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                y_out = model(x_batch)
                loss = loss_fun(y_out, y_labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            print(f"Epoch [{epoches+1}/{num_epoches}], Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {'epoch': epoches,
                        'global_step': num_batches,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history}
            torch.save(checkpoint, ckp_path)
    
    ## Testing
    print("\n" + "="*70)
    print("TESTING BASIC LSTM")
    print("="*70)
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_labels in test_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            y_pred = torch.round(y_out)
            correct += (y_pred == y_labels).sum().item()
            total += y_labels.size(0)
            all_predictions.extend(y_pred.cpu().numpy())
            all_labels.extend(y_labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Confusion matrix
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save results
    results = {
        'embedding_type': 'Random (No Pre-training)',
        'embedding_dim': embedding_dim,
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'confusion_matrix': {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
        },
        'loss_history': loss_history,
        'hyperparameters': {
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'embedding_dim': embedding_dim,
            'batch_size': Batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epoches,
            'sequence_length': input_len
        }
    }
    
    with open('results_basic.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1_score*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"\nResults saved to: results_basic.json")

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"\nTraining time: {(time_end - time_start)/60.0:.2f} mins")
