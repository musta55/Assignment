#!/usr/bin/env python
# coding: utf-8
'''Generate comprehensive visualizations for the report'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

# Set style
sns.set_palette("husl")

# Load results
with open('results_100d.json', 'r') as f:
    results_100d = json.load(f)

with open('results_200d.json', 'r') as f:
    results_200d = json.load(f)

print("Creating comprehensive visualizations...")

# Create main figure with subplots
fig = plt.figure(figsize=(18, 12))

# ============================================================
# 1. Training Loss Curves Comparison
# ============================================================
ax1 = plt.subplot(2, 3, 1)
epochs = list(range(1, 16))
ax1.plot(epochs, results_100d['loss_history'], 'b-o', linewidth=2.5, 
         markersize=7, label='Embedding Dim = 100', alpha=0.8)
ax1.plot(epochs, results_200d['loss_history'], 'r-s', linewidth=2.5, 
         markersize=7, label='Embedding Dim = 200', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Loss (BCE)', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss Curves Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# ============================================================
# 2. Confusion Matrix - 100d
# ============================================================
ax2 = plt.subplot(2, 3, 2)
cm_100d = results_100d['confusion_matrix']
conf_matrix_100d = np.array([[cm_100d['TN'], cm_100d['FP']], 
                              [cm_100d['FN'], cm_100d['TP']]])
sns.heatmap(conf_matrix_100d, annot=True, fmt='d', cmap='Blues', 
            cbar=True, ax=ax2, annot_kws={"size": 14, "weight": "bold"})
ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax2.set_title('Confusion Matrix (Embedding Dim = 100)', fontsize=14, fontweight='bold')
ax2.set_xticklabels(['Negative (0)', 'Positive (1)'])
ax2.set_yticklabels(['Negative (0)', 'Positive (1)'])

# ============================================================
# 3. Confusion Matrix - 200d
# ============================================================
ax3 = plt.subplot(2, 3, 3)
cm_200d = results_200d['confusion_matrix']
conf_matrix_200d = np.array([[cm_200d['TN'], cm_200d['FP']], 
                              [cm_200d['FN'], cm_200d['TP']]])
sns.heatmap(conf_matrix_200d, annot=True, fmt='d', cmap='Oranges', 
            cbar=True, ax=ax3, annot_kws={"size": 14, "weight": "bold"})
ax3.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax3.set_title('Confusion Matrix (Embedding Dim = 200)', fontsize=14, fontweight='bold')
ax3.set_xticklabels(['Negative (0)', 'Positive (1)'])
ax3.set_yticklabels(['Negative (0)', 'Positive (1)'])

# ============================================================
# 4. Performance Metrics Comparison
# ============================================================
ax4 = plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_100d = [results_100d['accuracy'], results_100d['precision'], 
               results_100d['recall'], results_100d['f1_score']]
values_200d = [results_200d['accuracy'], results_200d['precision'], 
               results_200d['recall'], results_200d['f1_score']]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, values_100d, width, label='Embedding Dim = 100', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, values_200d, width, label='Embedding Dim = 200', 
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=11)
ax4.legend(fontsize=10)
ax4.set_ylim([75, 85])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# 5. Loss Reduction Analysis
# ============================================================
ax5 = plt.subplot(2, 3, 5)
initial_loss_100d = results_100d['loss_history'][0]
final_loss_100d = results_100d['loss_history'][-1]
reduction_100d = ((initial_loss_100d - final_loss_100d) / initial_loss_100d) * 100

initial_loss_200d = results_200d['loss_history'][0]
final_loss_200d = results_200d['loss_history'][-1]
reduction_200d = ((initial_loss_200d - final_loss_200d) / initial_loss_200d) * 100

categories = ['Initial Loss', 'Final Loss', 'Reduction (%)']
values_100d_loss = [initial_loss_100d, final_loss_100d, reduction_100d/100]
values_200d_loss = [initial_loss_200d, final_loss_200d, reduction_200d/100]

x_loss = np.arange(len(categories))
bars1 = ax5.bar(x_loss - width/2, values_100d_loss, width, label='Embedding Dim = 100', 
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x_loss + width/2, values_200d_loss, width, label='Embedding Dim = 200', 
                color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax5.set_title('Loss Reduction Analysis', fontsize=14, fontweight='bold')
ax5.set_xticks(x_loss)
ax5.set_xticklabels(categories, fontsize=11)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 2:  # Reduction percentage
            label = f'{height*100:.1f}%'
        else:
            label = f'{height:.4f}'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# 6. Model Configuration Summary
# ============================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

config_text = f"""
"""

ax6.text(0.05, 0.5, config_text, fontsize=9.5, family='monospace',
         verticalalignment='center', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1.5))

plt.suptitle('LSTM Sentiment Analysis - Comprehensive Experimental Results\nComparison: Embedding Dimension 100 vs 200', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.01, 1, 0.99])
plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comprehensive_results.png")

# ============================================================
# Create separate detailed loss curve
# ============================================================
plt.figure(figsize=(14, 8))
plt.plot(epochs, results_100d['loss_history'], 'b-o', linewidth=3, 
         markersize=10, label='Embedding Dim = 100', alpha=0.8)
plt.plot(epochs, results_200d['loss_history'], 'r-s', linewidth=3, 
         markersize=10, label='Embedding Dim = 200', alpha=0.8)
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Training Loss (Binary Cross Entropy)', fontsize=16, fontweight='bold')
plt.title('Training Loss Curves: Embedding Dimension Comparison\n2-Layer LSTM with GloVe Pre-trained Embeddings', 
          fontsize=18, fontweight='bold', pad=20)
plt.legend(fontsize=14, loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
plt.xticks(epochs, fontsize=13)
plt.yticks(fontsize=13)

# Add annotations for initial and final loss
for results, color, offset in [(results_100d, 'blue', -0.03), (results_200d, 'red', 0.03)]:
    initial = results['loss_history'][0]
    final = results['loss_history'][-1]
    plt.annotate(f'Initial: {initial:.4f}', xy=(1, initial), xytext=(3, initial + offset),
                fontsize=11, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    plt.annotate(f'Final: {final:.4f}', xy=(15, final), xytext=(13, final + offset),
                fontsize=11, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

plt.tight_layout()
plt.savefig('loss_curves_detailed.png', dpi=300, bbox_inches='tight')
print("✅ Saved: loss_curves_detailed.png")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("Generated files:")
print("  1. comprehensive_results.png - Complete dashboard (6 panels)")
print("  2. loss_curves_detailed.png - Detailed loss comparison")
print("="*70)