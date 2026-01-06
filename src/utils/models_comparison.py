import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import model_0_results, model_1_results, model_2_results

def compare_models():
    # Model names and final test accuracies
    model_names = ['TinyVGG (10 units)', 'TinyVGG + Augmentation', 'ImprovedCNN (64 units)']
    final_test_accuracies = [
        model_0_results['test_acc'][-1],
        model_1_results['test_acc'][-1], 
        model_2_results['test_acc'][-1]
    ]
    
    # Create bar graph
    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_names, final_test_accuracies, 
                   color=['red', 'blue', 'green'], 
                   alpha=0.7)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{final_test_accuracies[i]:.3f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Model Comparison - Final Test Accuracy', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight the best model
    best_idx = final_test_accuracies.index(max(final_test_accuracies))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("=" * 50)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    for i, (name, acc) in enumerate(zip(model_names, final_test_accuracies)):
        status = " 🏆 BEST MODEL" if i == best_idx else ""
        print(f"{name}: {acc:.3f}{status}")
    print("=" * 50)

def plot_all_metrics():
    # Create comprehensive comparison
    model_0_df = pd.DataFrame(model_0_results)
    model_1_df = pd.DataFrame(model_1_results)
    model_2_df = pd.DataFrame(model_2_results)
    
    metrics = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
    final_values = {
        'TinyVGG': [model_0_df[metric].iloc[-1] for metric in metrics],
        'TinyVGG + Aug': [model_1_df[metric].iloc[-1] for metric in metrics],
        'ImprovedCNN': [model_2_df[metric].iloc[-1] for metric in metrics]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [final_values[model][i] for model in final_values.keys()]
        bars = axes[i].bar(final_values.keys(), values, 
                          color=['red', 'blue', 'green'], alpha=0.7)
        
        # Highlight best performance
        if 'acc' in metric:
            best_idx = values.index(max(values))
        else:  # loss metrics
            best_idx = values.index(min(values))
        
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        axes[i].set_title(f'Final {metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{values[j]:.3f}',
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_models()
    plot_all_metrics()
