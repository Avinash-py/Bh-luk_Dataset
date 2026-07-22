import os
from matplotlib import pyplot as plt


def plot_training_history(train_acc, val_acc, train_loss, val_loss, model_name, model_folder='results'):
    """Plots and saves the training and validation loss and accuracy.

    Ensures the output folder exists before attempting to save the figure.

    """
    os.makedirs(model_folder, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    epochs = range(1, len(train_acc) + 1)
    
    # Plot Accuracy
    ax1.plot(epochs, train_acc, 'bo-', label='Train Accuracy')
    ax1.plot(epochs, val_acc , 'rs--', label='Validation Accuracy')
    # ax1.set_title(f'{model_name} - Model Accuracy', fontsize=17)
    ax1.set_ylabel('Accuracy', fontsize=28)
    ax1.set_xlabel('Epoch', fontsize=28)
    ax1.legend(loc='best', fontsize=25)
    ax1.set_ylim(0, 1)  # Accuracy between 0 and 1
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.spines['bottom'].set_color('black')  # Darker x-axis line
    ax1.spines['left'].set_color('black')    # Darker y-axis line
    ax1.spines['bottom'].set_linewidth(4)  # Thicker x-axis line
    ax1.spines['left'].set_linewidth(4)
    
    # Plot Loss
    ax2.plot(epochs, train_loss, 'bo-', label='Train Loss')
    ax2.plot(epochs, val_loss, 'rs--', label='Validation Loss')
    # ax2.set_title(f'{model_name} - Model Loss', fontsize=17)
    ax2.set_ylabel('Loss', fontsize=28)
    ax2.set_xlabel('Epoch', fontsize=28)
    ax2.legend(loc='best', fontsize=25)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_ylim(1, max(max(train_loss), max(val_loss)) + 0.5)  # Dynamic upper limit
    # ax2.set_yticks([i * 0.5 for i in range(int(ax2.get_ylim()[1] / 0.5) + 1)])
    ax2.tick_params(axis='both', labelsize=18)
    ax2.spines['bottom'].set_color('black')  # Darker x-axis line
    ax2.spines['left'].set_color('black')    # Darker y-axis line
    ax2.spines['bottom'].set_linewidth(4)  # Thicker x-axis line
    ax2.spines['left'].set_linewidth(4)
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(model_folder, exist_ok=True)
    plot_filename = os.path.join(model_folder, f"{model_name.replace(' ', '_').replace('+', 'and')}_training_plots.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved training plots to {plot_filename}")
    plt.close(fig) 
    
def plot_model_comparison(models_data, plot_filename='model_comparison.png'):
    """
    Plots a comparative graph for training and validation history of multiple models.
    
    Args:
        models_data (list of dicts): A list where each dictionary contains a model's
                                     training history. Each dictionary should have these keys:
                                     - 'name': str, model name (e.g., 'GRU')
                                     - 'train_acc': list, training accuracy list
                                     - 'val_acc': list, validation accuracy list
                                     - 'train_loss': list, training loss list
                                     - 'val_loss': list, validation loss list
        plot_filename (str): Filename to save the plot.
    """
    
    if not models_data:
        print("Error: No model data provided for plotting.")
        return


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    ax1.set_title('Model Accuracy Comparison', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylim(0, 1.05)
    
    ax2.set_title('Model Loss Comparison', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=14)
    
  
    for i, model_data in enumerate(models_data):
        model_name = model_data['name']
        train_acc = model_data['train_acc']
        val_acc = model_data['val_acc']
        train_loss = model_data['train_loss']
        val_loss = model_data['val_loss']
        
        epochs = range(1, len(train_acc) + 1)

        color = colors[i % len(colors)]

        ax1.plot(epochs, train_acc, color=color, linestyle='-', marker='o', markersize=4, 
                 label=f'{model_name} Train Accuracy')
        ax1.plot(epochs, val_acc, color=color, linestyle='--', marker='x', markersize=4, 
                 label=f'{model_name} Validation Accuracy')
        

        ax2.plot(epochs, train_loss, color=color, linestyle='-', marker='o', markersize=4, 
                 label=f'{model_name} Train Loss')
        ax2.plot(epochs, val_loss, color=color, linestyle='--', marker='x', markersize=4, 
                 label=f'{model_name} Validation Loss')

    # Final plot adjustments
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"comparision plot {plot_filename} stored successfully.")
    plt.close(fig)
