import os
import time
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.augmentation import AdvancedAugmentation
from utils.ploting_loss import plot_model_comparison
from utils.plot_pca_tsne import plot_pre_training_visualization, plot_post_training_visualization

from .models import CNN, ResidualCNN, SimpleGRU, SimpleLSTM, SimpleTransformer, SimpleRNNAttention, MTSAN
from .dataset import load_deep_learning_data, TimeSeriesDataset
from .train import train_model
from .evaluate import evaluate_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your existing models (modify paths as needed)
# from models.gru import GRUClassifier
# from models.CNN import CNNClassifier
# from models.transformer import TransformerClassifier
# from models.lstm import LSTMClassifier
# from models.Rnn_with_attention import RNNAttentionClassifier

# For this demo, I'll include simplified versions of the models
# Replace these with your actual model imports
#fix the seed
def fix_seed(seed=42):
    """Fix random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_stage4_analysis():
    """Run Stage 4: Deep Learning Model Comparison"""

    # Configuration from previous stages
    train_dir = 'helper/train_voltage'
    val_dir = 'helper/validation_voltage'
    test_dir = 'helper/test_voltage'
    # train_dir = 'helper/train'
    # val_dir = 'helper/validation'
    # test_dir = 'helper/test'


    # Optimized parameters from Stages 1-3
    # optimal_sensors = ['voc2', 'co', 'eth', 'mq7','no2','temp', 'humidity']
    optimal_sensors = ['voc2', 'eth', 'co', 'no2', 'mq9', 'mq7','temp', 'humidity']
    # optimal_sensors = ['voc2', 'co', 'eth', 'mq7', 'no2', 'temp', 'humidity', 'mq135','mq3','mq9']
    len_sensors = len(optimal_sensors)
    optimal_window_size = 250
    optimal_stride = 250

    # Get file lists
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [f for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]

    print(f"Split sizes: Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")
    print(f"\n{'='*70}")
    print("STAGE 4: DEEP LEARNING MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Optimized configuration:")
    print(f"  Sensors: {optimal_sensors}")
    print(f"  Window size: {optimal_window_size}")
    print(f"  Stride: {optimal_stride}")
    print(f"  LogReg baseline: 37.50%")
    print(f"  Data: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")

    augmentation = AdvancedAugmentation(
        noise_std=0.005,
        time_warp_sigma=0.1,
        magnitude_warp_sigma=0.1
    )
    # Load data
    print(f"\nLoading optimized datasets...")
    X_train, y_train = load_deep_learning_data(train_dir, train_files, optimal_sensors, optimal_window_size, optimal_stride, is_train=True, augment_factor=1, augmentation=augmentation)
    X_val, y_val = load_deep_learning_data(val_dir, val_files, optimal_sensors, optimal_window_size, optimal_stride)
    X_test, y_test = load_deep_learning_data(test_dir, test_files, optimal_sensors, optimal_window_size, optimal_stride)

    print(f"Data shapes:")
    print(f"  Train: {X_train.shape}, Labels: {len(y_train)}")
    print(f"  Val: {X_val.shape}, Labels: {len(y_val)}")
    print(f"  Test: {X_test.shape}, Labels: {len(y_test)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"  Classes: {len(label_encoder.classes_)} - {label_encoder.classes_}")

    # Normalize data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train_normalized = X_train_flat.reshape(X_train.shape)

    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_flat = scaler.transform(X_val_flat)
    X_val_normalized = X_val_flat.reshape(X_val.shape)

    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_flat = scaler.transform(X_test_flat)
    X_test_normalized = X_test_flat.reshape(X_test.shape)


    # Visualize before training
    print(f"\nGenerating pre-training visualizations...")
    # plot_pre_training_visualization(X_train_normalized, y_train_encoded, label_encoder, "Train")
    # plot_pre_training_visualization(X_val_normalized, y_val_encoded, label_encoder, "Validation")
    plot_pre_training_visualization(X_test_normalized, y_test_encoded, label_encoder, "Test")
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_normalized, y_train_encoded)
    val_dataset = TimeSeriesDataset(X_val_normalized, y_val_encoded)
    test_dataset = TimeSeriesDataset(X_test_normalized, y_test_encoded)

    # Create data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Models to test
    models_config = {
        'CNN': CNN,
        # 'ResidualCNN': ResidualCNN,
        # 'GRU': SimpleGRU,
        # 'LSTM': SimpleLSTM,
        # 'RNN+Attention': SimpleRNNAttention,
        # 'Transformer': SimpleTransformer,
        # 'MTSAN': MTSAN
    }
    model_hyperparams = {
        'CNN': {
            'num_epochs': 500,
            'learning_rate': 0.00002,
            'dropout': 0.2
        },
        'ResidualCNN': {
            'num_epochs': 500,
            'learning_rate': 0.00002,
            'dropout': 0.2
        },
        'GRU': {
            'num_epochs': 300,
            'learning_rate': 0.00002,
            'hidden_size': 200,
            'num_layers': 2,
            'dropout': 0.4
        },
        'LSTM': {
            'num_epochs': 300,
            'learning_rate': 0.00002,
            'hidden_size': 200,
            'num_layers': 2,
            'dropout': 0.4
        },
        'RNN+Attention': {
            'num_epochs': 300,
            'learning_rate': 0.00002,
            'hidden_size': 200,
            'num_layers': 2,
            'dropout': 0.4
        },
        'Transformer': {
            'num_epochs': 300,
            'learning_rate': 0.00002,
            'hidden_size': 512,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.4
        },
        'MTSAN': {
            'num_epochs': 10,
            'learning_rate': 0.00002,
            'hidden_size': 200,
            'num_layers': 2,
            'dropout': 0.4
        }
    }

    input_size = len(optimal_sensors)
    num_classes = len(label_encoder.classes_)

    results = []
    all_models_data_for_plot = []
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Test each model
    for model_name, ModelClass in models_config.items():
        model_folder = os.path.join(results_folder, model_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)


        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")

        start_time = time.time()


        try:
            # Create model
            # if model_name == 'CNN' or model_name == 'ResidualCNN' :
            #     model = ModelClass(input_size=input_size, num_classes=num_classes, dropout=0.3)
            # else:
            #     model = ModelClass(input_size=input_size, num_classes=num_classes, hidden_size=128, num_layers=2, dropout=0.3)

            # Get hyperparameters for the current model
            params = model_hyperparams[model_name]
            num_epochs = params.get('num_epochs', 150)
            learning_rate = params.get('learning_rate', 0.0001)

            # Instantiate model with model-specific hyperparameters
            if model_name == 'CNN':
                model = ModelClass(input_size=input_size, num_classes=num_classes, dropout=params.get('dropout', 0.1))
            elif model_name == 'ResidualCNN':
                model = ModelClass(input_size=input_size, num_classes=num_classes, dropout=params.get('dropout', 0.4))
            elif model_name in ['GRU', 'LSTM']:
                model = ModelClass(input_size=input_size, num_classes=num_classes, hidden_size=params.get('hidden_size', 128),
                                 num_layers=params.get('num_layers', 2), dropout=params.get('dropout', 0.1))
            elif model_name == 'RNN+Attention':
                model = ModelClass(input_size=input_size, num_classes=num_classes, hidden_size=params.get('hidden_size', 128),
                                 num_layers=params.get('num_layers', 2), dropout=params.get('dropout', 0.1))
            elif model_name == 'Transformer':
                model = ModelClass(input_size=input_size, num_classes=num_classes, hidden_size=params.get('hidden_size', 128),
                                 num_layers=params.get('num_layers', 2), num_heads=params.get('num_heads', 4),
                                 dropout=params.get('dropout', 0.1))
            elif model_name == 'MTSAN':
                model = ModelClass(input_size=input_size, num_classes=num_classes, hidden_size=params.get('hidden_size', 128),
                                 num_layers=params.get('num_layers', 2), dropout=params.get('dropout', 0.4))

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Train model
            trained_model, best_val_acc, best_epoch, train_accs, val_accs, train_losses , val_losses = train_model(
                model, train_loader, val_loader, device,y_train_encoded, num_epochs=num_epochs, learning_rate=learning_rate, model_folder=model_folder
            )
            print(f"Generating post-training visualizations for {model_name}...")
            plot_post_training_visualization(trained_model, train_loader, device, label_encoder, model_name)
            # Evaluate on test set
            test_acc, test_f1, test_precision, test_recall = evaluate_model(
                trained_model, test_loader, device, label_encoder
            )

            # Append model data for the comparative plot
            all_models_data_for_plot.append({
                'name': model_name,
                'train_acc': train_accs,
                'val_acc': val_accs,
                'train_loss': train_losses,
                'val_loss': val_losses
            })

            result = {
                'model': model_name,
                'sensors': optimal_sensors,
                'window_size': optimal_window_size,
                'stride': optimal_stride,
                'val_accuracy': best_val_acc,
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'best_epoch': best_epoch,
                'final_train_acc': train_accs[best_epoch - 1] if best_epoch <= len(train_accs) else train_accs[-1],
                'n_parameters': sum(p.numel() for p in model.parameters()),
                'training_time': time.time() - start_time
            }

            results.append(result)

            print(f"\n{model_name} Results:")
            print(f"  Best Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            print(f"  Test Accuracy:     {test_acc:.4f}")
            print(f"  Test F1 Score:     {test_f1:.4f}")
            print(f"  Training Time:     {time.time() - start_time:.1f}s")

            # Cleanup
            del trained_model, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print("STAGE 4 FINAL RESULTS")
    print(f"{'='*70}")
    plot_model_comparison(all_models_data_for_plot, plot_filename='Final_Model_Comparison_Plots.png')
    if results:
        # Sort by test accuracy
        results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)

        print(f"{'Rank':<4} {'Model':<12} {'Val Acc':<8} {'Test Acc':<8} {'F1':<6} {'Epochs':<6} {'Time':<6}")
        print("-" * 70)

        logistic_baseline = 0.3750

        for i, result in enumerate(results_sorted, 1):
            improvement = result['test_accuracy'] - logistic_baseline
            print(f"{i:<4} {result['model']:<12} {result['val_accuracy']:<8.4f} {result['test_accuracy']:<8.4f} "
                  f"{result['test_f1']:<6.3f} {result['best_epoch']:<6} {result['training_time']:<6.0f}s")

        # Best model
        best_result = results_sorted[0]
        improvement = best_result['test_accuracy'] - logistic_baseline

        print(f"\n BEST MODEL: {best_result['model']}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
        # print(f"   Improvement over LogReg: {improvement:+.4f} ({improvement/logistic_baseline*100:+.1f}%)")
        print(f"   F1 Score: {best_result['test_f1']:.4f}")
        print(f"   Training Time: {best_result['training_time']:.1f}s")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'stage4_all_model_results_sensor_{len_sensors}.csv', index=False)
        print(f"\nResults saved to: stage4_deep_learning_results.csv")

        return results_sorted
    else:
        print("No results obtained!")
        return []


if __name__ == "__main__":
    # Fix random seed for reproducibility
    fix_seed(43)
    results = run_stage4_analysis()
