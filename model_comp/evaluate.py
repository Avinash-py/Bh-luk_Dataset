import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from .models import CNN, ResidualCNN, SimpleGRU, SimpleLSTM, SimpleRNNAttention, SimpleTransformer, MTSAN


def evaluate_model(model, test_loader, device, label_encoder):
    """Evaluate model on test set"""
    embedding_folder = 'embeddings'
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
    model.eval()
    all_predictions = []
    all_labels = []

    all_embeddings = []
    def get_embedding_hook(module, input, output):
        all_embeddings.append(output.detach().cpu().numpy())

    # Attach hook to the layer before the final fc (e.g., after adaptive_pool for CNN)
    if isinstance(model, CNN) or isinstance(model, ResidualCNN):
        hook = model.adaptive_pool.register_forward_hook(get_embedding_hook)
    elif isinstance(model, (SimpleGRU, SimpleLSTM, SimpleRNNAttention, SimpleTransformer, MTSAN)):
        hook = model.fc.register_forward_pre_hook(get_embedding_hook)

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Remove hook
    hook.remove()

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else None
    labels = np.array(all_labels)

    embed_path = os.path.join(embedding_folder, f'test_embeddings_{model.__class__.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez(embed_path, embeddings=embeddings, labels=labels)
    print(f"Saved test embeddings to {embed_path}")

    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_predictions)
    test_f1 = f1_score(all_labels, all_predictions, average='macro')
    test_precision = precision_score(all_labels, all_predictions, average='macro')
    test_recall = recall_score(all_labels, all_predictions, average='macro')
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    # Compute confusion matrix and normalize by true labels
    cm = confusion_matrix(all_labels, all_predictions)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_normalized = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype(int)  # Row-wise normalization to percentages

    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                annot_kws={"size": 12}
                )
    plt.title(f'Confusion Matrix for {model.__class__.__name__}' , fontsize=28)
    # plt.xlabel('Predicted Label', fontsize=16)
    # plt.ylabel('True Label' , fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model.__class__.__name__}.png')
    plt.close()
    return test_acc, test_f1, test_precision, test_recall
