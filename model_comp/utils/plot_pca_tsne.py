import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

def plot_pre_training_visualization(X_data, y_encoded, label_encoder, dataset_type):
    """Plot PCA and t-SNE before training with class name legend"""
    X_flat = np.mean(X_data, axis=1)  # Aggregate time series to [n_samples, 6]
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='tab20')  # tab20 supports up to 20 distinct colors
    # plt.title(f'PCA of {dataset_type} Data', fontsize =28)
    plt.xlabel('Principal component 1', fontsize=24)
    plt.ylabel('Principal Component 2', fontsize =24)
    plt.tick_params(axis='both', labelsize=20)
    # Create legend with class names
    legend_elements = [Line2D([0], [0], marker='o', color=plt.cm.tab20(i / len(label_encoder.classes_)), 
                             label=label, markerfacecolor=plt.cm.tab20(i / len(label_encoder.classes_)), 
                             markersize=10) for i, label in enumerate(label_encoder.classes_)]
    plt.legend(handles=legend_elements, title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, title_fontsize=10,
               frameon=True, edgecolor='black', facecolor='white')
    plt.tight_layout()
    plt.savefig(f'pca_{dataset_type}_before.png', dpi=300)
    plt.close()
    print(f"PCA Explained Variance Ratio ({dataset_type}): {pca.explained_variance_ratio_}")
    
    # t-SNE
    perplexity = min(30, max(5, X_flat.shape[0] // 3))
    # tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50 , n_iter=2000)
    X_tsne = tsne.fit_transform(X_flat)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap='tab20')
    # plt.title(f't-SNE of {dataset_type} Data', fontsize =28)
    plt.xlabel('t-SNE Component 1' , fontsize=28)
    plt.ylabel('t-SNE Component 2' , fontsize=28)
    plt.tick_params(axis='both', labelsize=20)
    # Create legend with class names
    legend_elements = [Line2D([0], [0], marker='o', color=plt.cm.tab20(i / len(label_encoder.classes_)), 
                             label=label, markerfacecolor=plt.cm.tab20(i / len(label_encoder.classes_)), 
                             markersize=10) for i, label in enumerate(label_encoder.classes_)]
    plt.legend(handles=legend_elements, title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, title_fontsize=10,
               frameon=True, edgecolor='black', facecolor='white')
    plt.tight_layout()
    plt.savefig(f'tsne_{dataset_type}_before.png',dpi =300)
    plt.close()

def plot_post_training_visualization(model, data_loader, device, label_encoder, model_name):
    """Plot PCA and t-SNE after training using penultimate layer features.
    
    Args:
        model: nn.Module, one of ResidualCNN, SimpleCNN, SimpleGRU, SimpleLSTM, SimpleRNNAttention, MTSAN
        data_loader: torch.utils.data.DataLoader, yields (batch_data, batch_labels)
        device: torch.device, e.g., 'cuda' or 'cpu'
        label_encoder: sklearn.preprocessing.LabelEncoder with 20 classes
        model_name: str, one of 'ResidualCNN', 'CNN', 'GRU', 'LSTM', 'RNN+Attention', 'MTSAN'
    """
    if model_name not in ['ResidualCNN', 'CNN', 'GRU', 'LSTM', 'RNN+Attention', 'MTSAN']:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            if batch_data.shape[1:] != (250, 8):
                raise ValueError(f"Expected batch_data shape (batch_size, 300, 8), got {batch_data.shape}")
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if model_name == 'ResidualCNN':
                x = batch_data.transpose(1, 2)
                x = torch.relu(model.conv1(x))
                x = model.pool(x)
                x = torch.relu(model.conv2(x))
                identity = model.pool(model.conv_match(batch_data.transpose(1, 2)))
                x = x + identity
                x = torch.relu(x)
                x = model.adaptive_pool(x)
                x = x.view(x.size(0), -1)
            elif model_name == 'CNN':
                x = batch_data.transpose(1, 2)
                x = torch.relu(model.conv1(x))
                x = model.pool(x)
                x = torch.relu(model.conv2(x))
                x = model.adaptive_pool(x)
                x = x.view(x.size(0), -1)
            elif model_name == 'GRU':
                h0 = torch.zeros(model.num_layers, batch_data.size(0), model.hidden_size).to(device)
                out, _ = model.gru(batch_data, h0)
                x = out[:, -1, :]
            elif model_name == 'LSTM':
                h0 = torch.zeros(model.num_layers, batch_data.size(0), model.hidden_size).to(device)
                c0 = torch.zeros(model.num_layers, batch_data.size(0), model.hidden_size).to(device)
                out, _ = model.lstm(batch_data, (h0, c0))
                x = out[:, -1, :]
            elif model_name == 'RNN+Attention':
                h0 = torch.zeros(model.num_layers, batch_data.size(0), model.hidden_size).to(device)
                out, _ = model.rnn(batch_data, h0)
                attn_weights = torch.softmax(model.attention(out), dim=1)
                x = torch.sum(attn_weights * out, dim=1)
            elif model_name == 'MTSAN':
                x_conv = batch_data.transpose(1, 2)
                conv1 = model.pool(torch.relu(model.bn1(model.conv1(x_conv))))
                conv2 = model.pool(torch.relu(model.bn2(model.conv2(x_conv))))
                conv3 = model.pool(torch.relu(model.bn3(model.conv3(x_conv))))
                conv_out = torch.cat((conv1, conv2, conv3), dim=1)
                conv_out_flat = conv_out.mean(dim=2)
                spatial_attn = model.spatial_attention(conv_out_flat)
                spatial_attn = spatial_attn.unsqueeze(1)
                x_weighted = batch_data * spatial_attn
                h0 = torch.zeros(2 * model.num_layers, batch_data.size(0), model.hidden_size).to(device)
                gru_out, _ = model.gru(x_weighted, h0)
                temp_attn = model.temporal_attention(gru_out)
                gru_weighted = torch.sum(gru_out * temp_attn, dim=1)
                conv_features = conv_out_flat
                gru_features = gru_weighted
                combined_features = torch.cat((conv_features, gru_features), dim=1)
                x = model.fusion_gate(combined_features) * combined_features  # Penultimate layer
            else:
                raise NotImplementedError(f"Feature extraction for {model_name} not implemented")
            
            features.append(x.cpu().numpy())
            labels.append(batch_labels.cpu().numpy().astype(np.int32))
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch: features {features.shape[0]} vs labels {labels.shape[0]}")
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20')
    # plt.title(f'PCA of {model_name} Features ({data_loader.dataset.__class__.__name__})', fontsize =28)
    plt.xlabel('Principal Component 1', fontsize=24)
    plt.ylabel('Principal Component 2', fontsize =24)
    plt.tick_params(axis='both', labelsize=20)
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             label=label, markerfacecolor=plt.cm.tab20(i/20), 
                             markersize=10) for i, label in enumerate(label_encoder.classes_)]
    plt.legend(handles=legend_elements, title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, title_fontsize=10,
               frameon=True, edgecolor='black', facecolor='white')
    plt.tight_layout()
    plt.savefig(f'pca_{model_name}_after.png')
    plt.close()
    print(f"PCA Explained Variance Ratio ({model_name}): {pca.explained_variance_ratio_}")
    
    # t-SNE
    perplexity = min(30, max(5, features.shape[0] // 3))
    # tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50 , n_iter=2000)
    X_tsne = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab20')
    # plt.title(f't-SNE of {model_name} Features ({data_loader.dataset.__class__.__name__})' , fontsize =25)
    plt.xlabel('t-SNE Component 1' , fontsize=28)
    plt.ylabel('t-SNE Component 2'  , fontsize=28)
    plt.tick_params(axis='both', labelsize=20)
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             label=label, markerfacecolor=plt.cm.tab20(i/20), 
                             markersize=10) for i, label in enumerate(label_encoder.classes_)]
    plt.legend(handles=legend_elements, title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, title_fontsize=10,
               frameon=True, edgecolor='black', facecolor='white')
    plt.tight_layout()
    plt.savefig(f'tsne_{model_name}_after.png')
    plt.close()

# def plot_post_training_visualization(model, data_loader, device, label_encoder, model_name):
#     """Plot PCA and t-SNE after training using penultimate layer features with class name legend"""
#     model.eval()
#     features = []
#     labels = []
    
#     with torch.no_grad():
#         for batch_data, batch_labels in data_loader:
#             batch_data = batch_data.to(device)
#             # Model-specific feature extraction
#             if model_name in ['ResidualCNN', 'CNN']:
#                 x = batch_data.transpose(1, 2)  # [batch, seq, features] -> [batch, features, seq]
#                 if model_name == 'ResidualCNN':
#                     x = torch.relu(model.conv1(x))
#                     x = model.pool(x)
#                     x = torch.relu(model.conv2(x))
#                     identity = model.pool(model.conv_match(batch_data.transpose(1, 2)))  # Apply to original input
#                     x = x + identity
#                     x = torch.relu(model.conv3(x))
#                 else:  # SimpleCNN
#                     x = torch.relu(model.conv1(x))
#                     x = model.pool(x)
#                     x = torch.relu(model.conv2(x))
#                     x = model.pool(x)
#                 x = model.adaptive_pool(x)
#                 x = x.view(x.size(0), -1)  # [batch, 64] or [batch, 128]
#             elif model_name in ['GRU', 'LSTM', 'RNN+Attention']:
#                 h0 = torch.zeros(model.num_layers, batch_data.size(0), model.hidden_size).to(device)
#                 if model_name == 'LSTM':
#                     c0 = torch.zeros(model.num_layers, batch_data.size(0), model.hidden_size).to(device)
#                     out, _ = model.lstm(batch_data, (h0, c0))
#                 elif model_name == 'RNN+Attention':
#                     out, _ = model.rnn(batch_data, h0)
#                     attn_weights = torch.softmax(model.attention(out), dim=1)  # [batch_size, seq_len, 1]
#                     context = torch.sum(attn_weights * out, dim=1)  # [batch_size, hidden_size]
#                     x = context
#                 else:  # GRU
#                     out, _ = model.gru(batch_data, h0)
#                 x = out[:, -1, :]  # Take last timestep
#             elif model_name == 'Transformer':
#                 x = model.embedding(batch_data)
#                 out = model.transformer(x)
#                 x = out[:, -1, :]  # Take last timestep
#             else:
#                 raise NotImplementedError(f"Feature extraction for {model_name} not implemented")
            
#             features.append(x.cpu().numpy())
#             labels.append(batch_labels.numpy())
    
#     features = np.concatenate(features, axis=0)
#     labels = np.concatenate(labels, axis=0)
    
#     # PCA
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(features)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20')
#     plt.title(f'PCA of Training Data (After {model_name} Training)')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     legend_elements = [Line2D([0], [0], marker='o', color=plt.cm.tab20(i / len(label_encoder.classes_)), 
#                              label=label, markerfacecolor=plt.cm.tab20(i / len(label_encoder.classes_)), 
#                              markersize=5) for i, label in enumerate(label_encoder.classes_)]
#     plt.legend(handles=legend_elements, title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(f'pca_training_after_{model_name}.png')
#     plt.close()
#     print(f"PCA Explained Variance Ratio (After {model_name}): {pca.explained_variance_ratio_}")
    
#     # t-SNE
#     perplexity = min(30, max(5, features.shape[0] // 3))
#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
#     # tsne = TSNE(n_components=2, random_state=42)
#     X_tsne = tsne.fit_transform(features)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab20')
#     plt.title(f't-SNE of Training Data (After {model_name} Training)')
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     legend_elements = [Line2D([0], [0], marker='o', color=plt.cm.tab20(i / len(label_encoder.classes_)), 
#                              label=label, markerfacecolor=plt.cm.tab20(i / len(label_encoder.classes_)), 
#                              markersize=5) for i, label in enumerate(label_encoder.classes_)]
#     plt.legend(handles=legend_elements, title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(f'tsne_training_after_{model_name}.png')
#     plt.close()