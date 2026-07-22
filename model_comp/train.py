import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.ploting_loss import plot_training_history


def train_model(model, train_loader, val_loader, device,y_train_encoded, num_epochs=150, learning_rate=0.007, model_folder='checkpoints'):
    """Train a deep learning model"""
    checkpoint_folder = 'checkpoints'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    model = model.to(device)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # # Use CrossEntropyLoss with weights
    # criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    # # Dynamic optimization with warmup
    # n_params = sum(p.numel() for p in model.parameters())
    # base_lr = learning_rate
    # warmup_epochs = 10
    # weight_decay = 1e-4 if n_params < 500000 else 5e-5  # Reduced weight decay
    # dropout_adjust = min(0.3, 0.2 + (n_params - 400000) / 1000000)  # Moderate dropout

    # optimizer = optim.Adam(model.parameters(), lr=base_lr / 100, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)




    # criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=1, gamma=1.5, weight=None)  # Use Focal Loss for class imbalance
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)#, patience=10)

    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    early_stop_counter = 0
    patience = 10

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training

        # if epoch < warmup_epochs:
        #     lr = base_lr * (epoch + 1) / warmup_epochs
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            # loss1 = criterion1(outputs, batch_labels)  # Use Focal Loss
            # loss = loss + loss1  # Combine losses
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

        train_acc = correct_train / total_train
        train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

        val_acc = correct_val / total_val
        val_loss = total_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_counter = 0
            # Save best model state
            model_path = os.path.join(checkpoint_folder, f'best_model_{model.__class__.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            # model_path = f'best_model_{model.__class__.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")


    #delete previous checkpoints except last 5
    all_checkpoints = sorted([f for f in os.listdir(checkpoint_folder) if f.startswith('best_model_')])
    for ckpt in all_checkpoints[:-5]:
        os.remove(os.path.join(checkpoint_folder, ckpt))
    # Load best model
    model.load_state_dict(best_model_state)

    plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses, model.__class__.__name__, model_folder)

    return model, best_val_acc, best_epoch, train_accuracies, val_accuracies, train_losses , val_losses
