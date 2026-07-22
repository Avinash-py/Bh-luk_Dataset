import torch
import torch.nn as nn
import torch.nn.functional as F


class MTSAN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.4):
        super(MTSAN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-Scale Convolutional Block
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_size, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)

        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Softmax(dim=1)
        )

        # Bidirectional GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)

        # Temporal Attention Module
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Dynamic Fusion Layer
        self.fusion_gate = nn.Sequential(
            nn.Linear(64 * 3 + hidden_size * 2, 64 * 3 + hidden_size * 2),
            nn.Sigmoid()
        )

        # Output Layer
        self.fc = nn.Linear(64 * 3 + hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # Multi-Scale Convolutional Block
        x_conv = x.transpose(1, 2)  # (batch, input_size, seq_len)
        conv1 = self.pool(F.relu(self.bn1(self.conv1(x_conv))))
        conv2 = self.pool(F.relu(self.bn2(self.conv2(x_conv))))
        conv3 = self.pool(F.relu(self.bn3(self.conv3(x_conv))))

        # Concatenate and reshape for spatial attention
        conv_out = torch.cat((conv1, conv2, conv3), dim=1)  # (batch, 64*3, seq_len/2)
        conv_out_flat = conv_out.mean(dim=2)  # (batch, 64*3)
        spatial_attn = self.spatial_attention(conv_out_flat)  # (batch, input_size)
        spatial_attn = spatial_attn.unsqueeze(1)  # (batch, 1, input_size)
        x_weighted = x * spatial_attn  # (batch, seq_len, input_size)

        # Bidirectional GRU
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        gru_out, _ = self.gru(x_weighted, h0)  # (batch, seq_len, hidden_size*2)

        # Temporal Attention
        temp_attn = self.temporal_attention(gru_out)  # (batch, seq_len, 1)
        gru_weighted = torch.sum(gru_out * temp_attn, dim=1)  # (batch, hidden_size*2)

        # Dynamic Fusion
        conv_features = conv_out_flat  # (batch, 64*3)
        gru_features = gru_weighted  # (batch, hidden_size*2)
        combined_features = torch.cat((conv_features, gru_features), dim=1)  # (batch, 64*3 + hidden_size*2)
        gate = self.fusion_gate(combined_features)  # (batch, 64*3 + hidden_size*2)
        fused_features = gate * combined_features  # (batch, 64*3 + hidden_size*2)

        # Output
        out = self.dropout(fused_features)
        out = self.fc(out)
        return out


class ResidualCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.4):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv_match = nn.Conv1d(input_size, 128, kernel_size=1)  # Matches conv2 output channels
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
        identity = x

        # First block with residual connection
        x = torch.relu(self.conv1(x))  # (batch, 64, seq_len)
        x = self.pool(x)  # (batch, 64, seq_len/2)
        x = torch.relu(self.conv2(x))  # (batch, 128, seq_len/2)
        identity = self.pool(self.conv_match(identity))  # (batch, 128, seq_len/2)

        x = x + identity  # Residual connection (now shapes match: batch, 128, seq_len/2)
        x = torch.relu(x)
        # x = torch.relu(self.conv3(x))  # (batch, 256, seq_len/4)
        x = self.adaptive_pool(x)  # (batch, 256, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        # x = self.pool(x)
        # x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SimpleGRU(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.1):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
        # out, _ = self.gru(x, h0)
        # out = out[:, -1, :]  # Take last timestep
        # out = self.dropout(out)  # Apply dropout here
        # out = self.fc(out)
        # return out


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
        # out = out[:, -1, :]
        # out = self.dropout(out)
        # out = self.fc(out)
        # return out


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, num_heads=4, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out


class SimpleRNNAttention(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.1):
        super(SimpleRNNAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)

        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)

        context = self.dropout(context)
        output = self.fc(context)
        return output
