import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamCNN_LSTM(nn.Module):
    def __init__(self, input_dim1, input_dim2, feature_dim=128, lstm_hidden_dim=128, num_classes=3):
        super(DualStreamCNN_LSTM, self).__init__()
        
        # Stream 1: process Landmarks
        self.stream1_conv1 = nn.Conv1d(in_channels=input_dim1, out_channels=64, kernel_size=3, padding=1)
        self.stream1_bn1 = nn.BatchNorm1d(64)
        self.stream1_pool = nn.MaxPool1d(kernel_size=2)
        
        self.stream1_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.stream1_bn2 = nn.BatchNorm1d(128)
        
        self.stream1_conv3 = nn.Conv1d(128, feature_dim, kernel_size=3, padding=1)
        self.stream1_bn3 = nn.BatchNorm1d(feature_dim)
        
        # Stream 2: process LBP
        self.stream2_conv1 = nn.Conv1d(in_channels=input_dim2, out_channels=64, kernel_size=3, padding=1)
        self.stream2_bn1 = nn.BatchNorm1d(64)
        self.stream2_pool = nn.MaxPool1d(kernel_size=2)
        
        self.stream2_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.stream2_bn2 = nn.BatchNorm1d(128)
        
        self.stream2_conv3 = nn.Conv1d(128, feature_dim, kernel_size=3, padding=1)
        self.stream2_bn3 = nn.BatchNorm1d(feature_dim)
        
        # merge the features withour maxpooling
        self.fusion_conv = nn.Conv1d(in_channels=feature_dim * 2, out_channels=feature_dim, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm1d(feature_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # full-connected layer
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x1, x2):
        # x1: (batch_size, seq_len, input_dim1)
        # x2: (batch_size, seq_len, input_dim2)
        
        # Stream 1
        x1 = x1.permute(0, 2, 1)  # (batch_size, input_dim1, seq_len)
        x1 = self.stream1_pool(F.relu(self.stream1_bn1(self.stream1_conv1(x1))))  # (batch_size, 64, seq_len/2)
        x1 = self.stream1_pool(F.relu(self.stream1_bn2(self.stream1_conv2(x1))))  # (batch_size, 128, seq_len/4)
        x1 = self.stream1_pool(F.relu(self.stream1_bn3(self.stream1_conv3(x1))))  # (batch_size, feature_dim, seq_len/8)
        
        # Stream 2
        x2 = x2.permute(0, 2, 1)  # (batch_size, input_dim2, seq_len)
        x2 = self.stream2_pool(F.relu(self.stream2_bn1(self.stream2_conv1(x2))))  # (batch_size, 64, seq_len/2)
        x2 = self.stream2_pool(F.relu(self.stream2_bn2(self.stream2_conv2(x2))))  # (batch_size, 128, seq_len/4)
        x2 = self.stream2_pool(F.relu(self.stream2_bn3(self.stream2_conv3(x2))))  # (batch_size, feature_dim, seq_len/8)
        
        # feature merged
        fused = torch.cat((x1, x2), dim=1)  # (batch_size, feature_dim * 2, seq_len/8)
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))  # (batch_size, feature_dim, seq_len/8)
        
        fused = fused.permute(0, 2, 1)  # (batch_size, seq_len/8, feature_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(fused)  # (batch_size, seq_len/8, lstm_hidden_dim)

        lstm_last = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)
        
        x = self.dropout(F.relu(self.fc(lstm_last)))  # (batch_size, num_classes)
        return x
