import torch.nn as nn


class CovFusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=in_channels // 2,
                               kernel_size=kernel_size,
                               stride=stride)
        self.max_pool1 = nn.MaxPool1d(kernel_size=kernel_size,
                                      stride=stride)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(in_channels=in_channels // 2,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.max_pool2 = nn.MaxPool1d(kernel_size=kernel_size,
                                      stride=stride)
        self.conv3 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.max_pool3 = nn.MaxPool1d(kernel_size=kernel_size,
                                      stride=stride)

    def forward(self, x):
        # x shape:[batch, feature_num + target_num, seq_len]
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.drop(x)
        # x = self.conv4(x)
        # x = self.max_pool4(x)
        return x
