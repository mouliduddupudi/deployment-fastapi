import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 61 * 61, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        #(3, 256, 256) ---> input
        x = self.conv1(x) 
        #(6, 252, 252) ---> output
        x = F.max_pool2d(x, kernel_size=2)
        #(6, 126, 126)
        x = F.relu(x)
        x = self.conv2(x)
        #(16, 122, 122)
        x = F.max_pool2d(x, kernel_size=2)
        #(16, 61, 61)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        h = x
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


idx_to_classes = {
    0: 'Atborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}