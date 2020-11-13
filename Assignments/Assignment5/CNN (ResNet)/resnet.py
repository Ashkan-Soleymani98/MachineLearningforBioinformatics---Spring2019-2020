import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, has_batchnorm=True, dropout_probs=[0.3, 0.2]):
        super(ResidualBlock, self).__init__()

        self.has_batchnorm = has_batchnorm        
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_probs[0])
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=kernel//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_probs[1])
        
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        if self.has_batchnorm:
            x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.has_batchnorm:
            x = self.bn2(x)
        x = self.bn2(x)
        
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        
        x += identity
        x = self.dropout2(x)
        x = self.relu(x)
        
        return x    


class ResNet(nn.Module):
    def __init__(self
                 , numbers_of_residual_block=3
                 , image_channels=1
                 , num_classes=2
                 , block_has_batchnorm=True
                 , fc_has_batchnorm=True
                 , block_dropout_probs=[0.3, 0.2]
                 , fc_dropout_prob=0.3):
        super(ResNet, self).__init__()
        
        cov1_kernel = 7
        cov1_out_channels = 16
        cov1_stride = 2
        self.cov1 = nn.Conv2d(image_channels, cov1_out_channels, kernel_size=cov1_kernel, stride=cov1_stride, padding=cov1_kernel//2)
        self.bn1 = nn.BatchNorm2d(cov1_out_channels)
        self.relu = nn.ReLU()
        
        pool1_kernel = 3
        pool1_stride = 2
        self.maxpool = nn.MaxPool2d(kernel_size=pool1_kernel, stride=pool1_stride, padding=pool1_kernel//2)
        
        self.blocks = list()
        
        temp_input_channels = cov1_out_channels
        channel_expansion_factor = 4
        for i in range(numbers_of_residual_block):
            residual_block = ResidualBlock(temp_input_channels
                                           , channel_expansion_factor * temp_input_channels
                                           , kernel=7 - 2*i 
                                           , stride=2
                                           , has_batchnorm=block_has_batchnorm
                                           , dropout_probs=block_dropout_probs)
            temp_input_channels = channel_expansion_factor * temp_input_channels
            self.blocks.append(residual_block)
            
        self.residual_part = nn.Sequential(*self.blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        fc_hidden_size = 512
        self.fc1 = nn.Linear(temp_input_channels, fc_hidden_size)
        self.dropout = nn.Dropout(fc_dropout_prob)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        
        self.fc_has_batchnorm = fc_has_batchnorm
        
    def forward(self, x):
        x = self.cov1(x)
        if self.fc_has_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.residual_part(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
            
        return x    