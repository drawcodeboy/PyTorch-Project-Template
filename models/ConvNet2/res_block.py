from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_dim:int = 16,
                 out_dim:int = 32):
        
        super().__init__()
        
        # Convolution layer
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        
        # Normalization layer
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.bn3 = nn.BatchNorm2d(out_dim)
        
        # Activation funciton
        self.relu = nn.ReLU()
        
    def forward(self, x):        
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        
        identity = x.clone()
        
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        
        # Residual connection
        x = x + identity
        
        return x