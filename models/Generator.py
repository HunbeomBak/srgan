import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        k3n64s1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3//2)
        bn2d = nn.BatchNorm2d(num_features=64)
        
        self.block = nn.Sequential(
            k3n64s1,
            bn2d,
            nn.PReLU(),
            k3n64s1,
            bn2d
        )
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        return out
    
class Upsampler(nn.Module):
    def __init__(self):
        super(Upsampler, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
    def forward(self, x):
        x = self.upsample(x)
        
        return x
    

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        B = 16
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9//2),
            nn.PReLU()
        )
        
        block = [ResBlock() for _ in range(16)]
        self.block = nn.Sequential(*block)
        
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(64)
        )
        
        upsample = [Upsampler() for _ in range(3)]
        self.upsample = nn.Sequential(*upsample)
        
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=9//2)
        
        
    def forward(self,x):
        x =x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x_res = x ##Residual
        x = self.block(x)
        x = self.conv2(x)
        x += x_res
        x = self.upsample(x)
        x = self.conv3(x)
        
        return x
    
if __name__ == "__main__":
    G = Generator()
    input_x = torch.rand(1,32,32,3)
    print(input_x.shape)
    
    y = G(input_x)
    print(y.size())