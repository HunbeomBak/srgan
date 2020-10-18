import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, k, in_n, out_n, s):
        super(ConvBlock, self).__init__()
        
        self.Conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_n, out_channels=out_n, kernel_size=k, stride=s, padding=k//2),
            nn.BatchNorm2d(out_n),
            nn.LeakyReLU())
        
    def forward(self, x):
        out = self.Conv_block(x)
        return out
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        
        conv = []
        conv.append(ConvBlock(k=3, in_n=64, out_n=64, s=2))
        conv.append(ConvBlock(k=3, in_n=64, out_n=128, s=1))
        conv.append(ConvBlock(k=3, in_n=128, out_n=128, s=2))
        conv.append(ConvBlock(k=3, in_n=128, out_n=256, s=1))
        conv.append(ConvBlock(k=3, in_n=256, out_n=256, s=2))
        conv.append(ConvBlock(k=3, in_n=256, out_n=512, s=1))
        conv.append(ConvBlock(k=3, in_n=512, out_n=512, s=2))
    
        self.conv2 = nn.Sequential(*conv)
        
        
        patch_size = 256
        num_of_block = 3
        n_feats = 64

        self.linear_size=((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))
        
        self.classfier = nn.Sequential(
            nn.Linear(self.linear_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(-1,self.linear_size)
        out = self.classfier(out)
        
        return out
    
    
    
if __name__ == "__main__":
    D = Discriminator()
    x = torch.rand(1,3,256,256)
    print(x.size())
    y =D(x)
    print(y.size())