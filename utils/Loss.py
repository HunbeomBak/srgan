import torch
import torch.nn as nn
import torchvision.models as models

## Setting import from parent directory
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


class Content_loss(nn.Module):
    def __init__(self):
        super(Content_loss,self).__init__()
        
        vgg = models.vgg19(pretrained=True)
        #vgg.eval()
        self.feature = nn.Sequential(*list(vgg.features.children())[:-1])
        self.feature.eval()
        self.MSE = nn.MSELoss()
        
    def forward(self, HR, SR):
        HR_feature = self.feature(HR)
        SR_feature = self.feature(SR)
        loss  = self.MSE(HR_feature, SR_feature)
        return loss.sum()
    
class Adversarial_loss(nn.Module):
    def __init__(self):
        super(Adversarial_loss,self).__init__()
        #self.D = Discriminator()
        
    def forward(self, D_SR):
        loss=-torch.log10(D_SR)
        
        return loss.sum()

    
class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss,self).__init__()
        self.content_loss = Content_loss()
        self.adversarial_loss = Adversarial_loss()
    def forward(self, HR, SR, D_SR):
        loss = self.content_loss(HR, SR) + 10**-3 * self.adversarial_loss(D_SR)
        return loss

if __name__ == "__main__":
    from models.Generator import Generator
    from models.Discriminator import Discriminator
    G = Generator()
    D = Discriminator()
    
    HR = torch.rand(5,3,256,256)
    print(HR.shape)
    SR = G(torch.rand(5,32,32,3))
    print(SR.shape)
    
    p = Perceptual_loss()
    print(p(HR, SR, D(SR)))