import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class sr_dataset(Dataset):
    def __init__(self, HR_path, LR_path):
        """
        HR_path : path to high resolution path
        LR_path : path to low resolution path
        
        """
        
        self.HR_list = [os.path.join(HR_path, i) for i in os.listdir(HR_path) if i.endswith('.png')]
        self.HR_list.sort()
        

        self.LR_list = [os.path.join(HR_path, i) for i in os.listdir(LR_path) if i.endswith('.png')]
        self.LR_list.sort()
    
        
    def __len__(self):
        return len(self.HR_list)
    
    def __getitem__(self, idx):
        HR = Image.open(self.HR_list[idx])
        LR = Image.open(self.LR_list[idx])
        
    
        return {'HR' : T.ToTensor()(HR), 
                'LR' : T.ToTensor()(LR)}
    
    
    
    
if __name__ == "__main__":
    HR = './data/CelebA-HQ_Dataset/Train/HQ_256x256/'
    LR = './data/CelebA-HQ_Dataset/Train/HQ_32x32/'
    ds = sr_dataset(HR_path=HR, LR_path=HR)
    
    print(len(ds))
    print(ds[0])
        
 