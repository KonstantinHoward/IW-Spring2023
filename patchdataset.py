import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class PatchDataset(Dataset):
    # @annotations_file : path to the Slide ID, necrosis percent csv 
    # @img_dir : path to the directory where patches are
    # @transfrom : inherited
    # @target_transform : inherited
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
       # self.normalize = normalize
        self.patch_IDs = os.listdir(self.img_dir)
        if ".DS_Store" in self.patch_IDs : self.patch_IDs.remove(".DS_Store")
        
        
        self.slide_map = {}
        
        for idx in self.img_labels.index :
            self.slide_map[self.img_labels['Slide_ID'][idx]] = self.img_labels["Necrosis_Percent"][idx]


    # return num patches NOT num slides
    def __len__(self):
        return len(self.patch_IDs)

    # @idx : index of a patch
    # return patch image for requested patch ID
    def __getitem__(self, idx):
        patch = self.patch_IDs[idx]
        slide_ID = patch[:12]
        label = self.slide_map[slide_ID]
        
        img_path = os.path.join(self.img_dir, patch)
        image = read_image(img_path).float()
        # from starter code
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label