import os
import re
import glob
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import functools
import cv2
import pandas as pd
import json
import skimage
from module.RandStainNA import RandStainNA


def list_sort_nicely(l):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def fixed_clahe(image, **kwargs):
    clahe_mat = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = clahe_mat.apply(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image[:, :, 0] = clahe_mat.apply(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image

def min_max_normalization(image, **kwargs):
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image = image.astype('float32')
        image -= image.min()
        image /= image.max() 
    return image


# RandStainNA: https://github.com/yunboer/AR-classifier-and-AR-CycleGAN/blob/da6fb12a516ef72e2dc246da136aeb575812ddc2/AR-classifier/dataset/dataloader.py#L174
#              https://github.com/Katalip/ca-stinv-cnn/blob/ebf74974b5ce42c5cf1d5090d2d02215854a3a66/src/modules/augs.py#L168
#              https://github.com/yiqings/RandStainNA/blob/4897b0337ed2cdac41d0332dbfd6a1bde2b96598/classification/timm/data/transforms_factory.py#L248

def get_transforms(mode="train"):
    # medical augmentation
    if mode == "train":
        return A.Compose([
            # https://github.com/TIO-IKIM/CellViT/blob/9a24732da2709fc7c89c7bb48fe91acbd651c2a7/docs/readmes/example_train_config.md?plain=1#L95
            
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True), # 3-ch is okay

            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussNoise(var_limit=50.0, mean=0, per_channel=True, p=1.0),
            ], p=0.5),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),    # https://arxiv.org/pdf/2212.04690.pdf
            A.ToGray(p=0.2),                                                                # https://arxiv.org/pdf/2212.04690.pdf            
            # RandStainNA(std_hyper=-0.3, distribution='normal', p=1.0),
            
            # Normalization
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255, p=1.0),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image', 'mask2':'mask'})

    else:
        return A.Compose([
            A.Lambda(image=fixed_clahe, always_apply=True),

            # Normalization
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255, p=1.0),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image', 'mask2':'mask'})
    






class OCELOT_Cell_Seg_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/dataset/ocelot2023_v0.1.2'
        self.mode = mode
        if mode == 'train':
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/train/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_csv/*.csv')))

        else:
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/valid/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_csv/*.csv')))
            
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.cell_image_path)

    def __getitem__(self, idx):
        # image
        image = cv2.imread(self.cell_image_path[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # label
        cell_gt = cv2.imread(self.cell_gt_path[idx], cv2.IMREAD_GRAYSCALE)

        # preprocessing
        image   = cv2.resize(image,   (512, 512), interpolation=cv2.INTER_CUBIC)
        cell_gt = cv2.resize(cell_gt, (512, 512), interpolation=cv2.INTER_NEAREST)

        # extract certain classes from mask
        cell_gt = [(cell_gt == v) for v in [255, 1, 2]]
        cell_gt = np.stack(cell_gt, axis=-1)   

        # augmentation
        sample  = self.transforms(image=image, mask=cell_gt)
        image   = sample['image'].float()
        cell_gt = sample['mask'].float()

        return image, cell_gt, self.cell_csv_path[idx]


class OCELOT_Cell_MTL_DET_SEG_REC_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/dataset/ocelot2023_v0.1.2'
        self.mode = mode
        if mode == 'train':
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/train/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_csv/*.csv')))
            self.tissue_mask_path  = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/tissue_cropped/*.png'))) # 1: normal cell, 2: cancer cell, 255: BG

        else:
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/valid/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_csv/*.csv')))
            self.tissue_mask_path  = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/tissue_cropped/*.png'))) # 1: normal cell, 2: cancer cell, 255: BG
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.cell_image_path)

    def __getitem__(self, idx):
        # image
        image = cv2.imread(self.cell_image_path[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label
        cell_gt = cv2.imread(self.cell_gt_path[idx], cv2.IMREAD_GRAYSCALE)
        tissue_segmap = cv2.imread(self.tissue_mask_path[idx], cv2.IMREAD_GRAYSCALE)

        # preprocessing
        image         = cv2.resize(image,         (512, 512), interpolation=cv2.INTER_CUBIC)
        cell_gt       = cv2.resize(cell_gt,       (512, 512), interpolation=cv2.INTER_NEAREST)
        tissue_segmap = cv2.resize(tissue_segmap, (512, 512), interpolation=cv2.INTER_NEAREST)

        # extract certain classes from mask
        tissue_segmap[tissue_segmap != 2] = 0 # only save the cancer parts
        tissue_segmap[tissue_segmap == 2] = 1 # only save the cancer parts
        tissue_segmap = np.expand_dims(tissue_segmap, axis=-1)           

        cell_gt = [(cell_gt == v) for v in [255, 1, 2]]
        cell_gt = np.stack(cell_gt, axis=-1)        

        # augmentation
        sample        = self.transforms(image=image, mask=cell_gt, mask2=tissue_segmap)
        image         = sample['image'].float()
        cell_gt       = sample['mask'].float()
        tissue_segmap = sample['mask2'].float()

        return image, tissue_segmap, cell_gt, self.cell_csv_path[idx]


class OCELOT_Cell_MTL_DET_SEG_POI_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/dataset/ocelot2023_v0.1.2'
        self.mode = mode
        if mode == 'train':
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/train/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_csv/*.csv')))
            self.tissue_mask_path  = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/tissue_cropped/*.png'))) # 1: normal cell, 2: cancer cell, 255: BG

        else:
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/valid/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_csv/*.csv')))
            self.tissue_mask_path  = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/tissue_cropped/*.png'))) # 1: normal cell, 2: cancer cell, 255: BG
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.cell_image_path)

    def __getitem__(self, idx):
        # image
        image = cv2.imread(self.cell_image_path[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label
        cell_gt = cv2.imread(self.cell_gt_path[idx], cv2.IMREAD_GRAYSCALE)
        tissue_segmap = cv2.imread(self.tissue_mask_path[idx], cv2.IMREAD_GRAYSCALE)

        try:
            df = pd.read_csv(self.cell_csv_path[idx], header=None)
            df.columns = ['x', 'y', 'class']
            p1 = torch.tensor(len(df[df['class']==1].values)).float() 
            p2 = torch.tensor(len(df[df['class']==2].values)).float() 
            point = torch.stack([p1, p2], dim=0)
        except:
            point = torch.stack([torch.tensor(0.0), torch.tensor(0.0)], dim=0)


        # preprocessing
        image         = cv2.resize(image,         (512, 512), interpolation=cv2.INTER_CUBIC)
        cell_gt       = cv2.resize(cell_gt,       (512, 512), interpolation=cv2.INTER_NEAREST)
        tissue_segmap = cv2.resize(tissue_segmap, (512, 512), interpolation=cv2.INTER_NEAREST)

        # extract certain classes from mask
        tissue_segmap[tissue_segmap != 2] = 0 # only save the cancer parts
        tissue_segmap[tissue_segmap == 2] = 1 # only save the cancer parts
        tissue_segmap = np.expand_dims(tissue_segmap, axis=-1)           

        cell_gt = [(cell_gt == v) for v in [255, 1, 2]]
        cell_gt = np.stack(cell_gt, axis=-1)        

        # augmentation
        sample        = self.transforms(image=image, mask=cell_gt, mask2=tissue_segmap)
        image         = sample['image'].float()
        cell_gt       = sample['mask'].float()
        tissue_segmap = sample['mask2'].float()

        return image, tissue_segmap, cell_gt, point, self.cell_csv_path[idx]


class OCELOT_Cell_MTL_DET_SEG_REC_POI_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/dataset/ocelot2023_v0.1.2'
        self.mode = mode
        if mode == 'train':
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/train/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/cell_csv/*.csv')))
            self.tissue_mask_path  = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/train/tissue_cropped/*.png'))) # 1: normal cell, 2: cancer cell, 255: BG

        else:
            self.cell_image_path   = list_sort_nicely(glob.glob(os.path.join(self.root, f'images/valid/cell/*.jpg')))
            self.cell_gt_path      = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_seg/*.png'))) # 1: BG, 2: CA, 255: Unknown
            self.cell_csv_path     = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/cell_csv/*.csv')))
            self.tissue_mask_path  = list_sort_nicely(glob.glob(os.path.join(self.root, f'annotations/valid/tissue_cropped/*.png'))) # 1: normal cell, 2: cancer cell, 255: BG
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.cell_image_path)

    def __getitem__(self, idx):
        # image
        image = cv2.imread(self.cell_image_path[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label
        cell_gt = cv2.imread(self.cell_gt_path[idx], cv2.IMREAD_GRAYSCALE)
        tissue_segmap = cv2.imread(self.tissue_mask_path[idx], cv2.IMREAD_GRAYSCALE)

        try:
            df = pd.read_csv(self.cell_csv_path[idx], header=None)
            df.columns = ['x', 'y', 'class']
            p1 = torch.tensor(len(df[df['class']==1].values)).float() 
            p2 = torch.tensor(len(df[df['class']==2].values)).float() 
            point = torch.stack([p1, p2], dim=0)
        except:
            point = torch.stack([torch.tensor(0.0), torch.tensor(0.0)], dim=0)


        # preprocessing
        image         = cv2.resize(image,         (512, 512), interpolation=cv2.INTER_CUBIC)
        cell_gt       = cv2.resize(cell_gt,       (512, 512), interpolation=cv2.INTER_NEAREST)
        tissue_segmap = cv2.resize(tissue_segmap, (512, 512), interpolation=cv2.INTER_NEAREST)

        # extract certain classes from mask
        tissue_segmap[tissue_segmap != 2] = 0 # only save the cancer parts
        tissue_segmap[tissue_segmap == 2] = 1 # only save the cancer parts
        tissue_segmap = np.expand_dims(tissue_segmap, axis=-1)           

        cell_gt = [(cell_gt == v) for v in [255, 1, 2]]
        cell_gt = np.stack(cell_gt, axis=-1)        

        # augmentation
        sample        = self.transforms(image=image, mask=cell_gt, mask2=tissue_segmap)
        image         = sample['image'].float()
        cell_gt       = sample['mask'].float()
        tissue_segmap = sample['mask2'].float()

        return image, tissue_segmap, cell_gt, point, self.cell_csv_path[idx]








# Define the DataLoader
def get_dataloader(name, mode, batch_size, num_workers):

    if name == 'ocelot_cell_segmentation':
        if mode == 'train':            
            train_dataset = OCELOT_Cell_Seg_Dataset(mode='train')
            print("Train [Total] number = ", len(train_dataset))
            data_loader   = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
        elif mode == 'valid':
            valid_dataset = OCELOT_Cell_Seg_Dataset(mode='valid')
            print("Valid [Total] number = ", len(valid_dataset))
            data_loader   = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        else :
            test_dataset = OCELOT_Cell_Seg_Dataset(mode='test')
            print("TEST [Total] number = ", len(test_dataset))
            data_loader   = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)  


    elif name == 'ocelot_cell_mtl_det_seg_rec':
        if mode == 'train':            
            train_dataset = OCELOT_Cell_MTL_DET_SEG_REC_Dataset(mode='train')
            print("Train [Total] number = ", len(train_dataset))
            data_loader   = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
        elif mode == 'valid':
            valid_dataset = OCELOT_Cell_MTL_DET_SEG_REC_Dataset(mode='valid')
            print("Valid [Total] number = ", len(valid_dataset))
            data_loader   = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        else :
            test_dataset = OCELOT_Cell_MTL_DET_SEG_REC_Dataset(mode='test')
            print("TEST [Total] number = ", len(test_dataset))
            data_loader   = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)  


    elif name == 'ocelot_cell_mtl_det_seg_poi':
        if mode == 'train':            
            train_dataset = OCELOT_Cell_MTL_DET_SEG_POI_Dataset(mode='train')
            print("Train [Total] number = ", len(train_dataset))
            data_loader   = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
        elif mode == 'valid':
            valid_dataset = OCELOT_Cell_MTL_DET_SEG_POI_Dataset(mode='valid')
            print("Valid [Total] number = ", len(valid_dataset))
            data_loader   = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        else :
            test_dataset = OCELOT_Cell_MTL_DET_SEG_POI_Dataset(mode='test')
            print("TEST [Total] number = ", len(test_dataset))
            data_loader   = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)  


    elif name == 'ocelot_cell_mtl_det_seg_rec_poi':
        if mode == 'train':            
            train_dataset = OCELOT_Cell_MTL_DET_SEG_REC_POI_Dataset(mode='train')
            print("Train [Total] number = ", len(train_dataset))
            data_loader   = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
        elif mode == 'valid':
            valid_dataset = OCELOT_Cell_MTL_DET_SEG_REC_POI_Dataset(mode='valid')
            print("Valid [Total] number = ", len(valid_dataset))
            data_loader   = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        else :
            test_dataset = OCELOT_Cell_MTL_DET_SEG_REC_POI_Dataset(mode='test')
            print("TEST [Total] number = ", len(test_dataset))
            data_loader   = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)  


    return data_loader


