'''
    Dataset_CSV_SEM_SEG: Pytorch dataset class for semantic segmentation
    Dataset_CSV_CLS:Pytorch dataset class for classification
        single label, without one-hot-encoding

'''

import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
import numpy as np


class Dataset_CSV_CLS(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None, test_mode=False):
        assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.transform = transform
        self.test_mode = test_mode

    def __getitem__(self, index):
        file_img = self.df.iloc[index][0]
        assert os.path.exists(file_img), f'image file {file_img} does not exists'
        image = cv2.imread(file_img)
        assert image is not None, f'{file_img} error.'
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        list_transform = []
        if self.transform:
            list_transform.append(self.transform)    # randomcrop may change the image size
        # if (self.image_shape is not None) and (image.shape[:2] != self.image_shape[:2]):
        if self.image_shape is not None:
            list_transform.append(A.Resize(height=self.image_shape[0], width=self.image_shape[1]))

        augmented = A.Compose(list_transform)(image=image)
        image = augmented['image']

        image = transforms.ToTensor()(image)   # convert numpy array to pytorch tensor and normalize to (0,1)

        if self.test_mode:
            return image
        else:
            label = int(self.df.iloc[index][1])
            return image, label

    def __len__(self):
        return len(self.df)



class Dataset_CSV_SEM_SEG(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None, test_mode=False, mask_threshold=127):
        assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.transform = transform
        self.test_mode = test_mode
        self.mask_threshold = mask_threshold

    def __getitem__(self, index):
        file_img = self.df.iloc[index][0]
        assert os.path.exists(file_img), f'image file {file_img} does not exists'
        image = cv2.imread(file_img)
        assert image is not None, f'{file_img} error.'
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.test_mode:
            file_mask = self.df.iloc[index][1]
            assert os.path.exists(file_mask), f'image mask file {file_mask} does not exists'
            mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f'{file_mask} error.'

        list_transform = []   # A.NoOp
        if self.transform:
            list_transform.append(self.transform)
        if self.image_shape is not None:  # and (image.shape[:2] != self.image_shape[:2]):  # randomcrop can change the image size
            list_transform.append(A.Resize(height=self.image_shape[0], width=self.image_shape[1]))

        if not self.test_mode:
            augmented = A.Compose(list_transform)(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            _, mask = cv2.threshold(mask, self.mask_threshold, 1, cv2.THRESH_BINARY)  # >127 set to 255, else 0. 127 set to 1,
            mask = np.expand_dims(mask, axis=0)  #grayscale image adding the channel dimension.
        else:
            augmented = A.Compose(list_transform)(image=image)
            image = augmented['image']

        image = transforms.ToTensor()(image)  # convert numpy array to pytorch tensor and normalize to (0,1)

        if self.test_mode:
            return image
        else:
            mask = torch.from_numpy(mask)
            return image, mask

    def __len__(self):
        return len(self.df)

