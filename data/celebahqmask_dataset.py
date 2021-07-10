import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from iglovikov_helper_functions.utils.image_utils import load_rgb, load_grayscale

from data.image_folder import make_dataset

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from utils.utils import onehot_parse_map

from data.ffhq_dataset import complex_imgaug, random_gray
from pytorch_toolbelt.utils.torch_utils import image_to_tensor


class CelebAHQMaskDataset(BaseDataset):

    def __init__(self, opt, transform, deg_transform, hr_transform):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.Pimg_size
        self.lr_size = opt.Gin_size
        self.hr_size = opt.Gout_size
        self.shuffle = True if opt.isTrain else False 

        self.transform = transform
        self.deg_transform = deg_transform
        self.hr_transform = hr_transform

        self.img_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'images')))
        self.mask_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'labels')))

    def __len__(self,):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        sample = {}
        img_path = self.img_dataset[idx]
        mask_path = self.mask_dataset[idx]

        image = load_rgb(img_path, lib="cv2")
        mask = load_grayscale(mask_path)
        sample = self.transform(image=image, mask=mask)

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        hr_sample = self.hr_transform(image=image, mask=mask)
        hr_image, hr_mask = hr_sample["image"], hr_sample["mask"]

        if self.deg_transform is not None:
            degraded_sample = self.deg_transform(image=image, mask=mask)
            degraded_image, degraded_mask = degraded_sample["image"], degraded_sample["mask"]
        else:
            degraded_image = hr_image
            degraded_mask = hr_mask

        hr_mask = torch.from_numpy(hr_mask)
        return {
            "HR_paths": img_path,
            "HR": image_to_tensor(hr_image),
            "Mask": torch.unsqueeze(hr_mask, 0).long(),
            "LR": image_to_tensor(degraded_image),
        }
