import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data


class image2imagedataset(data.Dataset):
    def __init__(self, dataroot, inputs, targets, image_names, isTrain=True):

        self.dataroot = dataroot
        self.inputs = inputs
        self.targets = targets
        self.isTrain = isTrain
        self.img_names = image_names


    def transform(self, image, mask):
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if self.isTrain:

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        image = TF.normalize(image, (0.5,), (0.5,))
        mask = TF.normalize(mask, (0.5,), (0.5,))

        return image, mask


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):

        image_name = self.img_names[idx]

        in_chs = []
        for ch_name in self.inputs:
            x_ch = cv2.imread(os.path.join(self.dataroot, ch_name, image_name), cv2.IMREAD_GRAYSCALE)
            in_chs.append(x_ch)

        # Load targets for both train and test. In test, allow missing targets (zeros fallback) for inference-only use.
        out_chs = []
        for ch_name in self.targets:
            target_path = os.path.join(self.dataroot, ch_name, image_name)
            if os.path.exists(target_path):
                y_ch = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
                out_chs.append(y_ch)
            else:
                # If target doesn't exist (test-time inference), use zeros of expected size
                # Determine size from an input channel if available, else default 256x256
                if len(in_chs) > 0 and in_chs[0] is not None:
                    h, w = in_chs[0].shape[:2]
                else:
                    h, w = 256, 256
                out_chs.append(np.zeros((h, w), dtype=np.uint8))

        y = np.stack(out_chs, axis=2)

        # Replace any None inputs with zeros before stacking
        sanitized_in_chs = []
        for ch in in_chs:
            if ch is None:
                sanitized_in_chs.append(np.zeros((y.shape[0], y.shape[1]), dtype=np.uint8))
            else:
                sanitized_in_chs.append(ch)

        x = np.stack(sanitized_in_chs, axis=2)
        x, y = self.transform(x, y)

        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)

        return {'A': x, 'B': y, 'img': self.img_names[idx]}
