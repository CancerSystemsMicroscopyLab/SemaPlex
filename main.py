import os
import sys

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from dataloaders.image2image_dataset import image2imagedataset
from runners import train_unet, test_unet


if __name__ == '__main__':

    ###### to set
    path_to_dataset = './sample_dataset'
    path_to_dataset = r'D:\Team Members\Andrew Gunawan\MD_unmixing\code_for_submission\ResViT\sample_dataset'
    inputs = ['mix555', 'DAPI']
    targets = ['lamin']
    ##########


    experiment_name = '_'.join(inputs)

    dataroot = path_to_dataset
    # checkpoints = './results/'
    checkpoints = r'D:\Team Members\Andrew Gunawan\MD_unmixing\code_for_submission\ResViT\results'
    target_images = np.array(os.listdir(os.path.join(dataroot, targets[0])))
    input_images = np.array(os.listdir(os.path.join(dataroot, inputs[0])))

    fold_folder = os.path.join(f'{experiment_name}', f'fold0')
    save_dir = os.path.join(f'{checkpoints}', f'{fold_folder}')
    os.makedirs(save_dir, exist_ok=True)

    log = open(os.path.join(save_dir, "trainset.txt"), "w")

    train_dataset = image2imagedataset(dataroot, inputs, targets, target_images)
    test_dataset = image2imagedataset(dataroot, inputs, targets, input_images, isTrain=False)

    train_unet(fold_folder, checkpoints, train_dataset, len(inputs), len(targets))
    test_unet(fold_folder, checkpoints, test_dataset, len(inputs), len(targets))

        