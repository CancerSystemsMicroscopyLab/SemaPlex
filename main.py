import os

import numpy as np

from dataloaders.image2image_dataset import image2imagedataset
from runners import train_unet, test_unet


if __name__ == '__main__':

    ###### to set
    ### path_to_dataset = './sample_dataset'
    ###path_to_dataset = r'D:\Team Members\Andrew Gunawan\MD_unmixing\code_for_submission\ResViT\sample_dataset'
    path_to_dataset = r'C:\Arnas\PythonWork\SemaPlex\sample_dataset'
    inputs = ['mix555', 'DAPI']
    targets = ['lamin']
    ##########


    experiment_name = '_'.join(inputs)

    dataroot = path_to_dataset
    # checkpoints = './results/'
    checkpoints = r'C:\Arnas\PythonWork\SemaPlex\results'
    
    # For training: use only images that exist in target folder
    target_images = np.array(os.listdir(os.path.join(dataroot, targets[0])))
    
    # For testing: use only images that exist in ALL input folders
    # (to avoid trying to load missing files)
    input_images_sets = [set(os.listdir(os.path.join(dataroot, inp))) for inp in inputs]
    common_images = input_images_sets[0].intersection(*input_images_sets[1:])
    input_images = np.array(sorted(list(common_images)))

    fold_folder = os.path.join(experiment_name, 'fold0')
    save_dir = os.path.join(f'{checkpoints}', f'{fold_folder}')
    os.makedirs(save_dir, exist_ok=True)

    log = open(os.path.join(save_dir, "trainset.txt"), "w")

    train_dataset = image2imagedataset(dataroot, inputs, targets, target_images)
    test_dataset = image2imagedataset(dataroot, inputs, targets, input_images, isTrain=False)

    # Optional: override training/testing hyperparameters here
    # Recommended tuning for small datasets (12 images):
    # - Longer base training without decay helps convergence
    # - Moderate lambda_A balances reconstruction quality and expressiveness
    # - continue_train=True to resume from existing checkpoint for more epochs
    
    # ACTIVE: Balanced configuration (between conservative and oversaturated)
    train_overrides = {
        'niter': 70,           # Longer stable learning phase
        'niter_decay': 30,     # Moderate decay period
        'lambda_A': 75,        # Balanced L1 penalty (between 50-100)
        'lr': 0.0002,          # Keep default learning rate
        'continue_train': False,  # Set True to continue from current checkpoint
    }
    test_overrides = {
        'norm': 'instance',    # Must match training norm
    }

    # Train the model (this will take 1-4 hours)
    print("Starting training... This will take 1-4 hours depending on your hardware.")
    train_unet(fold_folder, checkpoints, train_dataset, len(inputs), len(targets), train_overrides)
    
    # Test the model
    test_unet(fold_folder, checkpoints, test_dataset, len(inputs), len(targets), test_overrides)

        