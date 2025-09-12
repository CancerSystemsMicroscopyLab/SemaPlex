import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from models import create_model


def mse(a, b):
    return np.square(np.subtract(a,b)).mean()


def L1(a, b):
    return np.abs(np.subtract(a,b)).mean()


def test(opt, dataset):
    model = create_model(opt)

    test_results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    if not os.path.exists(test_results_dir):
        os.mkdir(test_results_dir)

    log = open(os.path.join(test_results_dir, f"results.csv"), "w")
    log.write('img,mean_intensity,L1,L2,SSIM,PCC\n')

    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()

        img_name = data['img']

        visuals = model.get_current_visuals()

        if not os.path.exists(os.path.join(test_results_dir, 'images')):
            os.mkdir(os.path.join(test_results_dir, 'images'))

        for ch in range(opt.input_nc):
            real_A = visuals['real_A'][:, :, ch]
            cv2.imwrite(os.path.join(test_results_dir, 'images', f'{img_name.split(".")[0]}_realA{ch}.tif'), real_A)

        for ch in range(opt.output_nc):
            fake_B = visuals['fake_B'][:, :, ch]
            cv2.imwrite(os.path.join(test_results_dir, 'images', f'{img_name.split(".")[0]}_fake{ch}.tif'), fake_B)

            # for testing purposes only, run performance metrics (all input fields must have a matching target image)
            # real_B = visuals['real_B'][:, :, ch]
            # cv2.imwrite(os.path.join(test_results_dir, 'images', f'{img_name.split(".")[0]}_real{ch}.tif'), real_B)
            #
            # real_mean_intensity = np.mean(real_B.flatten())
            # l1_loss = L1(real_B, fake_B)
            # mse_loss = mse(real_B, fake_B)
            # ssim_loss = ssim(real_B, fake_B)
            # pcc = np.corrcoef(real_B.flatten(), fake_B.flatten())[0][1]
            #
            # corrected_name = f'{img_name.split(".")[0]}_real{ch}.tif'
            # log.write(f"{corrected_name},{real_mean_intensity},{l1_loss},{mse_loss},{ssim_loss},{pcc}\n")

        print('%04d: process image... %s' % (i, img_name))

