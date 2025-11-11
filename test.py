import os

import cv2
import numpy as np

from models import create_model


def tensor2im_single_channel(image_tensor):
    """Convert a single-channel tensor to numpy array without tiling to 3 channels"""
    image_numpy = image_tensor[0].cpu().float().numpy()  # (C, H, W)
    if image_numpy.shape[0] == 1:
        image_numpy = image_numpy[0]  # Get first channel: (H, W)
    else:
        image_numpy = image_numpy.transpose(1, 2, 0)  # (H, W, C)
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def mse(a, b):
    return np.square(np.subtract(a,b)).mean()


def L1(a, b):
    return np.abs(np.subtract(a,b)).mean()


def test(opt, dataset):
    model = create_model(opt)

    test_results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    if not os.path.exists(test_results_dir):
        os.mkdir(test_results_dir)

    log = open(os.path.join(test_results_dir, "results.csv"), "w")
    log.write('img,mean_intensity,L1,L2,SSIM,PCC\n')

    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()

        img_name = data['img']

        # Get raw tensors and convert directly without the 3-channel tiling
        fake_B_numpy = tensor2im_single_channel(model.fake_B.data)
        
        visuals = model.get_current_visuals()
        
        # Debug: print what we got from the model
        print(f'  Direct conversion - fake_B shape: {fake_B_numpy.shape}, dtype: {fake_B_numpy.dtype}')
        print(f'  visuals keys: {visuals.keys()}')
        print(f'  visuals fake_B shape: {visuals["fake_B"].shape}, dtype: {visuals["fake_B"].dtype}')

        if not os.path.exists(os.path.join(test_results_dir, 'images')):
            os.mkdir(os.path.join(test_results_dir, 'images'))

        for ch in range(opt.input_nc):
            real_A = visuals['real_A'][:, :, ch]
            cv2.imwrite(os.path.join(test_results_dir, 'images', f'{img_name.split(".")[0]}_realA{ch}.tif'), real_A)

        # Use the direct conversion without 3-channel tiling
        for ch in range(opt.output_nc):
            print(f'  Channel {ch}: shape={fake_B_numpy.shape}, dtype={fake_B_numpy.dtype}, min={fake_B_numpy.min()}, max={fake_B_numpy.max()}, mean={fake_B_numpy.mean():.2f}')
            print(f'  Is C-contiguous: {fake_B_numpy.flags.c_contiguous}, Is F-contiguous: {fake_B_numpy.flags.f_contiguous}')
            # Ensure array is C-contiguous before saving
            fake_B_contiguous = np.ascontiguousarray(fake_B_numpy)
            cv2.imwrite(os.path.join(test_results_dir, 'images', f'{img_name.split(".")[0]}_fake{ch}.tif'), fake_B_contiguous)

            # for testing purposes only, run performance metrics (all input fields must have a matching target image)
            # real_B = visuals['real_B'][:, :, ch]
            # ... (the rest of the file)
            
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

