# Generative semantic multiplexing (SemaPlex) for accessible and scalable multiplexed fluorescence imaging

# Overview

Multiplexed fluorescence imaging enhances spatially-resolved interrogation of complex, multi-molecular cell processes that are insufficiently sampled using standard 4-5 plex imaging. To improve accessibility and scalability for multiplexed imaging, we demonstrate generative ‘Semantic Multiplexing’ (SemaPlex); a simple experimental and deep learning strategy for amplifying marker plexity several-fold by semantically unmixing multiple markers combined per imaging channel. 
# Installation

## Hardware Requirements

The package requires a CUDA enabled GPU to run. We suggest a computer with the minimum specs: <br />
RAM: 16+ GB  <br />
CPU: 4+ cores, 3.3+ GHz/core<br />
CUDA GPU: 16+ GB VRAM 

## Software Requirements

Users should install the following packages in a python environment (3.10)
```
torch>=2.2.1
torchvision>=0.17.1
cv2
ml_collections
cuda>=11.2
```

- Download or clone this repo. e.g.
```bash
git clone https://github.com/CancerSystemsMicroscopyLab/SemaPlex
```

Installation time ~20 mins on a typical computer with standard internet connection

# SemaPlex Unmixing

## Preprocessing
To use the model as described in the paper, images need to be in 8bit depth with 256x256 resolution.

To use the dataloader we have provided, images should be separated by channel and placed in a folder, inside a parent folder and have matching corresponding names, as done in the 'sample_dataset' folder example we have provided. E.g.

```
Parent
│
└───mix555
│   │   img1.tif
│   │   img2.tif
│   │   ...
│   
└───DAPI
    │   img1.tif
    │   img2.tif
    │   ...
  
```

## Training and applying the model

1. Set the dataset path and input/target channels in the main.py file. Guiding channels may be specified as an addition input channel. Also specify the target markers to be unmixed. For example, in this repo, parameters are set so that lamin will be unmixed from a mixture here named 555nm and augmented/guided by DAPI channel.

2. Run the main.py scripte which will then train a model using all data which have matching input-target pairs (by image file name). Inputs field without corresponding target fields will not be used for training but predicted using the trained model subsequently. All results are placed in the results folder.

3. To use further apply the model to unmix, change the path_to_dataset variable to point to the new dataset. To prevent retraining of a model, remove/comment out the call to 'train_unet' on line 36. The script will then unmix all images in the new dataset.

A sample result has been provided showing prediction of lamin from a mixture in 555nm further guided by DAPI which can be reproduced by running the sample main.py script as is.  
Run time can vary depending on hardware and dataset size. ~1hrs-4hrs runtime might be expected. 

For 

## Downstream processing
Downstream results demonstrated in the manuscript - TBC- were created using scripts hosted at - TBC - 


# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
TBC
```

For any questions, comments and contributions, please contact Dr John Lock (john.lock@unsw.edu.au) <br />

(c) Cancer Systems Microscopy Lab 2024

## Acknowledgments
This code uses libraries from [ResViT](https://github.com/icon-lab/ResViT), and [ExIF](https://github.com/CancerSystemsMicroscopyLab/VirtualLabelling) repository.
