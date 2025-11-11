from options.test_options import TestOptions
from options.train_options import TrainOptions
from test import test
from train import train

    
def train_unet(experiment_name, checkpoints_dir, dataset, input_nc, output_nc, train_overrides: dict | None = None):
    """Train UNet-based ResViT model.

    train_overrides: optional dict of option overrides, e.g.,
      {"niter": 20, "niter_decay": 0, "lr": 2e-4, "lambda_A": 50, "norm": "instance"}
    """
    opt = TrainOptions().parse()
    opt.name = experiment_name + '/unet'
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.lambda_adv = 0
    opt.lambda_A = 100
    # InstanceNorm is generally more stable than BatchNorm for tiny batches
    opt.norm = 'instance'
    opt.pool_size = 0
    opt.niter = 50  # 50 epochs should give reasonable results
    opt.niter_decay = 50  # Then 50 more with decay
    opt.checkpoints_dir = checkpoints_dir
    opt.lr = 0.0002
    opt.input_nc = input_nc
    opt.output_nc = output_nc

    # Apply user-provided overrides if any
    if train_overrides:
        for k, v in train_overrides.items():
            if hasattr(opt, k):
                setattr(opt, k, v)

    TrainOptions.publish(opt)
    train(opt, dataset)
    
def test_unet(experiment_name, checkpoints_dir, dataset, input_nc, output_nc, test_overrides: dict | None = None):
    opt = TestOptions().parse()
    opt.name = experiment_name + '/unet'
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.phase = 'test'
    # Match normalization used in training for model definition
    opt.norm = 'instance'
    opt.pool_size = 0
    opt.niter = 0
    opt.niter_decay = 0
    opt.checkpoints_dir = checkpoints_dir
    opt.input_nc = input_nc
    opt.output_nc = output_nc

    if test_overrides:
        for k, v in test_overrides.items():
            if hasattr(opt, k):
                setattr(opt, k, v)
    
    TestOptions.publish(opt)
    test(opt, dataset)