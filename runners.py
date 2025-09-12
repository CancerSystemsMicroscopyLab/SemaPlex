from options.test_options import TestOptions
from options.train_options import TrainOptions
from test import test
from train import train

    
def train_unet(experiment_name, checkpoints_dir, dataset, input_nc, output_nc):
    opt = TrainOptions().parse()
    opt.name = experiment_name + '/unet'
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.lambda_adv = 0
    opt.lambda_A = 100
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 50
    opt.niter_decay = 50
    opt.checkpoints_dir = checkpoints_dir
    opt.lr = 0.0002
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    TrainOptions.publish(opt)
    train(opt, dataset)
    
def test_unet(experiment_name, checkpoints_dir, dataset, input_nc, output_nc):
    opt = TestOptions().parse()
    opt.name = experiment_name + '/unet'
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.phase = 'test'
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 0
    opt.niter_decay = 0
    opt.checkpoints_dir = checkpoints_dir
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    
    TestOptions.publish(opt)
    test(opt, dataset)