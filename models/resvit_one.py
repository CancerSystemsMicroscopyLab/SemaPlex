import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.vit_name,opt.fineSize, opt.pre_trained_path,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)


        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)

        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.to(self.device)
            input_B = input_B.to(self.device)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = ''

    def forward(self):
        self.netG.train()

        self.real_A = Variable(self.input_A)
        self.fake_B= self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.netG.eval()

            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)
            
            # Debug: print raw tensor stats
            print(f'    Raw fake_B tensor: shape={self.fake_B.shape}, min={self.fake_B.min():.3f}, max={self.fake_B.max():.3f}, mean={self.fake_B.mean():.3f}')

    # get image paths
    def get_image_paths(self):
        return self.image_paths

        
    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_L1*1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        
        # Debug: check the shapes
        print(f'    After tensor2im - real_A: {real_A.shape}, fake_B: {fake_B.shape}, real_B: {real_B.shape}')
        
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
