import torch
from networks import DualGansGenerator, CycleGanResnetGenerator, GeneratorUNet
from huggan.pytorch.cyclegan.modeling_cyclegan import GeneratorResNet
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.utils import make_grid
import torch
import torch.nn as nn


class ModelLoader:

    def __init__(self, model, model_name, device, model_dict, gen, model_dict_recon):
         self.model = model
         self.model_name = model_name
         self.device = device
         self.model_dict = model_dict
         self.gen = gen
         self.model_dict_recon = model_dict_recon

     
    def model_loader(self):
        for i in range(len(self.model_name)):
            path = 'saved_models/test_' + self.model_name[i] + '_generator_b.pth'
            self.model_dict[self.model_name[i]] = path
        for i in range(len(self.model_name) - 1):
            path = 'saved_models/test_' + self.model_name[i] + '_generator_a.pth'
            self.model_dict_recon[self.model_name[i]] = path
                    
        return self.model_dict[self.model_name[i]]

    def load_model(self, generator, model_state_file):
        model = generator.to(self.device)
        print(model_state_file)
        model.load_state_dict(torch.load(model_state_file))
        return model

    def pred_pipeline(self, img, input_shape):
        orig_shape = img.shape
        img = img.squeeze(0)
        transform = Compose([
            T.ToPILImage(),
                T.Resize(input_shape),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
        input = transform(img[0])
        output_real = self.gen(input.cuda())
        
        out_img_real = make_grid(output_real,
                           nrow=1, normalize=True)          

        print(orig_shape)

        out_transform = Compose([
            T.Resize(orig_shape[2:]),
            T.ToPILImage()
        ])
        out = out_transform(out_img_real)
        width, height = out.size
        print(width, height)
        return out_transform(out_img_real)

    def model_init(self):

        self.model_dict[self.model_name[0]] = self.model_loader()
        print(self.model_dict[self.model_name[0]])
        if (self.model == 'dual_gans_un'):
            self.gen = self.load_model(DualGansGenerator(), self.model_dict[self.model_name[0]])

        elif (self.model == 'dual_gans_semi'):
            self.gen = self.load_model(DualGansGenerator(), self.model_dict[self.model_name[0]])

        elif (self.model == 'cycle_gan_un'):
            self.gen = self.load_model(CycleGanResnetGenerator(), self.model_dict[self.model_name[0]])
                
        elif (self.model == 'cycle_gan_semi'):
            self.gen = self.load_model(CycleGanResnetGenerator(), self.model_dict[self.model_name[0]])

        elif (self.model == 'cycle_gan_512'):
            n_channels = 3
            image_size = 512
            input_shape = (image_size, image_size)

            transform = Compose([
                T.ToPILImage(),
                    T.Resize(input_shape),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

            # self.sim2real = GeneratorResNet.from_pretrained('Chris1/sim2real-512', input_shape=(n_channels, image_size, image_size), 
            #     num_residual_blocks=9)
            self.gen = GeneratorResNet.from_pretrained('Chris1/sim2real-512', input_shape=(n_channels, image_size, image_size), num_residual_blocks=9).cuda()

        return self.gen
    