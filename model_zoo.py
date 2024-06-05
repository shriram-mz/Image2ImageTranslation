from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from networks import DualGansGenerator, CycleGanResnetGenerator, GeneratorUNet
from data_loader import DataLoader
from model_loader import ModelLoader
import argparse

class ModelZoo:
    def __init__(self, args):
        self.data_root = args.data_root
        print(self.data_root)
        self.model = args.model
        self.model_name = [self.model]
        self.model_dict = defaultdict.fromkeys(self.model_name)
        self.model_dict_recon = defaultdict.fromkeys(self.model_name)
        print(self.model_dict, self.model_dict_recon)
        self.image_size = 256
        ngpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gen = None
        self.output_dir = args.output_path
        self.ModelLoader = ModelLoader(self.model, self.model_name, self.device, self.model_dict, self.gen, self.model_dict_recon)
    
    def normalize(self, image):
        return (image - torch.min(image))/(torch.max(image) - torch.min(image))
    
    def load_model(self):
        gen = self.ModelLoader.model_init()
        return gen


    def infer(self, gen):
        batch_size = 4
        data = DataLoader(self.data_root, self.image_size, batch_size)
        pos = int(np.random.rand() * len(data.names)/batch_size)
        if pos < 0 or (pos+1) * batch_size >= len(data.names) * batch_size: 
            pos = 0
            
        x, y = next(data.data_generator(pos))

        og = Variable(y, requires_grad = False).to(self.device)
        og_im = self.normalize(og)
        if (self.model == 'cycle_gan_512'):
            h = self.image_size*2
            w = self.image_size*2
            print(h,w)
            input_shape = (h, w)
            output = self.ModelLoader.pred_pipeline(og.data, input_shape)
            output = np.array(output)
            img = Image.fromarray((output.astype(np.uint8)))
            img.save('output_512.png')
            print(type(output))

        else:   
            # batch_size = 4
            # data = DataLoader(self.data_root, self.image_size, batch_size)
            # pos = int(np.random.rand() * len(data.names)/batch_size)
            # if pos < 0 or (pos+1) * batch_size >= len(data.names) * batch_size: 
            #     pos = 0
                
            # x, y = next(data.data_generator(pos))

            # og = Variable(y, requires_grad = False).to(self.device)
            # og_im = self.normalize(og)
            
            
            gen_img = gen(og_im)
            output = torch.cat((og.data, gen_img.data), -2)
            output_path = self.output_dir
            save_image(output, f"{output_path}/model_zoo_{self.model_name}.png", normalize=True )


def main(args):
    mz = ModelZoo(args)
    gen = mz.load_model()
    mz.infer(gen)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
    description='Sim2Real model zoo')
    argparser.add_argument('--model',type=str,help='Name of the model to run.')
    argparser.add_argument('--data_root', type=str, help='Root directory of the dataset')   
    argparser.add_argument('--image_size', type=int, default=256, help='Size of the image.')
    argparser.add_argument('--output_path', type=str, help='Path where the output will get saved')
    args = argparser.parse_args()
    main(args)
