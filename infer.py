import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
from PIL import Image
import cv2
from models import UNetSemantic, GatedGenerator
import argparse
from configs import Config

class Predictor():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if cfg.cuda else 'cpu')
        self.masking = UNetSemantic().to(self.device)
        self.masking.load_state_dict(torch.load('weights\model_segm_19_135000.pth', map_location='cpu'))
        #self.masking.eval()

        self.inpaint = GatedGenerator().to(self.device)
        self.inpaint.load_state_dict(torch.load('weights/model_6_100000.pth', map_location='cpu')['G'])
        self.inpaint.eval()

    def save_image(self, img_list, save_img_path, nrow):
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow = nrow)
        print(f"Save image to {save_img_path}")

    def predict(self, image, outpath='sample/results.png'):
        outpath=f'sample/results_{image}.png'
        image = 'sample/'+image
        img = cv2.imread(image+'_masked.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0).to(self.device)

        img_ori = cv2.imread(image+'.jpg')
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        img_ori = cv2.resize(img_ori, (self.cfg.img_size, self.cfg.img_size))
        img_ori = torch.from_numpy(img_ori.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ori = img_ori.unsqueeze(0)
        with torch.no_grad():
            outputs = self.masking(img)
            _, out = self.inpaint(img, outputs)
            inpaint = img * (1 - outputs) + out * outputs
        masks = img * (1 - outputs) + outputs #torch.cat([outputs, outputs, outputs], dim=1)
        

         
        self.save_image([img, masks, inpaint, img_ori], outpath, nrow=4)

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training custom model')
    parser.add_argument('--image', default=None, type=str, help='resume training')
    parser.add_argument('config', default='config', type=str, help='config training')                         
    args = parser.parse_args() 

    config = Config(f'./configs/{args.config}.yaml')


    model = Predictor(config)
    model.predict(args.image)