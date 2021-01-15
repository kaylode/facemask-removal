import os
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image


from models import *
from losses import *
from datasets import Places365Dataset, FacemaskDataset


def adjust_learning_rate(optimizer, gamma, num_steps=1):
    for i in range(num_steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma

def get_epoch_iters(path):
    path = os.path.basename(path)
    tokens = path[:-4].split('_')
    try:
        if tokens[-1] == 'interrupted':
            epoch_idx = int(tokens[-3])
            iter_idx = int(tokens[-2])
        else:
            epoch_idx = int(tokens[-2])
            iter_idx = int(tokens[-1])
    except:
        return 0, 0

    return epoch_idx, iter_idx

def load_checkpoint(model_G, model_D, path):
    state = torch.load(path,map_location='cpu')
    model_G.load_state_dict(state['G'])
    model_D.load_state_dict(state['D'])
    print('Loaded checkpoint successfully')

class Trainer():
    def __init__(self, args, cfg):
        
        if args.resume is not None:
            epoch, iters = get_epoch_iters(args.resume)
        else:
            epoch = 0
            iters = 0

        if not os.path.exists(cfg.checkpoint_path):
            os.makedirs(cfg.checkpoint_path)
        if not os.path.exists(cfg.sample_folder):
            os.makedirs(cfg.sample_folder)

        self.cfg = cfg
        self.step_iters = cfg.step_iters
        self.gamma = cfg.gamma
        self.visualize_per_iter = cfg.visualize_per_iter
        self.print_per_iter = cfg.print_per_iter
        self.save_per_iter = cfg.save_per_iter
        
        self.start_iter = iters
        self.iters = 0
        self.num_epochs = cfg.num_epochs
        self.device = torch.device('cuda' if cfg.cuda else 'cpu')

        trainset = FacemaskDataset(cfg) # Places365Dataset(cfg) #

        self.trainloader = data.DataLoader(
            trainset, 
            batch_size=cfg.batch_size,
            num_workers = cfg.num_workers,
            pin_memory = True, 
            shuffle=True,
            collate_fn = trainset.collate_fn)

        self.epoch = int(self.start_iter / len(self.trainloader))
        self.iters = self.start_iter
        self.num_iters = (self.num_epochs+1) * len(self.trainloader)

        self.model_G = GatedGenerator().to(self.device)
        self.model_D = NLayerDiscriminator(cfg.d_num_layers, use_sigmoid=False).to(self.device)
        self.model_P = PerceptualNet(name = "vgg16", resize=False).to(self.device)

        if args.resume is not None:
            load_checkpoint(self.model_G, self.model_D, args.resume)

        self.criterion_adv = GANLoss(target_real_label=0.9, target_fake_label=0.1)
        self.criterion_rec = nn.SmoothL1Loss()
        self.criterion_ssim = SSIM(window_size = 11)
        self.criterion_per = nn.SmoothL1Loss()

        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=cfg.lr)
        self.optimizer_G = torch.optim.Adam(self.model_G.parameters(), lr=cfg.lr)

    def validate(self, sample_folder, sample_name, img_list):
        save_img_path = os.path.join(sample_folder, sample_name+'.png') 
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)

        # imgs shape: Bx5xCxWxH

        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow= 5)
        print(f"Save image to {save_img_path}")

    def fit(self):
        self.model_G.train()
        self.model_D.train()

        running_loss = {
            'D': 0,
            'G': 0,
            'P': 0,
            'R_1': 0,
            'R_2': 0,
            'T': 0,
        }

        running_time = 0
        step = 0
        try:
            for epoch in range(self.epoch, self.num_epochs):
                self.epoch = epoch
                for i, batch in enumerate(self.trainloader):
                    start_time = time.time()
                    imgs = batch['imgs'].to(self.device)
                    masks = batch['masks'].to(self.device)

                    # Train discriminator
                    self.optimizer_D.zero_grad()
                    self.optimizer_G.zero_grad()
                    
                    first_out, second_out = self.model_G(imgs, masks)

                    first_out_wholeimg = imgs * (1 - masks) + first_out * masks     
                    second_out_wholeimg = imgs * (1 - masks) + second_out * masks

                    masks = masks.cpu()

                    fake_D = self.model_D(second_out_wholeimg.detach())
                    real_D = self.model_D(imgs)

                    loss_fake_D = self.criterion_adv(fake_D, target_is_real=False)
                    loss_real_D = self.criterion_adv(real_D, target_is_real=True)

                    loss_D = (loss_fake_D + loss_real_D) * 0.5

                    loss_D.backward()
                    self.optimizer_D.step()

                    real_D = None

                    # Train Generator
                    self.optimizer_D.zero_grad()
                    self.optimizer_G.zero_grad()

                    fake_D = self.model_D(second_out_wholeimg)
                    loss_G = self.criterion_adv(fake_D, target_is_real=True)

                    fake_D = None
                    
                    # Reconstruction loss
                    loss_l1_1 = self.criterion_rec(first_out_wholeimg, imgs)
                    loss_l1_2 = self.criterion_rec(second_out_wholeimg, imgs)
                    loss_ssim_1 = self.criterion_ssim(first_out_wholeimg, imgs)
                    loss_ssim_2 = self.criterion_ssim(second_out_wholeimg, imgs)

                    loss_rec_1 = 0.5 * loss_l1_1 + 0.5 * (1 - loss_ssim_1)
                    loss_rec_2 = 0.5 * loss_l1_2 + 0.5 * (1 - loss_ssim_2)

                    # Perceptual loss
                    loss_P  = self.model_P(second_out_wholeimg, imgs)                          

                    loss = self.cfg.lambda_G * loss_G + self.cfg.lambda_rec_1 * loss_rec_1 + self.cfg.lambda_rec_2 * loss_rec_2 + self.cfg.lambda_per * loss_P
                    loss.backward()
                    self.optimizer_G.step()

                    end_time = time.time()

                    imgs = imgs.cpu()
                    # Visualize number
                    running_time += (end_time - start_time)
                    running_loss['D'] += loss_D.item()
                    running_loss['G'] += (self.cfg.lambda_G * loss_G.item())
                    running_loss['P'] += (self.cfg.lambda_per * loss_P.item())
                    running_loss['R_1'] += (self.cfg.lambda_rec_1 * loss_rec_1.item())
                    running_loss['R_2'] += (self.cfg.lambda_rec_2 * loss_rec_2.item())
                    running_loss['T'] += loss.item()
                    

                    if self.iters % self.print_per_iter == 0:
                        for key in running_loss.keys():
                            running_loss[key] /= self.print_per_iter
                            running_loss[key] = np.round(running_loss[key], 5)
                        loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                        print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                        
                        running_loss = {
                            'D': 0,
                            'G': 0,
                            'P': 0,
                            'R_1': 0,
                            'R_2': 0,
                            'T': 0,
                        }
                        running_time = 0
                
                    if self.iters % self.save_per_iter  == 0:
                        torch.save({
                            'D': self.model_D.state_dict(),
                            'G': self.model_G.state_dict(),
                        }, os.path.join(self.cfg.checkpoint_path, f"model_{self.epoch}_{self.iters}.pth"))

                    # Step learning rate
                    if self.iters == self.step_iters[step]:
                        adjust_learning_rate(self.optimizer_D, self.gamma)
                        adjust_learning_rate(self.optimizer_G, self.gamma)
                        step+=1
               
                    # Visualize sample
                    if self.iters % self.visualize_per_iter == 0:
                        masked_imgs = imgs * (1 - masks) + masks
                        
                        img_list = [imgs, masked_imgs, first_out, second_out, second_out_wholeimg]
                        #name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
                        filename = f"{self.epoch}_{str(self.iters)}"
                        self.validate(self.cfg.sample_folder, filename , img_list)

                    self.iters += 1

        except KeyboardInterrupt:
                torch.save({
                    'D': self.model_D.state_dict(),
                    'G': self.model_G.state_dict(),
                }, os.path.join(self.cfg.checkpoint_path, f"model_{self.epoch}_{self.iters}.pth"))
                    
        