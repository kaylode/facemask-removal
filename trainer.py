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

from models import *
from datasets import InpaintDataset

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


class Trainer():
    def __init__(self, args, cfg):
        
        if args.resume is not None:
            epoch, iters = get_epoch_iters(args.resume)
        else:
            epoch = 0
            iters = 0

        self.cfg = cfg
        self.print_per_iter = cfg.print_per_iter
        self.save_per_iter = cfg.save_per_iter
        self.epoch = epoch
        self.start_iter = iters
        self.iters = 0
        self.num_epochs = cfg.num_epochs
        self.device = torch.device('cuda' if cfg.cuda else 'cpu')

        trainset = Places365Dataset(img_dir=cfg.img_dir)

        self.trainloader = data.DataLoader(
            trainset, 
            batch_size=cfg.batch_size,
            num_workers = cfg.num_workers,
            pin_memory = True, 
            collate_fn = trainset.collate_fn)

        self.num_iters = (self.num_epochs+1) * len(self.trainloader)

        self.model_G = GatedGenerator().to(self.device)
        self.model_D = NLayerDiscriminator(cfg.d_num_layers, use_sigmoid=True).to(self.device)
        self.model_P = vgg19(pretrained=True).features.to(self.device)[:-2]
        self.model_P.eval()

        self.criterion_adv = GANLoss()
        self.criterion_rec = nn.MSELoss()
        self.criterion_per = nn.L1Loss()

        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=cfg.lr)
        self.optimizer_G = torch.optim.Adam(self.model_G.parameters(), lr=cfg.lr)

        self.scheduler_D = StepLR(self.optimizer_D, step_size=cfg.step_size, gamma=cfg.gamma)
        self.scheduler_G = StepLR(self.optimizer_G, step_size=cfg.step_size, gamma=cfg.gamma)

    def validate(self, sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
        for i in range(len(img_list)):
            img = img_list[i]
            # Recover normalization: * 255 because last layer is sigmoid activated
            img = img * 255
            # Process img_copy and do not destroy the data of img
            img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
            img_copy = np.clip(img_copy, 0, pixel_max_cnt)
            img_copy = img_copy.astype(np.uint8)
            # Save to certain path
            save_img_name = sample_name + '_' + name_list[i] + '.png'
            save_img_path = os.path.join(sample_folder, save_img_name)
            cv2.imwrite(save_img_path, img_copy)

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
        try:
            for epoch in range(self.epoch, self.num_epochs):
                self.epoch = epoch
                for i, batch in enumerate(self.trainloader):
                    start_time = time.time()
                    imgs = batch['imgs'].to(self.device)
                    masks = batch['masks'].to(self.device)

                    first_out, second_out = self.model_G(imgs, masks)

                    first_out_wholeimg = imgs * (1 - masks) + first_out * masks     
                    second_out_wholeimg = imgs * (1 - masks) + second_out * masks

                    # Train discriminator
                    self.optimizer_D.zero_grad()

                    fake_D = self.model_D(second_out_wholeimg.detach())
                    real_D = self.model_D(imgs)

                    loss_fake_D = self.criterion_adv(fake_D, target_is_real=False)
                    loss_real_D = self.criterion_adv(real_D, target_is_real=True)

                    loss_D = (loss_fake_D + loss_real_D) * 0.5

                    loss_D.backward()
                    self.optimizer_D.step()


                    # Train Generator
                    self.optimizer_G.zero_grad()

                    fake_D = self.model_D(second_out_wholeimg)
                    loss_G = self.criterion_adv(fake_D, target_is_real=True)

                    # Reconstruction loss
                    loss_rec_1 = self.criterion_rec(first_out_wholeimg, imgs)
                    loss_rec_2 = self.criterion_rec(second_out_wholeimg, imgs)

                    # Perceptual loss
                    img_featuremaps = self.model_P(imgs)                          
                    second_out_wholeimg_featuremaps = self.model_P(second_out_wholeimg)
                    loss_P = self.criterion_per(second_out_wholeimg_featuremaps, img_featuremaps)

                    loss = self.cfg.lambda_G * loss_G + self.cfg.lambda_rec_1 * loss_rec_1 + self.cfg.lambda_rec_2 * loss_rec_2 + self.cfg.lambda_per * loss_P
                    loss.backward()
                    self.optimizer_G.step()

                    end_time = time.time()

                    # Visualize number
                    running_time += (end_time - start_time)
                    running_loss['D'] += loss_D.item()
                    running_loss['G'] += (self.cfg.lambda_G * loss_G.item())
                    running_loss['P'] += (self.cfg.lambda_per * loss_P.item())
                    running_loss['R_1'] += (self.cfg.lambda_rec_1 * loss_rec_1.item())
                    running_loss['R_2'] += (self.cfg.lambda_rec_2 * loss_rec_2.item())
                    running_loss['T'] += loss.item()
                    self.iters = self.start_iter + len(self.trainloader)*self.epoch + i + 1

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
                    
                    # Visualize sample
                    if self.iters % self.visualize_per_iter == 0:
                        masked_imgs = imgs * (1 - masks) + masks
                        masks = torch.cat((masks, masks, masks), 1)
                        img_list = [imgs, masks, masked_imgs, first_out, second_out]
                        name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
                        filename = f"{self.epoch}_{str(self.iters)}"
                        self.validate(self.cfg.sample_folder, filename , img_list, name_list, pixel_max_cnt = 255)
        
                self.scheduler_D.step()
                self.scheduler_G.step()

        except KeyboardInterrupt:
                torch.save({
                    'D': self.model_D.state_dict(),
                    'G': self.model_G.state_dict(),
                }, os.path.join(self.cfg.checkpoint_path, f"model_{self.epoch}_{self.iters}.pth"))
                    
        