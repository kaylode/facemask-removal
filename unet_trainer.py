import os
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image


from models import UNetSemantic
from losses import DiceLoss
from datasets import FacemaskSegDataset
from metrics import *

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

def load_checkpoint(model, path):
    state = torch.load(path,map_location='cpu')
    model.load_state_dict(state)
    print('Loaded checkpoint successfully')

class UNetTrainer():
    def __init__(self, args, cfg):
        
        if args.resume is not None:
            epoch, iters = get_epoch_iters(args.resume)
        else:
            epoch = 0
            iters = 0

        self.cfg = cfg
        self.step_iters = cfg.step_iters
        self.gamma = cfg.gamma
        self.visualize_per_iter = cfg.visualize_per_iter
        self.print_per_iter = cfg.print_per_iter
        self.save_per_iter = cfg.save_per_iter
        
        self.start_iter = iters
        self.iters = 0
        self.num_epochs = cfg.num_epochs
        self.device = torch.device('cuda:1' if cfg.cuda else 'cpu')

        trainset = FacemaskSegDataset(cfg)
        valset = FacemaskSegDataset(cfg, train=False)
   
        self.trainloader = data.DataLoader(
            trainset, 
            batch_size=cfg.batch_size,
            num_workers = cfg.num_workers,
            pin_memory = True, 
            shuffle=True,
            collate_fn = trainset.collate_fn)

        self.valloader = data.DataLoader(
            valset, 
            batch_size=cfg.batch_size,
            num_workers = cfg.num_workers,
            pin_memory = True, 
            shuffle=True,
            collate_fn = valset.collate_fn)

        self.epoch = int(self.start_iter / len(self.trainloader))
        self.iters = self.start_iter
        self.num_iters = (self.num_epochs+1) * len(self.trainloader)

        self.model = UNetSemantic().to(self.device)
        self.criterion_dice = DiceLoss()
        self.criterion_bce = nn.BCELoss()

        if args.resume is not None:
            load_checkpoint(self.model, args.resume)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def validate(self, sample_folder, sample_name, img_list):
        save_img_path = os.path.join(sample_folder, sample_name+'.png') 
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)

        # imgs shape: Bx5xCxWxH

        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow= 3)
        print(f"Save image to {save_img_path}")


    def train_epoch(self):
        self.model.train()
        running_loss = {
                'DICE': 0,
                'BCE':0,
                 'T': 0,
            }
        running_time = 0

        for idx, batch in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            inputs = batch['imgs'].to(self.device)
            targets = batch['masks'].to(self.device)
            
            start_time = time.time()
            
            outputs = self.model(inputs)

            loss_bce = self.criterion_bce(outputs, targets)
            loss_dice = self.criterion_dice(outputs, targets)
            loss = loss_bce + loss_dice
            loss.backward()
            self.optimizer.step()
            
            end_time = time.time()
            
            running_loss['T'] += loss.item()
            running_loss['DICE'] += loss_dice.item()
            running_loss['BCE'] += loss_bce.item()
            running_time += end_time-start_time

            if self.iters % self.print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                running_time = np.round(running_time, 5)
                print('[{}/{}][{}/{}] || {} || Time: {}s'.format(self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string,  running_time))
                running_time = 0
                running_loss = {
                    'DICE': 0,
                    'BCE':0, 
                    'T': 0,
                }


            if self.iters % self.save_per_iter  == 0:
                save_path = os.path.join(
                        self.cfg.checkpoint_path, 
                        f"model_segm_{self.epoch}_{self.iters}.pth")
                torch.save(self.model.state_dict(),save_path)
                print(f'Save model at {save_path}')
            self.iters +=1

    def validate_epoch(self):
        #Validate
        
        self.model.eval()
        metrics = [DiceScore(1), PixelAccuracy(1)]
        running_loss = {
            'DICE': 0,
            'BCE':0,
             'T': 0,
        }

        running_time = 0
        print('=============================EVALUATION===================================')
        with torch.no_grad():
            start_time = time.time()
            for idx, batch in enumerate(tqdm(self.valloader)):
                
                inputs = batch['imgs'].to(self.device)
                targets = batch['masks'].to(self.device)
                outputs = self.model(inputs)
                loss_bce = self.criterion_bce(outputs, targets)
                loss_dice = self.criterion_dice(outputs, targets)
                loss = loss_bce + loss_dice
                running_loss['T'] += loss.item()
                running_loss['DICE'] += loss_dice.item()
                running_loss['BCE'] += loss_bce.item()
                for metric in metrics:
                    metric.update(outputs.cpu(), targets.cpu())

            end_time = time.time()
            running_time += (end_time - start_time)
            running_time = np.round(running_time, 5)
            for key in running_loss.keys():
                running_loss[key] /= len(self.valloader)
                running_loss[key] = np.round(running_loss[key], 5)

            loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
            
            print('[{}/{}] || Validation || {} || Time: {}s'.format(self.epoch, self.num_epochs, loss_string, running_time))
            for metric in metrics:
                print(metric)
            print('==========================================================================')
        

    def fit(self):
        try: 
            for epoch in range(self.epoch, self.num_epochs+1): 
                self.epoch = epoch
                self.train_epoch()
                self.validate_epoch()
        except KeyboardInterrupt:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.cfg.checkpoint_path, 
                        f"model_segm_{self.epoch}_{self.iters}.pth"))
                print('Model saved!')
                    
        