import os
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm

class Places365Dataset(data.Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.cfg = cfg
        self.load_images()
        
    def load_images(self):
        self.fns =[]
        idx = 0
        img_paths = os.listdir(self.root_dir)
        for cls_id in img_paths:
            paths = os.path.join(self.root_dir, cls_id)
            file_paths = os.listdir(paths)
            for img_name in file_paths:
                filename = os.path.join(paths, img_name)
                self.fns.append(filename)

    def __getitem__(self, index):
        img_path = self.fns[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        
        mask = self.random_ff_mask(
            shape = self.cfg.img_size, 
            max_angle = self.cfg.max_angle, 
            max_len = self.cfg.max_len, 
            max_width = self.cfg.max_width, 
            times = self.cfg.times)
        
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()

        return img, mask
    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }
    
    def __len__(self):
        return len(self.fns)
    
    def random_ff_mask(self, shape = 256 , max_angle = 4, max_len = 50, max_width = 20, times = 15):
            """Generate a random free form mask with configuration.
            Args:
                config: Config should have configuration including IMG_SHAPES,
                    VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
            Returns:
                tuple: (top, left, height, width)
            """
            height = shape
            width = shape
            mask = np.zeros((height, width), np.float32)
            times = np.random.randint(10, times)
            for i in range(times):
                start_x = np.random.randint(width)
                start_y = np.random.randint(height)
                for j in range(1 + np.random.randint(5)):
                    angle = 0.01 + np.random.randint(max_angle)
                    if i % 2 == 0:
                        angle = 2 * 3.1415926 - angle
                    length = 10 + np.random.randint(max_len)
                    brush_w = 5 + np.random.randint(max_width)
                    end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                    end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                    cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                    start_x, start_y = end_x, end_y
            return mask.reshape((1, ) + mask.shape).astype(np.float32)


class FacemaskDataset(data.Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.cfg = cfg

        self.mask_folder = os.path.join(self.root_dir, 'celeba512_30k_binary')
        self.img_folder = os.path.join(self.root_dir, 'celeba512_30k')
        self.load_images()
        
    def load_images(self):
        self.fns = []
        idx = 0
        img_paths = sorted(os.listdir(self.img_folder))
        for img_name in img_paths:
            mask_name = img_name.split('.')[0]+'_binary.jpg'
            img_path = os.path.join(self.img_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name)
            if os.path.isfile(mask_path): 
                self.fns.append([img_path, mask_path])

    def __getitem__(self, index):
        img_path, mask_path = self.fns[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        
        
        mask = cv2.imread(mask_path, 0)
        
        mask[mask>0]=1.0
        mask = np.expand_dims(mask, axis=0)
    
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        return img, mask
    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }
    
    def __len__(self):
        return len(self.fns)