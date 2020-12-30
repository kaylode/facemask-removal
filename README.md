# Inpainting the Masked Face using Gated Convolution + PatchGAN (Pytorch)

## Environments
- Windows 10
- Pytorch 1.6

## Pipeline:
- Data preparation:
  - Download CelebA dataset then crop the image but keeps while keeping ratio with [here](https://github.com/LynnHo/HD-CelebA-Cropper)
  - Create synthesis facemask segmentation dataset with [here](https://github.com/aqeelanwar/MaskTheFace)
 
- Edit configs on both ***segm.yaml*** and ***facemask.yaml***
- Train segmentation model in ***unet_trainer.py***
- Train inpainting model in ***trainer.py***

## Train facemask segmentation

```
python train.py segm --resume=<resume checkpoint>
```

## Train facemask inpainting

```
python train.py facemask --resume=<resume checkpoint>
```

## Results:
| | |
|:-------------------------:|:-------------------------:|
|<img width="900" alt="screen" src="sample/results1.png"> | <img width="900" alt="screen" src="sample/results2.png"> |

<p align="center">
Inpainting results on Masked CelebA-512 (from left to right: FaceMasked - Segmented - Inpainted - Ground Truth)
</p>

| | |
|:-------------------------:|:-------------------------:|
|<img width="900" alt="screen" src="sample/results3.png"> | <img width="900" alt="screen" src="sample/reesults4.png"> |

<p align="center">
Free-Form Inpainting results on Places365-256 (from left to right: Ground Truth - Masked - Inpainted )
</p>

## Paper References:
- Idea and training process from [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/abstract/document/9019697)
- Base model from [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

## Code References
- Generator from https://github.com/zhaoyuzhi/deepfillv2
- Discriminator from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- https://github.com/avalonstrel/GatedConvolution_pytorch
- https://github.com/LynnHo/HD-CelebA-Cropper
- https://github.com/aqeelanwar/MaskTheFace
