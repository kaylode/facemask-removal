# Impainting the Masked Face using Gated Convolution (Pytorch)

## Environments
- Windows 10
- Pytorch 1.6

## Pipeline:
- Data preparation:
  - Download CelebA dataset then crop the image but keeps while keeping ratio with [here](https://github.com/LynnHo/HD-CelebA-Cropper)
  - Create synthesis facemask segmentation dataset with [here](https://github.com/aqeelanwar/MaskTheFace)
 
- Edit configs on both ***segm.yaml*** and ***facemask.yaml***
- Train segmentation model in ***unet_trainer.py***
- Train impainting model in ***trainer.py***

## Train facemask segmentation

```
python train.py segm --resume=<resume checkpoint>
```

## Train facemask impainting

```
python train.py facemask --resume=<resume checkpoint>
```

## Results:
| | |
|:-------------------------:|:-------------------------:|
|<img width="900" alt="screen" src="sample/results1.png"> | <img width="900" alt="screen" src="sample/results2.png"> |

<p align="center">
Impainting results (from left to right: Masked - Segmented - Impainting - Ground Truth)
</p>

## Paper References:
- Idea and training process from [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/abstract/document/9019697)
- Base model from [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

## Code References
- https://github.com/zhaoyuzhi/deepfillv2
- https://github.com/avalonstrel/GatedConvolution_pytorch
- https://github.com/LynnHo/HD-CelebA-Cropper
- https://github.com/aqeelanwar/MaskTheFace
