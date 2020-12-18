import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.data as data
import functools
from torchvision.models import vgg19, vgg16

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation = 'lrelu', norm = 'in'):
        super(GatedConv2d, self).__init__()
        self.pad = nn.ZeroPad2d(padding)
        if norm is not None:
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
            
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        
       
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, norm=None, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, norm=norm)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x


class GatedGenerator(nn.Module):
    def __init__(self, in_channels=4, latent_channels=64, out_channels=3):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 7, 1, 3, norm = None),
            GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1),
            GatedConv2d(latent_channels, out_channels, 7, 1, 3, activation = 'tanh', norm = None)
        )
        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 7, 1, 3, norm = None),
            GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1),
            GatedConv2d(latent_channels, out_channels, 7, 1, 3, activation = 'tanh', norm = None)
        )
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
     
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), 1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                       # out: [B, 3, H, W]
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat((second_masked_img, mask), 1)     # in: [B, 4, H, W]
        second_out = self.refinement(second_in)                 # out: [B, 3, H, W]
        return first_out, second_out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PerceptualNet(nn.Module):
    # https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    def __init__(self, name = "vgg19", resize=True):
        super(PerceptualNet, self).__init__()
        blocks = []
        if name == "vgg19":
            blocks.append(vgg19(pretrained=True).features[:4].eval())
            blocks.append(vgg19(pretrained=True).features[4:9].eval())
            blocks.append(vgg19(pretrained=True).features[9:16].eval())
            blocks.append(vgg19(pretrained=True).features[16:23].eval())
        elif name == "vgg16":
            blocks.append(vgg16(pretrained=True).features[:4].eval())
            blocks.append(vgg16(pretrained=True).features[4:9].eval())
            blocks.append(vgg16(pretrained=True).features[9:16].eval())
            blocks.append(vgg16(pretrained=True).features[16:23].eval())
        else:
            assert "wrong model name"
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, inputs, targets):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = (inputs-self.mean) / self.std
        targets = (targets-self.mean) / self.std
        if self.resize:
            inputs = self.transform(inputs, mode='bilinear', size=(512, 512), align_corners=False)
            targets = self.transform(targets, mode='bilinear', size=(512, 512), align_corners=False)
        loss = 0.0
        x = inputs
        y = targets
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss



