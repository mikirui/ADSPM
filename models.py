import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockBatchNorm(nn.Module):
    """Residual Block with batch normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockBatchNorm, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class WarpGenerator256(nn.Module):
    """2 stage Generator in 256x256 scale"""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, repeat_num2=3, coff=1.0, image_size=128):
        super(WarpGenerator256, self).__init__()
        self.coff = coff
        self.image_size = image_size

        ### 128 scale
        ## Spontaneous Motion Module
        # encoder
        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.encoding = nn.Sequential(*layers)
        encoding_dim = curr_dim
        
        # decoder for motion prediction
        layers2 = []
        curr_dim = encoding_dim
        for i in range(2):
            layers2.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers2.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers2.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers2.append(nn.Conv2d(curr_dim, 2, kernel_size=7, stride=1, padding=3, bias=False))
        layers2.append(nn.Tanh())
        self.flow_pred = nn.Sequential(*layers2)
        
        # decoder for attention mask prediction
        layers3 = []
        curr_dim = encoding_dim
        for i in range(2):
            layers3.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers3.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers3.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers3.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        layers3.append(nn.Sigmoid())
        self.att_mask = nn.Sequential(*layers3)
        
        ## Refinement module
        layers4 = []
        layers4.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers4.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers4.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(repeat_num2):
            layers4.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        layers4.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers4.append(nn.Tanh())
        self.refine = nn.Sequential(*layers4)

        ### 256 scale
        # motion refine layers
        flow_refine = []
        flow_refine.append(nn.Conv2d(2, conv_dim//2, kernel_size=3, stride=1, padding=1, bias=False))
        flow_refine.append(nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True))
        flow_refine.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        flow_refine.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        flow_refine.append(nn.Conv2d(curr_dim, 2, kernel_size=3, stride=1, padding=1, bias=False))
        flow_refine.append(nn.Tanh())
        self.flow_refine = nn.Sequential(*flow_refine)
        
        # mask refine layers
        mask_refine = []
        mask_refine.append(nn.Conv2d(1, conv_dim // 2, kernel_size=3, stride=1, padding=1, bias=False))
        mask_refine.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        mask_refine.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        mask_refine.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        mask_refine.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        mask_refine.append(nn.Tanh())
        self.mask_refine = nn.Sequential(*mask_refine)
        
        # residual refine layers
        app_refine = []
        app_refine.append(nn.Conv2d(3, conv_dim // 2, kernel_size=3, stride=1, padding=1, bias=False))
        app_refine.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        app_refine.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        app_refine.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        app_refine.append(nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=False))
        app_refine.append(nn.Tanh())
        self.app_refine = nn.Sequential(*app_refine)

    def warp(self, x, flow, mode='bilinear', padding_mode='zeros', coff=1.0):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
        grid_x = grid + 2 * flow * coff
        warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
        return warp_x

    def forward(self, x, c, higher=False):
        c = c.view(c.size(0), c.size(1), 1, 1)
        if higher == False:
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x_cond = torch.cat([x, c], dim=1)
            feat = self.encoding(x_cond)
            flow = self.flow_pred(feat)
            mask = self.att_mask(feat)
            flow = flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
            warp_x = self.warp(x, flow, coff=self.coff)
            refine_x = self.refine(warp_x) * mask
            refine_warp_x = torch.clamp(warp_x + refine_x, min=-1.0, max=1.0)
            return refine_warp_x, warp_x, flow, mask
        else:
            # low resolution stage
            low_x = F.upsample(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
            c = c.repeat(1, 1, low_x.size(2), low_x.size(3))
            x_cond = torch.cat([low_x, c], dim=1)
            feat = self.encoding(x_cond)
            low_flow = self.flow_pred(feat)
            low_mask = self.att_mask(feat)
            low_warp_x = self.warp(low_x, low_flow.permute(0, 2, 3, 1), coff=self.coff)
            low_refine_x = self.refine(low_warp_x) * low_mask
            low_refine_warp_x = torch.clamp(low_warp_x + low_refine_x, min=-1.0, max=1.0)

            # high resolution stage
            high_flow = F.upsample(low_flow, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True) * self.coff
            high_flow = high_flow + self.flow_refine(high_flow) * self.coff
            high_flow = high_flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
            high_warp_x = self.warp(x, high_flow, coff=1.0)

            high_mask = F.upsample(low_mask, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            high_mask = high_mask + self.mask_refine(high_mask)
            high_mask = torch.clamp(high_mask, min=0.0, max=1.0)

            high_refine_x = F.upsample(low_refine_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            high_refine_x = high_refine_x + self.app_refine(high_refine_x)
            high_refine_x = high_refine_x * high_mask
            high_refine_warp_x = torch.clamp(high_warp_x + high_refine_x, min=-1.0, max=1.0)
            
            return high_refine_warp_x, low_refine_warp_x, high_warp_x, low_warp_x, high_flow, high_mask


class WarpGenerator512(nn.Module):
    """3 stage Generator in 512x512 scale"""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, repeat_num2=3, coff=1.0, image_size=128, mid_image_size=256, high_image_size=512):
        super(WarpGenerator512, self).__init__()
        self.coff = coff
        self.image_size = image_size
        self.mid_image_size = mid_image_size
        self.high_image_size = high_image_size

        ### 128 scale
        ## Spontaneous motion module
        # encoder
        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.encoding = nn.Sequential(*layers)
        encoding_dim = curr_dim
        
        # decoder for motion prediction
        layers2 = []
        curr_dim = encoding_dim
        for i in range(2):
            layers2.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers2.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers2.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers2.append(nn.Conv2d(curr_dim, 2, kernel_size=7, stride=1, padding=3, bias=False))
        layers2.append(nn.Tanh())
        self.flow_pred = nn.Sequential(*layers2)
        
        # decoder for attention mask prediction
        layers3 = []
        curr_dim = encoding_dim
        for i in range(2):
            layers3.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers3.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers3.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers3.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        layers3.append(nn.Sigmoid())
        self.att_mask = nn.Sequential(*layers3)
        
        ## Refinement module
        layers4 = []
        layers4.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers4.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers4.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(repeat_num2):
            layers4.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        layers4.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers4.append(nn.Tanh())
        self.refine = nn.Sequential(*layers4)

        ### 256 scale
        # motion refine layers
        flow_refine = []
        flow_refine.append(nn.Conv2d(2, conv_dim//2, kernel_size=3, stride=1, padding=1, bias=False))
        flow_refine.append(nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True))
        flow_refine.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        flow_refine.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        flow_refine.append(nn.Conv2d(curr_dim, 2, kernel_size=3, stride=1, padding=1, bias=False))
        flow_refine.append(nn.Tanh())
        self.flow_refine = nn.Sequential(*flow_refine)
        # mask refine layers
        mask_refine = []
        mask_refine.append(nn.Conv2d(1, conv_dim // 2, kernel_size=3, stride=1, padding=1, bias=False))
        mask_refine.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        mask_refine.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        mask_refine.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        mask_refine.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        mask_refine.append(nn.Tanh())
        self.mask_refine = nn.Sequential(*mask_refine)
        # residual refine layers
        app_refine = []
        app_refine.append(nn.Conv2d(3, conv_dim // 2, kernel_size=3, stride=1, padding=1, bias=False))
        app_refine.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        app_refine.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        app_refine.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        app_refine.append(nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=False))
        app_refine.append(nn.Tanh())
        self.app_refine = nn.Sequential(*app_refine)

        ### 512 scale
        # motion refine layers
        flow_refine512 = []
        flow_refine512.append(nn.Conv2d(2, conv_dim // 2, kernel_size=7, stride=1, padding=3, bias=False))
        flow_refine512.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        flow_refine512.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        flow_refine512.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        flow_refine512.append(nn.Conv2d(curr_dim, 2, kernel_size=7, stride=1, padding=3, bias=False))
        flow_refine512.append(nn.Tanh())
        self.flow_refine512 = nn.Sequential(*flow_refine512)
        # mask refine layers
        mask_refine512 = []
        mask_refine512.append(nn.Conv2d(1, conv_dim // 2, kernel_size=7, stride=1, padding=3, bias=False))
        mask_refine512.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        mask_refine512.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        mask_refine512.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        mask_refine512.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        mask_refine512.append(nn.Tanh())
        self.mask_refine512 = nn.Sequential(*mask_refine512)
        # residual refine layers
        app_refine512 = []
        app_refine512.append(nn.Conv2d(3, conv_dim // 2, kernel_size=7, stride=1, padding=3, bias=False))
        app_refine512.append(nn.InstanceNorm2d(conv_dim // 2, affine=True, track_running_stats=True))
        app_refine512.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim // 2
        app_refine512.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        app_refine512.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        app_refine512.append(nn.Tanh())
        self.app_refine512 = nn.Sequential(*app_refine512)

    def warp(self, x, flow, mode='bilinear', padding_mode='zeros', coff=1.0):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
        grid_x = grid + 2 * flow * coff
        warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
        return warp_x

    def forward(self, x, c, stage=0):
        c = c.view(c.size(0), c.size(1), 1, 1)
        if stage == 0:
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x_cond = torch.cat([x, c], dim=1)
            feat = self.encoding(x_cond)
            flow = self.flow_pred(feat)
            mask = self.att_mask(feat)
            flow = flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
            warp_x = self.warp(x, flow, coff=self.coff)
            refine_x = self.refine(warp_x) * mask
            refine_warp_x = torch.clamp(warp_x + refine_x, min=-1.0, max=1.0)
            return refine_warp_x, warp_x, flow, mask
        elif stage == 1:
            # 128 scale stage
            low_x = F.upsample(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
            c = c.repeat(1, 1, low_x.size(2), low_x.size(3))
            x_cond = torch.cat([low_x, c], dim=1)
            feat = self.encoding(x_cond)
            low_flow = self.flow_pred(feat)
            low_mask = self.att_mask(feat)
            low_warp_x = self.warp(low_x, low_flow.permute(0, 2, 3, 1), coff=self.coff)
            low_refine_x = self.refine(low_warp_x) * low_mask
            low_refine_warp_x = torch.clamp(low_warp_x + low_refine_x, min=-1.0, max=1.0)

            # 256 scale stage
            high_flow = F.upsample(low_flow, size=(x.size(2), x.size(3)), mode='bilinear',
                                  align_corners=True) * self.coff
            high_flow = high_flow + self.flow_refine(high_flow) * self.coff
            high_flow = high_flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
            high_warp_x = self.warp(x, high_flow, coff=1.0)
            high_mask = F.upsample(low_mask, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            high_mask = high_mask + self.mask_refine(high_mask)
            high_mask = torch.clamp(high_mask, min=0.0, max=1.0)
            high_refine_x = F.upsample(low_refine_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            high_refine_x = high_refine_x + self.app_refine(high_refine_x)
            high_refine_x = high_refine_x * high_mask
            high_refine_warp_x = torch.clamp(high_warp_x + high_refine_x, min=-1.0, max=1.0)

            return high_refine_warp_x, low_refine_warp_x, high_warp_x, low_warp_x, high_flow, high_mask
        else:
            # 128 scale stage
            low_x = F.upsample(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
            c = c.repeat(1, 1, low_x.size(2), low_x.size(3))
            x_cond = torch.cat([low_x, c], dim=1)
            feat = self.encoding(x_cond)
            low_flow = self.flow_pred(feat)
            low_mask = self.att_mask(feat)
            low_warp_x = self.warp(low_x, low_flow.permute(0, 2, 3, 1), coff=self.coff)
            low_refine_x = self.refine(low_warp_x) * low_mask
            low_refine_warp_x = torch.clamp(low_warp_x + low_refine_x, min=-1.0, max=1.0)

            # 256 scale stage
            x256 = F.upsample(x, size=(self.mid_image_size, self.mid_image_size), mode='bilinear', align_corners=True)
            high_flow = F.upsample(low_flow, size=(self.mid_image_size, self.mid_image_size), mode='bilinear', align_corners=True) * self.coff
            high_flow = high_flow + self.flow_refine(high_flow) * self.coff
            high_warp_x = self.warp(x256, high_flow.permute(0, 2, 3, 1), coff=1.0)
            high_mask = F.upsample(low_mask, size=(self.mid_image_size, self.mid_image_size), mode='bilinear', align_corners=True)
            high_mask = high_mask + self.mask_refine(high_mask)
            high_mask = torch.clamp(high_mask, min=0.0, max=1.0)
            high_refine_x = F.upsample(low_refine_x, size=(self.mid_image_size, self.mid_image_size), mode='bilinear', align_corners=True)
            high_refine_x = high_refine_x + self.app_refine(high_refine_x)
            high_refine_x = high_refine_x * high_mask
            high_refine_warp_x = torch.clamp(high_warp_x + high_refine_x, min=-1.0, max=1.0)

            # 512 scale stage
            high_flow512 = F.upsample(high_flow, size=(self.high_image_size, self.high_image_size), mode='bilinear', align_corners=True)
            high_flow512 = high_flow512 + self.flow_refine512(high_flow512) * self.coff
            high_flow512 = high_flow512.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
            high_warp_x512 = self.warp(x, high_flow512, coff=1.0)
            high_mask512 = F.upsample(high_mask, size=(self.high_image_size, self.high_image_size), mode='bilinear', align_corners=True)
            high_mask512 = high_mask512 + self.mask_refine512(high_mask512)
            high_mask512 = torch.clamp(high_mask512, min=0.0, max=1.0)
            high_refine_x512 = F.upsample(high_refine_x, size=(self.high_image_size, self.high_image_size), mode='bilinear', align_corners=True)
            high_refine_x512 = high_refine_x512 + self.app_refine512(high_refine_x512)
            high_refine_x512 = high_refine_x512 * high_mask512
            high_refine_warp_x512 = torch.clamp(high_warp_x512 + high_refine_x512, min=-1.0, max=1.0)

            return high_refine_warp_x512, high_refine_warp_x, low_refine_warp_x, \
                   high_warp_x512, high_warp_x, low_warp_x, \
                   high_flow512, high_mask512


class Discriminator(nn.Module):
    """Discriminator for 128x128 scale"""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class Discriminator_deep(nn.Module):
    """Discriminator for 256x256 scale"""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_deep, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num - 1):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

class Discriminator_deep512(nn.Module):
    """Discriminator for 512x512 scale"""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_deep512, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num - 2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

class uv_classifier(nn.Module):
    """uv classifier"""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=3, coff=0.1):
        super(uv_classifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3 + c_dim + 2, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))
        self.main = nn.Sequential(*layers)
        self.coff = coff

    def forward(self, uv, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        uv = uv.permute(0, 3, 1, 2) * self.coff
        uv_cond = torch.cat([uv, x, c], dim=1)

        out_cls = self.main(uv_cond)
        return out_cls.view(out_cls.size(0), out_cls.size(1))

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]