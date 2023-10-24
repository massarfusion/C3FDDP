from collections import OrderedDict
import math

import torch.nn as nn
from timm import create_model
import torch
from torchvision import models
import torchvision
import  timm
import torch.nn.functional as F
import pdb

from MultiStageMerge import MultiStageMerging

class DDP(nn.Module):
    def __init__(self, pretrained=True):
        super(DDP, self).__init__()
        self.convnext = create_model('convnext_tiny', pretrained=pretrained, features_only=True)
        self.fpn=torchvision.ops.FeaturePyramidNetwork(in_channels_list=[96, 192, 384, 768], out_channels=256)
        self.multiStageMerger=MultiStageMerging(in_channels=[256, 256, 256, 256],out_channels=256,kernel_size=1,)
        self.beforeUnet=nn.Conv2d(in_channels=256+1, out_channels=256, kernel_size=1,stride=1, padding=0)
        self.unet=...
        self.sample_range=[0, 0.999]

        
        
    
    def forward(self, x, gt_map=None, train=True):
        if train:
            assert gt_map!=None, "If in training, ground truth map must be fed."
            return self.forward_train(x, gt_map)
        else:
            return self.forward_test(x)
         
    def forward_train(self, x, gt_map):
        ho, wo = x.shape[-2], x.shape[-1]
        stage1 = self.convnext(x)
        
        temp = OrderedDict()# Because it takes OrderedDict
        for i, datum in enumerate(stage1):
            temp["{}".format(i)] = datum
        stage2 = self.fpn(temp)
        stage2 = [v for k, v in stage2.items()]
        
        extracted_feature = self.multiStageMerger(stage2)[0]
        # Feature map, from original RGB Image, should be shaped as (batch, 256, H/4, W/4)
        
        # Check if we are receiving expected size
        batch, c, h, w, device, = *extracted_feature.shape, extracted_feature.device
        assert gt_map.shape==(batch,1,ho,wo), "Something wrong with data loading, ground truth map not in correct shape"
        # if gt_map.shape != (batch, 1, ho, wo) and gt_map.numel() == ho * wo:
        #     gt_map = gt_map.reshape((batch, 1, ho, wo)).contiguous()
        
        # To bring gt map down to same size as feature map
        gt_map = F.interpolate(input=gt_map, size=extracted_feature.shape[2:], mode='bilinear')
        
        # cook up noise and add to ground truth map(following cosine noise schedule)
        eps = torch.randn_like(gt_map)
        t = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0], self.sample_range[1])  # [batch]
        while t.ndim<gt_map.ndim:
            t=t.unsqueeze(-1)
        gt_map_corrupted = torch.sqrt(self.gamma(t)) * gt_map + torch.sqrt(1 - self.gamma(t)) * eps
        
        # "Fuse" together 1.extracted feature map(from RGB image) and 2.corrupted noise
        feat = torch.cat([extracted_feature, gt_map_corrupted], dim=1)  # (batch, 256+1, H/4, W/4)
        feat = self.beforeUnet(feat)  # (batch, 256, H/4, W/4)
        
        # get predicted x_0, we directly calculate loss on this, with ground truth density map.
        pred_map=self.unet(feat) # (batch, 1, H/4, W/4)
        return pred_map
        
    
    def forward_test(self, x):
        pass
    
    def gamma(self, t, ns=0.0002, ds=0.00025):
        return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2