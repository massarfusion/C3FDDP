from collections import OrderedDict
import math

import torch.nn as nn
from timm import create_model
import torch
from torch import optim
import torchvision
import torch.nn.functional as F
import pdb
import pytorch_lightning as pl
from einops import rearrange, reduce, repeat

from .MultiStageMerge import MultiStageMerging
from .Transformer import UViT

class DDP(pl.LightningModule):
    def __init__(self, pretrained=True):
        super(DDP, self).__init__()
        self.convnext = create_model('convnext_tiny', pretrained=pretrained, features_only=True)
        self.fpn=torchvision.ops.FeaturePyramidNetwork(in_channels_list=[96, 192, 384, 768], out_channels=256)
        self.multiStageMerger=MultiStageMerging(in_channels=[256, 256, 256, 256],out_channels=256,kernel_size=1,)
        self.beforeUnet=nn.Conv2d(in_channels=256+1, out_channels=256, kernel_size=1,stride=1, padding=0)
        self.unet=UViT(
            img_size=(192, 256),
            patch_size=4,
            in_chans=256,
            embed_dim=768,
            depth=6,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            mlp_time_embed=False,
            num_classes=-1,
            use_checkpoint=True,
            conv=True,
            skip=True,
        )
        self.sample_range=[0, 999]
        self.timesteps_upper_limit=1000
        self.loss_fn = nn.MSELoss()
        self.LR=1e-3
        # Sampling Configs. How many steps in total and how much steps to take one time.
        self.timesteps_inference=3
        self.timestep_step_length =1
        self.ddim=True
        
        
        
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        loss, scores, y = self._forward_step(x,y)
        self.log_dict({"loss":loss})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_map = self._inference_step(x)
        assert pred_map.shape == y.shape
        mse = nn.MSELoss()(pred_map, y)
        mae = nn.L1Loss()(pred_map, y)
        self.log_dict({"val_mae": mae, "val_mse":mse})
        return mae
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred_map = self._inference_step(x)
        assert pred_map.shape == y.shape
        mse = nn.MSELoss()(pred_map, y)
        mae = nn.L1Loss()(pred_map, y)
        self.log_dict({"test_mae": mae, "test_mse": mse})
        return mae
    def predict_step(self, batch, batch_idx: int):
        x, y = batch
        pred_map = self._inference_step(x)
        return pred_map
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.LR)
    
    def _forward_step(self, x, gt_map):
        ho, wo = x.shape[-2], x.shape[-1]
        # stage1 = self.convnext(x)
        #
        # temp = OrderedDict()  # Because it takes OrderedDict
        # for i, datum in enumerate(stage1):
        #     temp["{}".format(i)] = datum
        # stage2 = self.fpn(temp)
        # stage2 = [v for k, v in stage2.items()]
        #
        # extracted_feature = self.multiStageMerger(stage2)[0]
        # Feature map, from original RGB Image, should be shaped as (batch, 256, H/4, W/4)
        extracted_feature  =  self._extract_feature(x)
        
        
        # Check if we are receiving expected size
        batch, c, h, w, device, = *extracted_feature.shape, extracted_feature.device
        assert gt_map.shape == (
            batch, 1, ho, wo), "Something wrong with data loading, ground truth map not in correct shape"
        
        # To bring gt map down to same size as feature map
        gt_map = F.interpolate(input=gt_map, size=extracted_feature.shape[2:], mode='bilinear')
        
        # cook up noise and add to ground truth map(following cosine noise schedule)
        eps = torch.randn_like(gt_map)
        t = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0], self.sample_range[1]).ceil()
        t_noise_control = t / self.timesteps_upper_limit  # size is [batch, ]
        while t_noise_control.ndim<gt_map.ndim:
            t_noise_control=t_noise_control.unsqueeze(-1)
        gt_map_corrupted = torch.sqrt(self.gamma(t_noise_control)) * gt_map + torch.sqrt(1 - self.gamma(t_noise_control)) * eps
        
        # "Fuse" together 1.extracted feature map(from RGB image) and 2.corrupted noise together.
        feat = torch.cat([extracted_feature, gt_map_corrupted], dim=1)  # (batch, 256+1, H/4, W/4)
        feat = self.beforeUnet(feat)  # (batch, 256, H/4, W/4)
        
        # Prepare time information
        # must be  from [0,0.999] range, this is to be fed to unet argument 't'
        timestep_for_unet=t / self.timesteps_upper_limit
        
        # get predicted x_0, we directly calculate loss on this, with ground truth density map.
        pred_map = self.unet(feat, timestep_for_unet)  # (batch, 1, H/4, W/4)
        
        # Get loss and get out
        # Yes gt_map has never entered latent space(beyond simple scaling)
        loss = self.loss_fn(pred_map, gt_map)
        return loss, pred_map, gt_map
    
    def _inference_step(self, x, gt_map=None, rescale=True):
        ho, wo = x.shape[-2], x.shape[-1]
        extracted_feature = self._extract_feature(x)
        batch, c, h, w, device, = *extracted_feature.shape, extracted_feature.device
        ## No ground truth needed during inferencing
        # if gt_map==None:
        #     gt_map=torch.randn_like(extracted_feature)
        # assert gt_map.shape == (
        #     batch, 1, ho, wo), "Something wrong with data loading, ground truth map not in correct shape"
        # # To bring gt map down to same size as feature map
        # gt_map = F.interpolate(input=gt_map, size=extracted_feature.shape[2:], mode='bilinear')
        out= self.sample_fn(extracted_feature)
        if rescale:
            out = F.interpolate(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                )
        return out
            # F.interpolate(input=gt_map, size=extracted_feature.shape[2:], mode='bilinear')
        '''
        

 _____   ____    ____   _______    _____      ________   ____    ____   ________   ____  _____   _________
|_   _| |_   \  /   _| |_   __ \  |_   _|    |_   __  | |_   \  /   _| |_   __  | |_   \|_   _| |  _   _  |
  | |     |   \/   |     | |__) |   | |        | |_ \_|   |   \/   |     | |_ \_|   |   \ | |   |_/ | | \_|
  | |     | |\  /| |     |  ___/    | |   _    |  _| _    | |\  /| |     |  _| _    | |\ \| |       | |
 _| |_   _| |_\/_| |_   _| |_      _| |__/ |  _| |__/ |  _| |_\/_| |_   _| |__/ |  _| |_\   |_     _| |_
|_____| |_____||_____| |_____|    |________| |________| |_____||_____| |________| |_____|\____|   |_____|
        

        '''
    
    def _extract_feature(self, x):
        stage1 = self.convnext(x)
        
        temp = OrderedDict()  # Because it takes OrderedDict
        for i, datum in enumerate(stage1):
            temp["{}".format(i)] = datum
        stage2 = self.fpn(temp)
        stage2 = [v for k, v in stage2.items()]
        
        extracted_feature = self.multiStageMerger(stage2)[0]
        return extracted_feature
    
    def gamma(self, t, ns=0.0002, ds=0.00025):
        '''
        : t within [0.0, 0.999] range, in guided_diffusion it takes sample from [0,999] and divide it by 1000, now we do this outside
        : return alphas_cumprod at (input_t * 1000) step
        '''
        return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2
    
    @torch.no_grad()
    def sample_fn(self,x):
        '''
        Take **extracted** feature map x(1/4 of original RGB image) as input, generate a random noise,  fuse the noise and input feature map, send into unet
        '''
        self.randsteps = 1
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)# a list of (2, batch) tensors
        # >>>>>This ?
        # x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        # depth_t = torch.randn((self.randsteps, 1, h, w), device=device)
        # >>>>> Or This ?
        x=x
        depth_t = torch.randn(size=(b,1,h,w),device=x.device)
        # randsteps是什么？
        for times_now, times_next in time_pairs:
            feat = torch.cat([x, depth_t], dim=1)
            feat = self.beforeUnet(feat)
            depth_pred = self.unet(feat, times_next)
            # t = self.time_mlp(times_now)
            # depth_pred = self._decode_head_forward_test([feat], t, img_metas=img_metas)
            #>>>>>This ?
            # depth_pred_ = ((depth_pred - self.min_depth) / (self.max_depth - self.min_depth))
            # depth_pred_ = ((depth_pred_ * 2) - 1) * self.bit_scale
            #>>>>>or?
            depth_pred_ =  depth_pred
            while times_now.ndim < feat.ndim:
                times_now = times_now.unsqueeze(-1)
            while times_next.ndim < feat.ndim:
                times_next = times_next.unsqueeze(-1)
            sample_func = self.ddim_step if self.ddim else self.ddpm_step
            depth_t = sample_func(depth_t, depth_pred_, times_now, times_next)
        # out = depth_pred.mean(dim=0, keepdim=True)
        out = depth_pred
        return out
        
    
    
    @torch.no_grad()
    def ddim_step(self, x_t, x_pred, t_now, t_next):
        '''
        Special Variant of DDIM, totally deterministic
        '''
        alpha_now = self.gamma(t=t_now)
        alpha_next = self.gamma(t=t_next)
        # x_pred = x_pred.clamp_(-self.bit_scale, self.bit_scale)
        x_pred = x_pred.clamp_(0,1)
        eps = (1 / (1 - alpha_now).sqrt()) * (x_t - alpha_now.sqrt() * x_pred)
        x_next = alpha_next.sqrt() * x_pred + (1 - alpha_next).sqrt() * eps
        return x_next
    
    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps_inference):
            t_now = 1 - step / self.timesteps_inference
            t_next = max(1 - (step + 1 + self.timestep_step_length) / self.timesteps_inference, 0)
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times
    
    
    
    '''
.__   __.   ______   .___________.    __  .__   __.    __    __       _______. _______
|  \ |  |  /  __  \  |           |   |  | |  \ |  |   |  |  |  |     /       ||   ____|
|   \|  | |  |  |  | `---|  |----`   |  | |   \|  |   |  |  |  |    |   (----`|  |__
|  . `  | |  |  |  |     |  |        |  | |  . `  |   |  |  |  |     \   \    |   __|
|  |\   | |  `--'  |     |  |        |  | |  |\   |   |  `--'  | .----)   |   |  |____
|__| \__|  \______/      |__|        |__| |__| \__|    \______/  |_______/    |_______|
    
    '''
    
    def forward(self, x, gt_map=None, train=True):
        print("You have entered forward function")
        if train:
            assert gt_map!=None, "If in training, ground truth map must be fed."
            return self.forward_train(x, gt_map)
        else:
            return self.forward_test(x)
        print("You are about to leave forward function")
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
        t = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0], self.sample_range[1])
        t = t / self.timesteps# size is [batch, ]
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
    
    