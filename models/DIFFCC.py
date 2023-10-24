import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name,cfg):
        super(CrowdCounter, self).__init__()        

        if model_name == 'Diffusion':
            from guided_diffusion import dist_util, logger
            from guided_diffusion.resample import create_named_schedule_sampler
            from guided_diffusion.script_util import (
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                args_to_dict,
                add_dict_to_argparser,
            )
            from .Diffusion_Model.Diffusion import Diffusion as diffuser

        
        
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(cfg, model_and_diffusion_defaults().keys())
        )
        schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion, maxt=cfg.diffusion_steps)
        
        
        self.CCN=diffuser(model,diffusion)
        
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.to(device)
        
        
        self.loss_mse_fn = nn.MSELoss().to(device)
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    # 举例在选用SHHB和Res50时 gt_map(BATCH,768,1024) img(BATCH,3,768,1024) density_map(BATCH,1,768,1024) self.loss_mse一个数值
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

