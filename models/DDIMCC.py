import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# from minlora import (
#     LoRAParametrization,
#     add_lora,
#     apply_to_lora,
#     merge_lora,
# )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()
        
        if model_name == 'DDP':
            from .DDIMCC_Model import DDP as net
        
        
        self.CCN = net()
        self.loss_mse_fn = nn.MSELoss().to(device)
    
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        if density_map.shape[-2:-1] != gt_map.shape[-2:-1]:
            print("Warning, 8* downsample upsample size mismatch")
            return 'None'
        self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map
    
    # 举例在选用SHHB和Res50时 gt_map(BATCH,768,1024) img(BATCH,3,768,1024) density_map(BATCH,1,768,1024) self.loss_mse是一个独立的数值
    def build_loss(self, density_map, gt_data):
        # print(density_map.shape, gt_data.shape)
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse
    
    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map

