# from convnextv2 import convnextv2_tiny
from collections import OrderedDict

import torch
import torchvision
import timm
from torch import  nn
from timm import create_model


from MultiStageMerge import MultiStageMerging
from Transformer import PatchEmbed, UViT


loss = nn.L1Loss()(torch.ones((2,1,12,12))*2, torch.zeros((2,1,12,12)))



convnext = create_model('convnext_tiny', pretrained=True,features_only=True)
input=torch.randn(size=(2,3,768,1024))
output=convnext(input)
'''
...torch.Size([2, 96, 192, 256])
...torch.Size([2, 192, 96, 128])
...torch.Size([2, 384, 48, 64])
...torch.Size([2, 768, 24, 32])
'''
for x in output:
    print(x.shape)
for name ,param in convnext.named_parameters():
    print(name)

fpn=torchvision.ops.FeaturePyramidNetwork(in_channels_list=[96, 192, 384, 768], out_channels=256)
x=OrderedDict()
for i, datum in enumerate(output):
    x["{}".format(i)]=datum
    
output=fpn(x)
print([(k, v.shape) for k, v in output.items()])
'''
[('0', torch.Size([2, 256, 192, 256])),
('1', torch.Size([2, 256, 96, 128])),
('2', torch.Size([2, 256, 48, 64])),
('3', torch.Size([2, 256, 24, 32]))]
'''

input=[v for k,v in output.items()]

merger=MultiStageMerging(in_channels=[256, 256, 256, 256],
            out_channels=256,
            kernel_size=1,)

merged=merger(input)[0]

print("Shape after merging:{}".format(merged.shape))
'''
...Shape after merging:torch.Size([2, 256, 192, 256])
'''



patch_embed=PatchEmbed(patch_size=4, in_chans=3, embed_dim=768)
input=torch.randn(size=(2,3,768,1024))
output=patch_embed(input)
print("output shape is after PatchEmbed: ", output.shape)
'''
...output shape is after PatchEmbed:  torch.Size([2, 49152, 768])
'''

from DDP import DDP
ddp=DDP().cuda()
input=torch.randn(size=(2,3,768,1024)).cuda()
pseudo_gt_map=torch.randn(size=(2,1,768,1024)).cuda()
way2=ddp._inference_step(input, pseudo_gt_map)
# way2=ddp._forward_step(input, pseudo_gt_map,)
print(way2.shape, "is way2 shape")

# from Transformer import timestep_embedding
# input=torch.arange(1,11)
# temb=timestep_embedding(input, dim=1024)# (10, 1024)
# print("timestep_embed size", temb.shape, temb)

# uvit=UViT(
#     img_size=(192, 256),
#     patch_size=4,
#     in_chans=256,
#     embed_dim=768,
#     depth=6,
#     num_heads=12,
#     mlp_ratio=4.,
#     qkv_bias=False,
#     qk_scale=None,
#     norm_layer=nn.LayerNorm,
#     mlp_time_embed=False,
#     num_classes=-1,
#     use_checkpoint=True,
#     conv=True,
#     skip=True
# )
# input=torch.randn(size=(2,256,192,256))
# timestep=...# (Batch, )
# timestep=torch.arange(0,2)
# uvit.cuda()
# input=input.cuda()
# timestep = timestep.cuda()
# print()
# output=uvit(input, timestep)
#
# from torchviz import  make_dot
# make_dot(output.mean(), params=dict(uvit.named_parameters()))
#
# print("output shape is after PatchEmbed: ", output.shape)
# '''
# ...output shape is after PatchEmbed:  torch.Size([2, 256, 192, 256])
# '''



