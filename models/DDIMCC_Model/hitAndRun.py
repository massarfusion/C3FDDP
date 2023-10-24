# from convnextv2 import convnextv2_tiny
from collections import OrderedDict

import torch
import torchvision
import timm
from timm import create_model


from MultiStageMerge import MultiStageMerging


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

from DDP import DDP
ddp=DDP()
input=torch.randn(size=(2,3,768,1024))
pseudo_gt_map=torch.randn(size=(2,1,768,1024))
way2=ddp.forward(input, pseudo_gt_map, train=True)
print(way2.shape, "is way2 shape")
