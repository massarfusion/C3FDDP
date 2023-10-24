import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiStageMerging(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 ):
        super(MultiStageMerging, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down=nn.Conv2d(in_channels=sum(in_channels), out_channels=out_channels, kernel_size=kernel_size,stride=1,padding=0)
    
    
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = list()
        size = inputs[0].shape[2:]
        for index, input in enumerate(inputs):
            # input = resize(input,
            #                size=size,
            #                mode='bilinear',
            #                align_corners=self.align_corners)
            temp=F.interpolate(input=input, size=size,mode='bilinear')
            outs.append(temp)
        out = torch.cat(outs, dim=1)
        out = self.down(out)
        return [out]