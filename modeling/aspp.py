"""
ASPP - Atrous Spatial Pyramid Pooling
https://developers.arcgis.com/python/guide/how-deeplabv3-works/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm) -> None:
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()
    
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __init__(self, output_stride, BatchNorm, global_avg_pool_bn=True) -> None:
        super().__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6,12,18]
        elif output_stride == 8:
            dilations = [1,12,24,36]
        else:
            raise NotImplementedError
    

        self.aspp1 = _ASPPModule(
            inplanes, 256,1,padding=0,dilation=dilations[0], BatchNorm=BatchNorm
        )

        self.aspp2 = _ASPPModule(
            inplanes,
            256,3,padding=dilations[1],dilation=dilations[1],BatchNorm=BatchNorm
        )
        self.aspp3 = _ASPPModule(
            inplanes, 256,3,padding=dilations[2], dilation=dilations[2],
            BatchNorm=BatchNorm
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            256,3,padding=dilations[3],dilation=dilations[3],
            BatchNorm=BatchNorm
        )

        ## for batch size 1???
        if global_avg_pool_bn:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(inplanes, 256,1,stride=1,bias=False),
                BatchNorm(256),
                nn.ReLU()
            )
        else:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(inplanes, 256,1,stride=1,bias=False),
                nn.ReLU()
            )
        
        self.conv1 = nn.Conv2d(1280, 256,1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self,x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.concat((x1,x2,x3,x4,x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

def build_aspp(output_stride, BatchNorm, global_avg_pool_bn=True):
    return ASPP(output_stride, BatchNorm, global_avg_pool_bn)


if __name__=="__main__":
    model = ASPP(8,nn.BatchNorm2d, global_avg_pool_bn=True)
    x = torch.rand((2,2048,256,256))
    y = model(x)


        
