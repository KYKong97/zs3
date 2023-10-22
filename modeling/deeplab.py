import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.resnet import build_backbone

class DeepLab(nn.Module):

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init__(self, 
                 output_stride=16,
                 num_classes = 21,
                 sync_bn=False,
                 freeze_bn=False,
                 pretrained=True,
                 global_avg_pool_bn=True,
                 imagenet_pretrained_path="") -> None:
        super().__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(
            output_stride=output_stride,
            BatchNorm=BatchNorm,
            pretrained=pretrained,
            imagenet_pretrained_path=imagenet_pretrained_path
        )

        self.aspp = build_aspp(output_stride,BatchNorm, global_avg_pool_bn)
        self.decoder = build_decoder(num_classes, BatchNorm)

        if freeze_bn==True:
            self.freeze_bn()
    
    def _prediction(self,input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        
        return x, low_level_feat
    
    def forward(self, input):

        x, low_level_feat = self._prediction(input)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
        return x

    def forward_before_class_prediction(self, input):
        x, low_level_feat = self._prediction(input)
        x = self.decoder.forward_before_class_prediction(x,low_level_feat)
        return x

    def forward_class_prediction(self, x, input_size):
        x = self.decoder.forward_class_prediction(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x

    def forward_before_last_conv_finetune(self,input):
        x, low_level_feat = self._prediction(input)
        x = self.decoder.forward_before_last_conv_finetune(x, low_level_feat)
        return x
    
    def forward_class_last_conv_finetune(self,x):
        x = self.decoder.forward_class_last_conv_finetune(x)
        return x
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    


if __name__=="__main__":
    model = DeepLab(
        imagenet_pretrained_path=None, pretrained=False
    )
    x = torch.rand((2,3,312,312))
    y = model(x)

        


        