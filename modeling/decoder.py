import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def __init__(self, num_classes, BatchNorm) -> None:
        super().__init__()
        low_level_inplanes = 256
        self.conv1 = nn.Conv2d(low_level_inplanes, 48,1,bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256,kernel_size=3, stride=1,padding=1,bias=False),
            BatchNorm(256),
            nn.Dropout(0.5),
            nn.Conv2d(256,256, kernel_size=3,stride=1, padding=1,bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.pred_conv = nn.Conv2d(256, num_classes, kernel_size=1,stride=1)

    def _prediction(self,x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(
            x, size=low_level_feat.size()[2:],mode="bilinear",
            align_corners=True
        )
        x = torch.concat((x,low_level_feat), dim=1)

        return x


    def forward(self, x, low_level_feat):
        
        x = self._prediction(x,low_level_feat)

        
        x = self.last_conv(x)
        x = self.pred_conv(x)
        return x
    
    
    def forward_before_class_prediction(self, x, low_level_feat):
        x = self._prediction(x, low_level_feat)
        x = self.last_conv(x)
        return x
    
    def forward_before_last_conv_finetune(self, x, low_level_feat):
        x = self._prediction(x, low_level_feat=low_level_feat)
        x = self.last_conv[:4](x)
        return x
    
    def forward_class_prediction(self,x):
        x = self.pred_conv(x)
        return x
    
    def forward_class_last_conv_finetune(self,x):
        x = self.last_conv[4:](x)
        return x

def build_decoder(num_classes,BatchNorm):
    return Decoder(num_classes,BatchNorm=BatchNorm)

if __name__=="__main__":
    model = Decoder(1000,nn.BatchNorm2d)
    x = torch.rand((2,3,224,224))
    low_level_feat = torch.rand((2,512,224,224))
    y = model(x,low_level_feat)

    