import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.modules import MSDeformAttn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from .msdeformattn import MSDeformAttnTransformerEncoderOnly
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
import fvcore.nn.weight_init as weight_init
import numpy as np
@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, cfg, in_channels,
                 strides, feat_channels,out_channels,num_outs,

                 ):
        super().__init__()
        self.strides = strides
        self.num_input_leves = len(in_channels)
        self.num_encoder_levels = cfg['encoder']['transformerlayers']['attn_cfgs']['num_levels']
        assert self.num_encoder_levels >= 1, "num_levels in attn_cfgs must be at least one"
        input_conv_list = []
        for i in range(self.num_input_levels - 1, self.num_input_levels - self.num_encoder_levels - 1, -1):
            input_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], feat_channels,kernel_size=1),
                nn.GroupNorm(32, feat_channels)
            )
            input_conv_list.append(input_conv)
        self.input_convs = nn.ModuleList(input_conv_list)

        for layer in self.input_convs:
            nn.init.xavier_uniform_(layer[0].weight,gain=1)
            nn.init.constant_(layer[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=cfg['encoder']['d_model'],
            dropout=cfg['encoder']['dropout'],
            nhead=cfg['encoder']['n_heads'],
            dim_feedforward=cfg['encoder']['dim_feedforward'],
            num_encoder_layers=cfg['encoder']['encoder_layers'],
            num_feature_levels=cfg['encoder']['num_feature_levels']
        )
        N_steps = feat_channels // 2
        self.pe_layer = PositionEmbeddingSine(N_steps,normalize=True)

        self.mask_dim = cfg['mask_dim']
        self.mask_features = Conv2d(
            feat_channels,
            self.mask_dim,
            kernel_size=1,stride=1,padding=0
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3
        self.common_stride = cfg['common']['stride']
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = cfg['norm'] == ""
        idx = 0
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_norm = get_norm(cfg['norm'], feat_channels)
            output_norm = get_norm(cfg['norm'], feat_channels)

            lateral_conv = Conv2d(
                in_channels,feat_channels,kernel_size=1,bias=use_bias,norm=lateral_norm
            )
            output_conv = Conv2d(
                feat_channels,feat_channels,kernel_size=3,stride=1,
                padding=1,bias=use_bias,norm=output_norm,activation=F.relu
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)
            idx = idx + 1
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
