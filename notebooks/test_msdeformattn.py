
import sys
sys.path.append("/app")
from modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
import torch
from detectron2.layers import ShapeSpec

import os

input_1 = torch.load("/data/dinov2-main/data/pixel_decoder_input_0.pth")
input_2 = torch.load("/data/dinov2-main/data/pixel_decoder_input_1.pth")
input_3 = torch.load("/data/dinov2-main/data/pixel_decoder_input_2.pth")
input_4 = torch.load("/data/dinov2-main/data/pixel_decoder_input_3.pth")

input_shape = {
    "res2":ShapeSpec(channels=1536,stride=4),
    "res3":ShapeSpec(channels=1536,stride=8),
    "res4":ShapeSpec(channels=1536,stride=16),
    "res5":ShapeSpec(channels=1536,stride=32)
}

model = MSDeformAttnPixelDecoder(
    input_shape=input_shape,
    transformer_dropout=0.0,
    transformer_nheads=32,
    transformer_dim_feedforward=1536,
    transformer_enc_layers=3,
    conv_dim=1536,
    mask_dim=1536,
    norm="GN",
    common_stride=4,
    transformer_in_features=["res3", "res4", "res5"]
)