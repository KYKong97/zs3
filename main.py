from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import torch
from open_vocab_seg.modelling.backbone.vit_adapter import ViTAdapter
from detectron2.projects.deeplab import add_deeplab_config
from open_vocab_seg.config import add_ovseg_config
from open_vocab_seg.data import MaskFormerSemanticDatasetMapper
from open_vocab_seg.data import build_detection_test_loader, build_detection_train_loader


args = default_argument_parser().parse_args()
print(args.opts)
cfg = get_cfg()

add_deeplab_config(cfg=cfg)
add_ovseg_config(cfg)

cfg.merge_from_file("configs/backbone_only.yaml")
cfg.merge_from_list(args.opts)
mapper = MaskFormerSemanticDatasetMapper(cfg, True)
loader = build_detection_train_loader(cfg, mapper=mapper, dataset=None)
model = DefaultTrainer.build_model(cfg)
print(model)

random_tensor = torch.randn(size=(1,3,224,224))
result = model(random_tensor)