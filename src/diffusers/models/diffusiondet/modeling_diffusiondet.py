import torch
from torch import nn
from torch.nn.functional import l1_loss
from transformers.utils.backbone_utils import load_backbone

from diffusers.models.diffusiondet.head import DiffusionDetHead
from diffusers.models.diffusiondet.loss import HungarianMatcherDynamicK, CriterionDynamicK


class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, config):
        super(DiffusionDet, self).__init__()

        self.training = True

        self.preprocess_image = None
        self.backbone = None  # load_backbone(config)

        roi_input_shape = {
            'p2': {'stride': 4},
            'p3': {'stride': 8},
            'p4': {'stride': 16},
            'p5': {'stride': 32},
            'p6': {'stride': 64}
        }
        self.head = DiffusionDetHead(config, roi_input_shape=roi_input_shape, num_classes=80)

        self.deep_supervision = config.deep_supervision
        self.use_focal = config.use_focal
        self.use_fed_loss = config.use_fed_loss
        self.use_nms = config.use_nms

        weight_dict = {
            "loss_ce": config.class_weight, "loss_bbox": config.l1_weight, "loss_giou": config.giou_weight
        }
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = CriterionDynamicK(config, num_classes=80, weight_dict=weight_dict)

        self.in_features = 0

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
        """
        if self.training:
            features = [torch.rand(1, 256, i, i) for i in [144, 72, 36, 18]]
            x_boxes = torch.rand(1, 300, 4)
            t = torch.rand(1)
            targets = None

            outputs_class, outputs_coord = self.head(features, x_boxes, t)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)

            return loss_dict
