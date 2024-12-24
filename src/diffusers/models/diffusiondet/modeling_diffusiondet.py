import math
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss
from torchvision import ops
from transformers.utils.backbone_utils import load_backbone

from diffusers.models.diffusiondet.head import DiffusionDetHead
from diffusers.models.diffusiondet.loss import HungarianMatcherDynamicK, CriterionDynamicK

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def detector_postprocess(results_per_image, height, width):
    return


class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, config):
        super(DiffusionDet, self).__init__()

        self.device = torch.device('cuda')

        self.in_features = config.roi_head_in_features
        self.num_classes = 80
        self.num_proposals = config.num_proposals
        self.num_heads = config.num_heads

        self.preprocess_image = None
        self.backbone = None  # load_backbone(config)

        # build diffusion
        betas = cosine_beta_schedule(1000)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        sampling_timesteps = config.sample_step
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = 1.
        self.scale = config.snr_scale

        roi_input_shape = {
            'p2': {'stride': 4},
            'p3': {'stride': 8},
            'p4': {'stride': 16},
            'p5': {'stride': 32},
            'p6': {'stride': 64}
        }
        self.head = DiffusionDetHead(config, roi_input_shape=roi_input_shape, num_classes=self.num_classes)

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

        self.criterion = CriterionDynamicK(config, num_classes=self.num_classes, weight_dict=weight_dict)

    def model_predictions(self, backbone_feats, images_whwh, x, t):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = ops.box_convert(x_boxes, 'cxcywh', 'xyxy')
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = ops.box_convert(x_start, 'xyxy', 'cxcywh')
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images):
        bs = 1
        shape = (bs, self.num_proposals, 4)

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        outputs_class, outputs_coord = None, None
        for time, time_next in time_pairs:
            time_cond = torch.full((bs,), time, device=self.device, dtype=torch.long)

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
            threshold = 0.5
            score_per_image = torch.sigmoid(score_per_image)
            value, _ = torch.max(score_per_image, -1, keepdim=False)
            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            pred_noise = pred_noise[:, keep_idx, :]
            x_start = x_start[:, keep_idx, :]
            img = img[:, keep_idx, :]

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)

            if self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)

            if self.use_nms:
                # TODO: verify if the box_pred_per_image is in right format
                keep = ops.nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            # TODO: choose the right format to save results
            results = None
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
        """
        features = [torch.rand(1, 256, i, i) for i in [144, 72, 36, 18]]
        x_boxes = torch.rand(1, 300, 4)
        t = torch.rand(1)
        targets = None

        if not self.training:
            return self.ddim_sample()

        if self.training:
            outputs_class, outputs_coord = self.head(features, x_boxes, t)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    # TODO: verify if the box_pred_per_image is in right format
                    keep = ops.nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                # TODO: choose the right format to save results
                results = None
        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    # TODO: verify if the box_pred_per_image is in right format
                    keep = ops.nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                # TODO: choose the right format to save results
                results = None

        return results


