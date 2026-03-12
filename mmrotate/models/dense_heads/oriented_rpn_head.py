# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.core import anchor_inside_flags, unmap

from mmrotate.core import obb2xyxy
from ..builder import ROTATED_HEADS
from .rotated_rpn_head import RotatedRPNHead

import os
import numpy as np
import torch.nn.functional as F

class PDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(PDC, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
    
class LieGroupRotationPDC(nn.Module):
    """S-PDC: Symmetry-aware Pixel Difference Convolution (kernel_size=3).

    Modulates PDC weights with a fixed PHT (Polar Harmonic Transform) symmetry
    kernel H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i), scaled by a trainable
    coefficient alpha, then averages responses over 8 SO(2) rotations.

    Paper eq.:
        y = alpha * sum_{i != c} w_i * H_i^(n,l) * (x_i - q)  (q absorbed into PDC diff)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(LieGroupRotationPDC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        # Trainable harmonic order coefficient alpha_{n,l} (initialised to 1.0)
        self.alpha = nn.Parameter(torch.ones(1))
        # Pre-calibrated per-position spectral weights sw_i = target_i / H_i^{(n,l)}
        # such that H_i * sw_i recovers the centre-surround symmetry kernel exactly
        self.register_buffer('_pht_spectral_weights', self._precompute_spectral_weights())

    def _precompute_spectral_weights(self):
        """Offline: solve sw_i = target_i / H_i^{(n=2,l=4)} on the 3x3 grid (k=1).

        H = cos(2*pi*2*r^2 + 4*theta):
          centre -> cos(0)=1,  edges -> cos(4*pi+4*theta_e)=1,  corners -> cos(8*pi+4*theta_c)=-1
        sw = target / H:
          centre->-4, edges->1, corners->0
        => H * sw = target  (exact recovery of the symmetry kernel)
        """
        N = self.kernel_size
        k = N // 2
        eps = 1e-6
        coords = torch.arange(N, dtype=torch.float32) - k
        grid_v, grid_u = torch.meshgrid(coords, coords)
        r2 = (grid_u ** 2 + grid_v ** 2) / (k ** 2 + eps)
        theta = torch.atan2(grid_v, grid_u + eps)
        H = torch.cos(2.0 * math.pi * 2 * r2 + 4 * theta)   # PHT order (n=2, l=4)
        target = torch.zeros(N, N)
        target[k, k]     = -4.0
        target[k - 1, k] =  1.0
        target[k + 1, k] =  1.0
        target[k, k - 1] =  1.0
        target[k, k + 1] =  1.0
        sw = torch.where(target != 0, target / H, torch.zeros_like(target))
        return sw  # [N, N]

    def forward(self, x):
        weights = self.conv.weight
        device = x.device
        weights = weights.to(device)

        # PHT harmonic kernel H_i^{(n,l)}, scaled by trainable alpha_{n,l}
        pht_kernel = self.create_pht_kernel(x.device)
        out = self.apply_lie_group_rotation(x, self.alpha * pht_kernel * weights, device, 1)

        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = self.apply_lie_group_rotation(x, kernel_diff, device, 0)

        output = out - self.theta * out_diff

        return output

    def create_pht_kernel(self, device):
        """Build the 3x3 PHT harmonic symmetry kernel H_i^{(n,l)}.

        Values are pre-evaluated on the discrete 3x3 grid using the polar
        harmonic formula:
            H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i)
        A small epsilon is added for numerical stability, matching
        Eq. (3-4) in the paper.
        """
        N = self.kernel_size
        k = N // 2
        eps = 1e-6
        coords = torch.arange(N, dtype=torch.float32, device=device) - k
        grid_v, grid_u = torch.meshgrid(coords, coords)
        r2 = (grid_u ** 2 + grid_v ** 2) / (k ** 2 + eps)
        theta = torch.atan2(grid_v, grid_u + eps)
        # H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i),  order (n=2, l=4)
        H = torch.cos(2.0 * math.pi * 2 * r2 + 4 * theta)
        # Combine with pre-calibrated spectral weights -> target symmetry kernel
        pht_kernel = H * self._pht_spectral_weights.to(device)
        pht_kernel = pht_kernel + eps
        pht_kernel = pht_kernel.unsqueeze(0).unsqueeze(0)
        pht_kernel = pht_kernel.repeat(self.in_channels, self.in_channels, 1, 1)
        return pht_kernel

    def apply_lie_group_rotation(self, x, weights, device, pad):
        rotation_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        rotated_weights = []
        output = []

        for angle in rotation_angles:
            rotation_matrix = self.rotation_matrix_2d(angle).to(device)
            rotated_weight = self.rotate_kernel(weights, rotation_matrix, device)
            rotated_weights.append(rotated_weight)

            output.append(F.conv2d(x, rotated_weight, stride=self.stride, padding=pad, dilation=self.dilation, groups=self.groups))

        return torch.stack(output).mean(0)

    def rotation_matrix_2d(self, angle):
        return torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]
        ])

    def rotate_kernel(self, weights, rotation_matrix, device):
        _, _, kernel_height, kernel_width = weights.shape
        rotated_kernel = torch.zeros_like(weights)

        for h in range(kernel_height):
            for w in range(kernel_width):
                rotated_h = int(h * rotation_matrix[0, 0] + w * rotation_matrix[0, 1])
                rotated_w = int(h * rotation_matrix[1, 0] + w * rotation_matrix[1, 1])

                if 0 <= rotated_h < kernel_height and 0 <= rotated_w < kernel_width:
                    rotated_kernel[:, :, rotated_h, rotated_w] = weights[:, :, h, w]

        return rotated_kernel.to(device)
    
class LieGroupRotationConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(LieGroupRotationConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        #self.theta = 0.7

    def forward(self, x):
        weights = self.conv.weight
        device = x.device
        weights = weights.to(device)
        
        output = self.apply_lie_group_rotation(x, weights, device, 2) #1 #9
        
        return output

    def create_laplacian_kernel(self, device):
        laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)
        laplacian_kernel = laplacian_kernel.repeat(self.in_channels, self.in_channels, 1, 1)
        return laplacian_kernel

    def apply_lie_group_rotation(self, x, weights, device, pad):
        rotation_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        rotated_weights = []
        output = []
        
        for angle in rotation_angles:
            rotation_matrix = self.rotation_matrix_2d(angle).to(device)
            rotated_weight = self.rotate_kernel(weights, rotation_matrix, device)
            rotated_weights.append(rotated_weight)
            
            output.append(F.conv2d(x, rotated_weight, stride=self.stride, padding=pad, dilation=self.dilation, groups=self.groups))

        return torch.stack(output).mean(0)

    def rotation_matrix_2d(self, angle):
        return torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]
        ])

    def rotate_kernel(self, weights, rotation_matrix, device):
        _, _, kernel_height, kernel_width = weights.shape
        rotated_kernel = torch.zeros_like(weights)
        
        for h in range(kernel_height):
            for w in range(kernel_width):
                rotated_h = int(h * rotation_matrix[0, 0] + w * rotation_matrix[0, 1])
                rotated_w = int(h * rotation_matrix[1, 0] + w * rotation_matrix[1, 1])
                
                if 0 <= rotated_h < kernel_height and 0 <= rotated_w < kernel_width:
                    rotated_kernel[:, :, rotated_h, rotated_w] = weights[:, :, h, w]
        
        return rotated_kernel.to(device)

@ROTATED_HEADS.register_module()
class OrientedRPNHead(RotatedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = LieGroupRotationPDC(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors ,4)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        gt_hbboxes = obb2xyxy(gt_bboxes, self.version)

        assign_result = self.assigner.assign(
            anchors, gt_hbboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_hbboxes)

        if gt_bboxes.numel() == 0:
            sampling_result.pos_gt_bboxes = gt_bboxes.new(
                (0, gt_bboxes.size(-1))).zero_()
        else:
            sampling_result.pos_gt_bboxes = \
                gt_bboxes[sampling_result.pos_assigned_gt_inds, :]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((anchors.size(0), 6))
        bbox_weights = anchors.new_zeros((anchors.size(0), 6))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 4).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 6)
        bbox_weights = bbox_weights.reshape(-1, 6)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 6)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

          Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx, _ in enumerate(cls_scores):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 6)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2]
            h = proposals[:, 3]
            valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if proposals.numel() > 0:
            hproposals = obb2xyxy(proposals, self.version)
            _, keep = batched_nms(hproposals, scores, ids, cfg.nms)
            dets = torch.cat([proposals, scores[:, None]], dim=1)
            dets = dets[keep]
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]
