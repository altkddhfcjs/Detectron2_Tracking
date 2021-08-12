# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import json
import math
import torch
from torch import nn
from torch.autograd.function import Function
from typing import Dict, List, Optional, Tuple, Union

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
#from .custom_fast_rcnn import CustomFastRCNNOutputLayers

#from .track_head.bounding_box import BoxList
##utils
from .track_head.EMM.bounding_box import cat_boxlist
from .track_head.track_head import build_track_head
from .track_head.track_utils import build_track_utils
from .track_head.track_solver import builder_tracker_solver
##tracking

@ROI_HEADS_REGISTRY.register()
class roi_tracking(StandardROIHeads):
    def __init__(
        self, 
        cfg, 
        input_shape: ShapeSpec,
        **kwargs
    ):
        super().__init__(cfg, input_shape, **kwargs)
        self.track_utils, self.track_pool = build_track_utils(cfg)
        self.track_head = build_track_head(cfg, self.track_utils, self.track_pool)
        self.track_solver = builder_tracker_solver(cfg, self.track_pool)
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        track_memory = None
    ):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        #del targets


        ## dataset need tracking ground-truth
        ## You'll coding tracking data processing code
        ## 
        print(targets)
        assert "test" 
        losses = {}

        if self.training:
            x, loss_detection = self._forward_box(features, proposals)
            losses.update(loss_detection)
        else:
            x, pred_instances = self._forward_box(features, proposals)

        
        if self.cfg.MODEL.TRACK_ON:
            y, tracks, loss_track = self.track_head(features, proposals, targets, track_memory)
            if self.training:
                losses.update(loss_track)
            # solver is only needed during inference
            else:
                if tracks is not None:
                    tracks = self._refine_tracks(features, tracks)
                    pred_instances = [cat_boxlist(pred_instances + tracks)]

                pred_instances = self.track_solver(pred_instances)

                # get the current state for tracking
                x = self.track.get_track_memory(features, pred_instances)

        return x, pred_instances, losses
    
    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        #del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            print(losses)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return box_features, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return box_features, pred_instances

