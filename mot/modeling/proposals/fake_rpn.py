import math
import json
import copy
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Instances, Boxes
from detectron2.modeling import detector_postprocess
from detectron2.utils.comm import get_world_size


@PROPOSAL_GENERATOR_REGISTRY.register()
class fake_rpn(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d()
        )


    def forward(self):
        return