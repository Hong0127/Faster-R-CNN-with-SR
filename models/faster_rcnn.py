import torch
import torch.nn as nn
from .backbone import Backbone
from .rpn import RPN
from .roi_align import RoIAlign
from .head import Head
from .super_resolution import SuperResolutionNet

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.backbone = Backbone()
        self.super_res = SuperResolutionNet()
        self.rpn = RPN(256)
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0)
        self.head = Head(256 * 7 * 7, num_classes)

    def forward(self, x, proposals):
        feature_map = self.backbone(x)
        feature_map = self.super_res(feature_map)
        rpn_logits, rpn_bbox_pred = self.rpn(feature_map)
        pooled_features = self.roi_align(feature_map, proposals)
        cls_score, bbox_pred = self.head(pooled_features)
        return cls_score, bbox_pred, rpn_bbox_pred