import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels=256, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        return logits, bbox_pred

def generate_anchors(feature_map_size, sizes, ratios):
    anchors = []
    for y in range(feature_map_size[0]):
        for x in range(feature_map_size[1]):
            for size in sizes:
                for ratio in ratios:
                    w = size * ratio[0]
                    h = size * ratio[1]
                    anchors.append([x, y, w, h])
    return torch.tensor(anchors, dtype=torch.float32)