import torch
import torch.nn as nn
from torchvision.ops import roi_align

class RoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, proposals):
        rois = []
        for I, proposal in enumerate(proposals):
            proposal = proposal.to(feature_map.device)
            num_proposals = proposal.size(0)
            ids = torch.full((num_proposals, 1), I, dtype=torch.float32, device=feature_map.device)
            rois.append(torch.cat([ids, proposal], dim=1))

        rois = torch.cat(rois, dim=0)
        return roi_align(feature_map, rois, self.output_size, spatial_scale=self.spatial_scale)