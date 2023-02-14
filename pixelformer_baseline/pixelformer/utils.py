import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms

import os, sys
import numpy as np
import math
import torch


class silog_loss(nn.Module):

    def __init__(self, variance_focus):
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
