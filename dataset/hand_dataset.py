import glob
import json
import logging
import os

import torch

from .base_dataset import ROOT_JOINT_IDX, HandDataset

logger = logging.getLogger(__name__)


class FreiHANDDataset(HandDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.image = opt.image
        self.hand = opt.hand

        # paths
        self.fidxs = sorted(
            [int(os.path.basename(p).split(".")[0]) for p in glob.glob(opt.hand.path)]
        )

        # True / False
        self.nimble = opt.hand.nimble
        self.arm = opt.hand.arm

        # root xyz
        with open(opt.hand.xyz, "r") as f:
            img_fidx = int(opt.image.path.split("/")[-1].split(".")[0])
            xyz = json.load(f)[img_fidx]
        self.root_xyz = torch.tensor(xyz[ROOT_JOINT_IDX])
        self.xyz = xyz
