import glob
import json
import logging
import os

from .base_dataset import HandDataset
from typing import Dict
import re 

logger = logging.getLogger(__name__)


class FreiHANDDataset(HandDataset):
    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)

        # fidxs
        fidxs = []
        pattern_whole = re.compile(opt.hand.path.replace("%07d", "[0-9]{7}"))
        pattern_part = re.compile("[0-9]{7}")
        for p in glob.glob(opt.hand.path.replace("%07d", "*")):
            if pattern_whole.match(p) is not None:
                match = pattern_part.search(p)
                fidxs.append(int(match.group(0)))
            else:
                logger.error(f"Invalid path: {p}")
        if len(fidxs) == 0:
            logger.error("No hand file found")
        fidxs = sorted(fidxs)

        # resume fidxs
        resume_fidx = 0
        end_fidx = fidxs[-1]
        resume_fidx_path = os.path.join(self.cfg.output_dir, "resume_fidx.txt")
        if os.path.exists(resume_fidx_path):
            with open(resume_fidx_path, "r") as f:
                resume_fidx, end_fidx = map(int, f.read().split("\n"))
            logger.info(f"Resuming from {resume_fidx}")
        self.fidxs = fidxs[fidxs.index(resume_fidx):fidxs.index(end_fidx)+1]        
        self.end_fidx = end_fidx

        # True / False
        self.nimble = opt.hand.nimble
        self.arm = opt.hand.arm
