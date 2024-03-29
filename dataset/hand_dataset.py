import glob
import json
import logging
import os
import re

from .base_dataset import HandDataset

logger = logging.getLogger(__name__)


class FreiHANDDataset(HandDataset):
    def __init__(self, opt, cfg, device):
        super().__init__(opt, cfg, device)

        # fidxs
        fidxs = []
        pattern_whole = re.compile(opt.image.path.replace("%08d", "[0-9]{8}"))
        pattern_part = re.compile("[0-9]{8}")
        for p in glob.glob(opt.image.path.replace("%08d", "*")):
            if pattern_whole.match(p) is not None:
                match = pattern_part.search(p)
                fidxs.append(int(match.group(0)))
            else:
                logger.error(f"Invalid path: {p}")
                raise FileNotFoundError
        if len(fidxs) == 0:
            logger.error("No hand file found")
            raise FileNotFoundError
        fidxs = sorted(fidxs)

        # resume fidxs
        resume_fidx = 0
        end_fidx = fidxs[-1]
        resume_fidx_path = os.path.join(self.cfg.output_dir, "resume_fidx.txt")
        if os.path.exists(resume_fidx_path):
            with open(resume_fidx_path, "r") as f:
                resume_fidx, end_fidx = map(int, f.read().split("\n"))
            logger.info(f"If continue to train, will resume from {resume_fidx}")
        elif cfg.testtime_optimize.start_fidx is not None:
            resume_fidx = cfg.testtime_optimize.start_fidx
            assert cfg.testtime_optimize.end_fidx is not None
            end_fidx = cfg.testtime_optimize.end_fidx
            logger.info(f"Start from fidx {resume_fidx}")
        self.fidxs = fidxs[fidxs.index(resume_fidx):fidxs.index(end_fidx)+1]        
        self.end_fidx = end_fidx

        # True / False
        self.nimble = opt.hand.nimble
        self.arm = opt.hand.arm
