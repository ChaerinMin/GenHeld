import glob
import json
import logging
import os

from .base_dataset import HandDataset
import re 

logger = logging.getLogger(__name__)


class FreiHANDDataset(HandDataset):
    def __init__(self, opt):
        super().__init__(opt)

        # fidxs
        self.fidxs = []
        pattern_whole = re.compile(opt.hand.path.replace("%07d", "[0-9]{7}"))
        pattern_part = re.compile("[0-9]{7}")
        for p in glob.glob(opt.hand.path.replace("%07d", "*")):
            if pattern_whole.match(p) is not None:
                match = pattern_part.search(p)
                self.fidxs.append(int(match.group(0)))
            else:
                logger.error(f"Invalid path: {p}")
        if len(self.fidxs) == 0:
            logger.error("No hand file found")
        self.fidxs = sorted(self.fidxs)

        # True / False
        self.nimble = opt.hand.nimble
        self.arm = opt.hand.arm
