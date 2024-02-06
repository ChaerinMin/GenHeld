import logging

from .base_dataset import ObjectDataset
import re
import glob

logger = logging.getLogger(__name__)


class YCBDataset(ObjectDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.object = opt

        self.fidxs = []
        pattern = re.compile(self.object.path)
        for p in glob.glob(self.object.path):
            match = pattern.match(p)
            self.fidxs.append(match.group(0))
