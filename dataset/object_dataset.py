import logging

from .base_dataset import ObjectDataset
import re
import glob

logger = logging.getLogger(__name__)


class YCBDataset(ObjectDataset):
    def __init__(self, opt):
        super().__init__(opt)

        # fidxs
        self.fidxs = []
        pattern_whole = re.compile(self.object.path.replace("%s", "[0-9]{3}[a-z_\-]+"))
        pattern_part = re.compile("[0-9]{3}[a-z_\-]+\/")  # detect /
        for p in glob.glob(self.object.path.replace("%s", "*")):
            if pattern_whole.match(p) is not None:
                match = pattern_part.search(p)
                self.fidxs.append(match.group(0)[:-1])
            else:
                logger.error(f"Invalid path: {p}")
        if len(self.fidxs) == 0:
            logger.error("No object file found")
        self.fidxs = sorted(self.fidxs)

        return