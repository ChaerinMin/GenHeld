import logging
import multiprocessing
import glob
import json
import os
import re
import random
import shutil
import copy 

import objaverse
from pytorch3d.io.experimental_gltf_io import load_meshes
from iopath.common.file_io import PathManager

from .base_dataset import ObjectDataset

logger = logging.getLogger(__name__)
n_processes = multiprocessing.cpu_count()

class YCBDataset(ObjectDataset):
    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)

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
                raise FileNotFoundError
        if len(self.fidxs) == 0:
            logger.error("No object file found")
            raise FileNotFoundError
        self.fidxs = sorted(self.fidxs)

        return

    def __len__(self):
        return len(self.fidxs)

    def get_idx(self, name, name_type=None, batch_idx=None):
        return self.fidxs.index(name)
    

class ObjaverseDataset(ObjectDataset):
    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        self.dir = os.path.dirname(self.object.path)
        objaverse.BASE_PATH = self.dir
        objaverse._VERSIONED_PATH = os.path.join(self.dir, "tmp")

        self.name2uid_path = os.path.join("assets","objaverse", "name_to_uid.json")
        if not os.path.exists(self.name2uid_path):
            with open(self.name2uid_path, "w") as f:
                f.write("{}")
                logger.info(f"Created a new {self.name2uid_path}")

        # same hand, same category
        self.prev_cate = None     
        # same hand, no duplicate object
        self.used = []
        return 
    
    def __len__(self):
        return 1  # always on the fly
    
    def get_idx(self, name, name_type, batch_idx):
        '''
        name_type: "trainset" (YCB) or "testset" (Objaverse)
        '''
        if name_type=="trainset":
            # remove unwanted prefix
            name = re.sub(r"^[0-9]+_", "", name)
            original_name = name
            if name.startswith("large_"):
                name = name[6:]
            name_words = name.split("_")[-2:]
            name = ' '.join(name_words)

            # find uids
            with open(self.name2uid_path, "r") as f:
                name2uid = json.load(f)
            if original_name in name2uid and not self.object.refresh_download:
                uids = name2uid[original_name]
            else:
                logger.debug("Loading Objaverse annotations")
                annotations = objaverse.load_annotations()
                lvis = objaverse.load_lvis_annotations()
                uids = []
                perfect_matches = []
                lvis_matches = []
                front_matches = []
                back_matches = []
                for uid, anno in annotations.items():
                    if anno['archives']['glb']['textureCount'] == 0:
                        continue
                    if name.lower() == anno['name'].lower():
                        perfect_matches.append(uid)
                    if name_words[0].lower() == anno['name'].lower():
                        front_matches.append(uid)
                    if len(name_words) == 2 and name_words[1].lower() == anno['name'].lower():
                        back_matches.append(uid)
                if len(name_words) == 2 and name_words[1].lower() in lvis.keys():
                    lvis_match = lvis[name_words[1].lower()]
                    for uid in lvis_match:
                        if name_words[1].lower() in annotations[uid]['name'].lower():
                            lvis_matches.append(uid)
                if len(perfect_matches) > 0:
                    uids = perfect_matches
                elif len(lvis_matches) > 0:
                    uids = lvis_matches
                elif len(front_matches) > 0:
                    uids = front_matches
                elif len(back_matches) > 0:
                    uids = back_matches
                else:
                    logger.error(f"No match found for {name}")
                    raise FileNotFoundError
                with open(self.name2uid_path, "w") as f:
                    name2uid[original_name] = uids
                    json.dump(name2uid, f)

            # download
            while True:
                uids_unused = copy.deepcopy(uids)
                if len(self.used) == batch_idx:
                    self.used.append(set())
                elif len(self.used) < batch_idx:
                    logger.error(f"Something wrong")
                    raise ValueError
                while True:
                    uid_choice = random.choice(uids_unused)
                    if uid_choice in self.used[batch_idx]:
                        uids_unused.remove(uid_choice)
                    else:
                        break
                self.used[batch_idx].add(uid_choice)
                fidx = os.path.join(original_name, uid_choice[:5])
                glb_path = os.path.splitext(self.object.path % fidx)[0] + ".glb"
                if not os.path.exists(self.object.path % fidx) or self.object.refresh:
                    name_dir = os.path.join(self.dir, original_name)
                    if not os.path.exists(name_dir):
                        os.makedirs(name_dir)
                    # download
                    tmp_path = objaverse.load_objects([uid_choice])[uid_choice]
                    # check broken
                    try: 
                        glb_mesh = load_meshes(tmp_path, PathManager())
                    except Exception as e:
                        logger.warning(f"Objaverse uid {uid_choice}: {e}")
                        uids.remove(uid_choice)
                        continue
                    # check texture
                    if glb_mesh[0][1].textures is None:
                        logger.warning(f"Objaverse uid {uid_choice} has no texture")
                        uids.remove(uid_choice)
                        continue
                    # success
                    try:
                        shutil.move(tmp_path, glb_path % fidx)
                        logger.debug(f"Downloaded {fidx}")
                    except Exception as e:
                        logger.error(e)
                        raise FileNotFoundError
                    with open(self.name2uid_path, "w") as f:
                        name2uid[original_name] = uids
                        json.dump(name2uid, f)
                        logger.info(f"Updated {self.name2uid_path}")
                break
            logger.info(f"Selected {fidx}")
        elif name_type=="testset":
            fidx = os.path.join(name[:-6], name[-5:])
        else:
            logger.error(f"Invalid name_type: {name_type}")
            raise ValueError

        self.fidxs = [fidx]
        return 0