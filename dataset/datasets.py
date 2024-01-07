import os
import pickle
import logging
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pytorch3d.io import load_obj, load_ply
from submodules.NIMBLE_model.utils import vertices2landmarks

logger = logging.getLogger(__name__)


@dataclass
class Data:
    hand_verts: Tensor
    hand_faces: NamedTuple
    object_verts: Tensor
    object_faces: NamedTuple
    hand_aux: NamedTuple = None
    object_aux: NamedTuple = None
    sampled_verts: Tensor = None
    contacts: Tensor = None
    partitions: Tensor = None

    def to(self, device):
        self.hand_verts = self.hand_verts.to(device)
        self.object_verts = self.object_verts.to(device)
        if self.sampled_verts is not None:
            self.sampled_verts = self.sampled_verts.to(device)
        if self.contacts is not None:
            self.contacts = self.contacts.to(device)
        if self.partitions is not None:
            self.partitions = self.partitions.to(device)
        return self


class BaseDataset(Dataset):
    def __init__(self) -> None:
        return

    def nimble_to_mano(self, verts):
        """
        verts: torch.tensor B x V x 3
        """
        skin_v = verts
        skin_f = self.skin_f
        skin_f = skin_f.to(verts.device)
        self.nimble_mano_vreg_fidx = self.nimble_mano_vreg_fidx.to(verts.device)
        self.nimble_mano_vreg_bc = self.nimble_mano_vreg_bc.to(verts.device)
        nimble_mano = torch.cat(
            [
                vertices2landmarks(
                    skin_v,
                    skin_f.squeeze(),
                    self.nimble_mano_vreg_fidx[i],
                    self.nimble_mano_vreg_bc[i],
                ).unsqueeze(0)
                for i in range(20)
            ]
        )
        skin_f = skin_f.cpu()
        self.nimble_mano_vreg_fidx = self.nimble_mano_vreg_fidx.cpu()
        self.nimble_mano_vreg_bc = self.nimble_mano_vreg_bc.cpu()
        nimble_mano_v = nimble_mano.mean(0)

        nimble_mano_f = self.mano_f
        return nimble_mano_v, nimble_mano_f


class ManualDataset(BaseDataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.hand = opt.hand
        self.object = opt.object
        self.nimble = opt.hand.nimble

        self.mano_f = torch.from_numpy(
            pickle.load(open(self.hand.mano_right, "rb"), encoding="latin1")[
                "f"
            ].astype(np.int64),
        )

        # nimble model
        nimble_pm_dict = np.load(opt.hand.nimble_pm_dict, allow_pickle=True)
        nimble_mano_vreg = np.load(opt.hand.nimble_mano_vreg, allow_pickle=True)
        self.skin_f = nimble_pm_dict["skin_f"]
        self.nimble_mano_vreg_fidx = nimble_mano_vreg["lmk_faces_idx"]
        self.nimble_mano_vreg_bc = nimble_mano_vreg["lmk_bary_coords"]

        return

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # hand
        hand_ext = os.path.splitext(self.hand.path)[1]
        if hand_ext == ".obj":
            load_textures = True if self.nimble else False
            hand_verts, hand_faces, hand_aux = load_obj(
                self.hand.path, load_textures=load_textures
            )
        elif hand_ext == ".ply":
            if self.nimble:
                logger.error("We only support .obj hand when nimble=True")
            hand_verts, hand_faces = load_ply(self.hand.path)
            hand_aux = None
        else:
            raise ValueError(f"hand file extension {hand_ext} not supported")

        # object
        object_ext = os.path.splitext(self.object.path)[1]
        if object_ext == ".obj":
            object_verts, object_faces, object_aux = load_obj(
                self.object.path, load_textures=True
            )
        elif object_ext == ".ply":
            if self.nimble:
                logger.error("We only support .obj object when nimble=True")
            object_verts, object_faces = load_ply(self.object.path)
            object_aux = None
        else:
            raise ValueError(f"object file extension {object_ext} not supported")

        # ContactGen
        sampled_verts = torch.from_numpy(np.load(self.object.sampled_verts_path))
        contacts = torch.from_numpy(np.load(self.object.contacts_path))
        partitions = torch.from_numpy(np.load(self.object.partitions_path))

        return_dict = dict(
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            object_verts=object_verts,
            object_faces=object_faces,
            sampled_verts=sampled_verts,
            contacts=contacts,
            partitions=partitions,
        )

        # add aux only if not None
        if hand_aux is not None and object_aux is not None:
            hand_aux = {k: v for k, v in hand_aux._asdict().items() if v is not None}
            object_aux = {
                k: v for k, v in object_aux._asdict().items() if v is not None
            }
            return_dict["hand_aux"] = hand_aux
            return_dict["object_aux"] = object_aux

        return return_dict
