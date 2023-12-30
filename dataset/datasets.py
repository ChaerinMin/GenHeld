from torch.utils.data import Dataset
from pytorch3d.io import load_obj, load_ply
import torch
import numpy as np
import os
from dataclasses import dataclass
from torch import Tensor


@dataclass
class Data:
    hand_verts: Tensor
    hand_faces: Tensor
    object_verts: Tensor
    object_faces: Tensor
    sampled_verts: Tensor = None
    contacts: Tensor = None
    partitions: Tensor = None

    def to(self, device):
        self.hand_verts = self.hand_verts.to(device)
        self.hand_faces = self.hand_faces.to(device)
        self.object_verts = self.object_verts.to(device)
        self.object_faces = self.object_faces.to(device)
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


class ManualDataset(BaseDataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.hand = opt.hand
        self.object = opt.object
        return

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # hand
        hand_ext = os.path.splitext(self.hand.path)[1]
        if hand_ext == ".obj":
            hand_verts, hand_faces, _ = load_obj(self.hand.path, load_textures=False)
            hand_faces = hand_faces.verts_idx
        elif hand_ext == ".ply":
            hand_verts, hand_faces = load_ply(self.hand.path)
        else:
            raise ValueError(f"hand file extension {hand_ext} not supported")

        # object
        object_ext = os.path.splitext(self.object.path)[1]
        if object_ext == ".obj":
            object_verts, object_faces, _ = load_obj(
                self.object.path, load_textures=False
            )
            object_faces = object_faces.verts_idx
        elif object_ext == ".ply":
            object_verts, object_faces = load_ply(self.object.path)
        else:
            raise ValueError(f"object file extension {object_ext} not supported")

        # ContactGen
        sampled_verts = torch.from_numpy(np.load(self.object.sampled_verts_path))
        contacts = torch.from_numpy(np.load(self.object.contacts_path))
        partitions = torch.from_numpy(np.load(self.object.partitions_path))

        return dict(
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            object_verts=object_verts,
            object_faces=object_faces,
            sampled_verts=sampled_verts,
            contacts=contacts,
            partitions=partitions,
        )
