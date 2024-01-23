import logging
from torch import Tensor
import torch
from typing import NamedTuple
from .base_dataset import _P3DFaces
from dataclasses import dataclass
from .base_dataset import ObjectDataset
from pytorch3d.io import load_obj, load_ply
import os
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ObjectData:
    object_segs: Tensor
    object_verts: Tensor
    object_faces: NamedTuple
    object_aux: NamedTuple = None
    sampled_verts: Tensor = None
    contacts: Tensor = None
    partitions: Tensor = None

    def to(self, device):
        self.object_segs = self.object_segs.to(device)
        self.object_verts = self.object_verts.to(device)

        # to(device) of NamedTuple
        object_faces = {
            field: getattr(self.object_faces, field).to(device)
            for field in self.object_faces._fields
        }
        self.object_faces = _P3DFaces(**object_faces)

        # to(device) of Dict
        if self.object_aux is not None:
            object_aux = {}
            for k, v in self.object_aux.items():
                if isinstance(v, torch.Tensor):
                    object_aux[k] = v.to(device)
                else:
                    object_aux[k] = v
                    logger.debug(
                        f"{k} hasn't been moved to device. Got {type(object_aux[k])}"
                    )
            self.object_aux = object_aux

        # to(device) of Optional
        if self.sampled_verts is not None:
            self.sampled_verts = self.sampled_verts.to(device)
        if self.contacts is not None:
            self.contacts = self.contacts.to(device)
        if self.partitions is not None:
            self.partitions = self.partitions.to(device)
        return self


class YCBDataset(ObjectDataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
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
            object_verts=object_verts,
            object_faces=object_faces,
            sampled_verts=sampled_verts,
            contacts=contacts,
            partitions=partitions,
        )

        # add only if not None
        if object_aux is not None:
            object_aux = {
                k: v for k, v in object_aux._asdict().items() if v is not None
            }
            return_dict["object_aux"] = object_aux

        return return_dict