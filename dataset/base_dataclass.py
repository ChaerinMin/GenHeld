import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from pytorch3d.structures import Pointclouds, join_pointclouds_as_batch


_P3DFaces = namedtuple(
    "_P3DFaces",
    ["verts_idx", "normals_idx", "textures_idx", "materials_idx"],
    defaults=(None,) * 4,
)  # Python 3.7+

logger = logging.getLogger(__name__)


class PaddedTensor:
    def __init__(self, tensors=None, padding_value=0):
        if tensors is not None:
            self._split_sizes = torch.tensor([t.shape[0] for t in tensors])
            self._padded = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=padding_value
            )
        else:
            logger.debug("PaddedTensor has not automatically computed. Please set manually.")

    def to(self, device):
        self._padded = self._padded.to(device)
        self._split_sizes = self._split_sizes.to(device)
        return self

    @property
    def padded(self):
        return self._padded

    @property
    def split_sizes(self):
        return self._split_sizes
    
    @padded.setter
    def padded(self, value):
        self._padded = value
    
    @split_sizes.setter
    def split_sizes(self, value):
        self._split_sizes = value

    @staticmethod
    def from_padded(padded, split_sizes):
        pt = PaddedTensor()
        pt.padded = padded
        pt.split_sizes = split_sizes
        return pt


@dataclass
class HandData:
    fidxs: int
    images: Tensor
    # handarm_segs: Tensor
    # object_segs: Tensor
    hand_theta: Tensor
    hand_verts: Tensor
    hand_faces: NamedTuple
    xyz: Tensor = None
    inpainted_images: Tensor = None
    hand_aux: NamedTuple = None

    def to(self, device):
        # self.handarm_segs = self.handarm_segs.to(device)
        # self.object_segs = self.object_segs.to(device)
        self.hand_tehta = self.hand_theta.to(device)
        self.hand_verts = self.hand_verts.to(device)
        self.xyz = self.xyz.to(device)

        # to(device) of NamedTuple
        hand_faces = {
            field: getattr(self.hand_faces, field).to(device)
            for field in self.hand_faces._fields
        }
        self.hand_faces = _P3DFaces(**hand_faces)

        # to(device) of Dict
        if self.hand_aux is not None:
            hand_aux = {}
            for k, v in self.hand_aux.items():
                if isinstance(v, torch.Tensor):
                    hand_aux[k] = v.to(device)
                else:
                    hand_aux[k] = v
                    logger.debug(
                        f"{k} hasn't been moved to device. Got {type(hand_aux[k])}"
                    )
            self.hand_aux = hand_aux


@dataclass
class ObjectData:
    fidx: str
    object_verts: PaddedTensor
    object_faces: NamedTuple
    object_aux: NamedTuple = None
    sampled_verts: PaddedTensor = None
    contacts: PaddedTensor = None
    partitions: PaddedTensor = None

    def to(self, device):
        self.object_verts = self.object_verts.to(device)

        # to(device) of NamedTuple
        object_faces = {}
        for field in self.object_faces._fields:
            object_faces[field] = getattr(self.object_faces, field).to(device)
        self.object_faces = _P3DFaces(**object_faces)

        # to(device) of Dict
        if self.object_aux is not None:
            object_aux = {}
            for k, v in self.object_aux.items():
                if isinstance(v, (torch.Tensor, PaddedTensor)):
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

    @staticmethod
    def collate_fn(data):
        collated = {}
        keys = list(data[0].keys())

        fidx = []
        object_verts = []
        object_faces = []
        object_aux = []
        sampled_verts = []
        contacts = []
        partitions = []

        for d in data:
            fidx.append(d["fidx"])
            object_verts.append(d["object_verts"])
            object_faces.append(d["object_faces"])

            # optional data
            if "object_aux" in keys:
                object_aux.append(d["object_aux"])
            if "sampled_verts" in keys:
                sampled_verts.append(d["sampled_verts"])
            if "contacts" in keys:
                contacts.append(d["contacts"])
            if "partitions" in keys:
                partitions.append(d["partitions"])

        collated["fidx"] = fidx
        collated["object_verts"] = PaddedTensor(object_verts)
        p3dfaces = {}
        for field in _P3DFaces._fields:
            faces = [getattr(face, field) for face in object_faces]
            if faces[0] is None:
                faces = None
            elif isinstance(faces[0], Tensor):
                faces = PaddedTensor(faces, padding_value=-1)
            else:
                faces = torch.tensor(faces)
            p3dfaces[field] = faces
        collated["object_faces"] = _P3DFaces(**p3dfaces)

        # optional data
        if "object_aux" in keys:
            p3daux = {}
            for k in object_aux[0].keys():
                aux = [a[k] for a in object_aux]
                if isinstance(aux[0], Tensor):
                    aux = PaddedTensor(aux)
                elif k == "material_colors":
                    coll_material_colors = {}
                    for material in aux[0].keys():
                        coll_material = {}
                        for property in aux[0][material].keys():
                            coll_material[property] = torch.cat(
                                [aux[i][material][property] for i in range(len(aux))],
                                dim=0,
                            )
                        coll_material_colors[material] = coll_material
                    aux = coll_material_colors
                elif k == "texture_images":
                    coll_texture = {}
                    for material in aux[0].keys():
                        coll_texture[material] = torch.stack(
                            [aux[i][material] for i in range(len(aux))]
                        )
                    aux = coll_texture
                else:
                    aux = torch.tensor(aux)
                p3daux[k] = aux
            collated["object_aux"] = p3daux
        if "sampled_verts" in keys:
            collated["sampled_verts"] = PaddedTensor(sampled_verts)
        if "contacts" in keys:
            collated["contacts"] = PaddedTensor(contacts)
        if "partitions" in keys:
            collated["partitions"] = PaddedTensor(partitions, padding_value=-1)

        return collated


@dataclass
class SelectorData:
    fidxs: str
    hand_theta: Tensor
    hand_verts_n: Tensor
    hand_contacts_n: Tensor
    class_vecs: Tensor
    object_pcs_n: Pointclouds

    def to(self, device):
        self.hand_theta = self.hand_theta.to(device)
        self.hand_verts_n = self.hand_verts_n.to(device)
        self.hand_contacts_n = self.hand_contacts_n.to(device)
        self.class_vecs = self.class_vecs.to(device)
        self.object_pcs_n = self.object_pcs_n.to(device)
        return self
    
    @staticmethod
    def collate_fn(data):
        keys = list(data[0].keys())
        collated = {}
        for k in keys:
            if isinstance(data[0][k], torch.Tensor):
                collated[k] = torch.stack([d[k] for d in data], dim=0)
            elif isinstance(data[0][k], Pointclouds):
                collated[k] = join_pointclouds_as_batch([d[k] for d in data])
        return collated