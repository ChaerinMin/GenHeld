import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple, Dict

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
    def device(self):
        return self._padded.device

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

    def clean(self, split, fill=0.0):
        batch_size = self._padded.shape[0]
        assert batch_size == len(split)
        for b in range(batch_size):
            self._padded[b, split[b] :] = fill

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
    hand_verts: Tensor
    hand_faces: NamedTuple
    mano_verts_r: Tensor
    mano_joints_r: Tensor
    xyz: Tensor = None
    inpainted_images: Tensor = None
    hand_aux: NamedTuple = None

    def to(self, device):
        self.hand_verts = self.hand_verts.to(device)
        self.mano_verts_r = self.mano_verts_r.to(device)
        self.mano_joints_r = self.mano_joints_r.to(device)
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
    object_verts_highres: PaddedTensor
    object_faces: NamedTuple
    object_faces_highres: NamedTuple
    object_aux: Dict = None
    object_aux_highres: Dict = None
    sampled_verts: PaddedTensor = None
    contacts: PaddedTensor = None
    partitions: PaddedTensor = None

    def to(self, device):
        self.object_verts = self.object_verts.to(device)
        self.object_verts_highres = self.object_verts_highres.to(device)

        # to(device) of NamedTuple
        object_faces = {}
        for field in self.object_faces._fields:
            object_faces[field] = getattr(self.object_faces, field).to(device)
        self.object_faces = _P3DFaces(**object_faces)
        object_faces_highres = {}
        for field in self.object_faces_highres._fields:
            object_faces_highres[field] = getattr(self.object_faces_highres, field).to(
                device
            )
        self.object_faces_highres = _P3DFaces(**object_faces_highres)

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
            object_aux_highres = {}
            for k, v in self.object_aux_highres.items():
                if isinstance(v, (torch.Tensor, PaddedTensor)):
                    object_aux_highres[k] = v.to(device)
                else:
                    object_aux_highres[k] = v
                    logger.debug(
                        f"{k} hasn't been moved to device. Got {type(object_aux_highres[k])}"
                    )
            self.object_aux_highres = object_aux_highres

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
        object_verts_highres = []
        object_faces = []
        object_faces_highres = []
        object_aux = []
        object_aux_highres = []
        sampled_verts = []
        contacts = []
        partitions = []

        for d in data:
            fidx.append(d["fidx"])
            object_verts.append(d["object_verts"])
            object_verts_highres.append(d["object_verts_highres"])
            object_faces.append(d["object_faces"])
            object_faces_highres.append(d["object_faces_highres"])

            # optional data
            if "object_aux" in keys:
                object_aux.append(d["object_aux"])
            if "object_aux_highres" in keys:
                object_aux_highres.append(d["object_aux_highres"])
            if "sampled_verts" in keys:
                sampled_verts.append(d["sampled_verts"])
            if "contacts" in keys:
                contacts.append(d["contacts"])
            if "partitions" in keys:
                partitions.append(d["partitions"])

        collated["fidx"] = fidx
        collated["object_verts"] = PaddedTensor(object_verts)
        collated["object_verts_highres"] = PaddedTensor(object_verts_highres)
        p3dfaces = {}
        p3dfaces_highres = {}
        for field in _P3DFaces._fields:
            faces = [getattr(face, field) for face in object_faces]
            faces_highres = [getattr(face, field) for face in object_faces_highres]
            if faces[0] is None:
                faces = None
            elif isinstance(faces[0], Tensor):
                faces = PaddedTensor(faces, padding_value=-1)
            else:
                faces = torch.tensor(faces)
            if faces_highres[0] is None:
                faces_highres = None
            elif isinstance(faces_highres[0], Tensor):
                faces_highres = PaddedTensor(faces_highres, padding_value=-1)
            else:
                faces_highres = torch.tensor(faces_highres)
            p3dfaces[field] = faces
            p3dfaces_highres[field] = faces_highres
        collated["object_faces"] = _P3DFaces(**p3dfaces)
        collated["object_faces_highres"] = _P3DFaces(**p3dfaces_highres)

        # optional data
        if "object_aux" in keys:
            p3daux = {}
            for k in object_aux[0].keys():
                if k == "material_colors" or k == "normals":
                    continue
                aux = [a[k] for a in object_aux]
                if isinstance(aux[0], Tensor):
                    aux = PaddedTensor(aux)
                elif k == "texture_images":
                    coll_texture = []
                    for i in range(len(aux)):
                        assert len(aux[i]) == 1, "Only one material is supported"
                        coll_texture.append(next(iter(aux[i].values())))
                    aux = {'material_0': torch.stack(coll_texture)}
                else:
                    aux = torch.tensor(aux)
                p3daux[k] = aux
            collated["object_aux"] = p3daux
        if "object_aux_highres" in keys:
            p3daux_highres = {}
            for k in object_aux_highres[0].keys():
                if k == "material_colors" or k == "normals":
                    continue
                aux = [a[k] for a in object_aux_highres]
                if isinstance(aux[0], Tensor):
                    aux = PaddedTensor(aux)
                elif k == "texture_images":
                    coll_texture = []
                    for i in range(len(aux)):
                        assert len(aux[i]) == 1, "Only one material is supported"
                        coll_texture.append(next(iter(aux[i].values())))
                    aux = {'material_0': torch.stack(coll_texture)}
                else:
                    aux = torch.tensor(aux)
                p3daux_highres[k] = aux
            collated["object_aux_highres"] = p3daux_highres
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
    hand_verts_r: Pointclouds
    hand_joints_r: Tensor
    hand_contacts_r: Tensor
    class_vecs: Tensor
    shape_codes: Tensor
    object_pcs_r: Pointclouds

    def to(self, device):
        self.hand_verts_r = self.hand_verts_r.to(device)
        self.hand_joints_r = self.hand_joints_r.to(device)
        self.hand_contacts_r = self.hand_contacts_r.to(device)
        self.class_vecs = self.class_vecs.to(device)
        self.shape_codes = self.shape_codes.to(device)
        self.object_pcs_r = self.object_pcs_r.to(device)
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
            elif isinstance(data[0][k], str):
                collated[k] = [d[k] for d in data]
        return collated

@dataclass
class SelectorTestData:
    @staticmethod
    def collate_fn(data):
        keys = list(data[0].keys())
        collated = {}
        for k in keys:
            if isinstance(data[0][k], torch.Tensor):
                collated[k] = torch.stack([d[k] for d in data], dim=0)
            elif isinstance(data[0][k], Pointclouds):
                collated[k] = join_pointclouds_as_batch([d[k] for d in data])
            elif isinstance(data[0][k], str):
                collated[k] = [d[k] for d in data]
        return collated