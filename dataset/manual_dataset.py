import os
import pickle
from typing import Dict
import logging
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import cv2
from PIL import Image
import torch
from torch import Tensor
from pytorch3d.io import load_obj, load_ply
from .base_dataset import BaseDataset, _P3DFaces, NIMBLE_N_VERTS, ROOT_JOINT_IDX
import json


logger = logging.getLogger(__name__)


@dataclass
class ManualData:
    images: Tensor
    intrinsics: Tensor
    light: Dict[str, Tensor]
    handarm_segs: Tensor
    object_segs: Tensor
    hand_verts: Tensor
    hand_faces: NamedTuple
    object_verts: Tensor
    object_faces: NamedTuple
    mano_pose: Tensor = None
    inpainted_images: Tensor = None
    hand_aux: NamedTuple = None
    object_aux: NamedTuple = None
    sampled_verts: Tensor = None
    contacts: Tensor = None
    partitions: Tensor = None

    def to(self, device):
        self.handarm_segs = self.handarm_segs.to(device)
        self.object_segs = self.object_segs.to(device)
        self.hand_verts = self.hand_verts.to(device)
        self.object_verts = self.object_verts.to(device)

        # to(device) of NamedTuple
        hand_faces = {
            field: getattr(self.hand_faces, field).to(device)
            for field in self.hand_faces._fields
        }
        self.hand_faces = _P3DFaces(**hand_faces)
        object_faces = {
            field: getattr(self.object_faces, field).to(device)
            for field in self.object_faces._fields
        }
        self.object_faces = _P3DFaces(**object_faces)

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


class ManualDataset(BaseDataset):
    def __init__(self, opt) -> None:
        super().__init__()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            logger.warning("CPU only, this will be slow!")

        self.image = opt.image
        self.hand = opt.hand
        self.object = opt.object
        self.nimble = opt.hand.nimble
        self.arm = opt.hand.arm

        # root xyz
        with open(opt.hand.xyz, 'r') as f:
            img_fidx = int(opt.image.path.split("/")[-1].split(".")[0])
            xyz = json.load(f)[img_fidx]
        self.root_xyz = torch.tensor(xyz[ROOT_JOINT_IDX])
        self.xyz = xyz

        self.mano_f = torch.from_numpy(
            pickle.load(open(self.hand.mano_model, "rb"), encoding="latin1")[
                "f"
            ].astype(np.int64),
        )

        # nimble
        nimble_pm_dict = np.load(opt.hand.nimble_pm_dict, allow_pickle=True)
        nimble_mano_vreg = np.load(opt.hand.nimble_mano_vreg, allow_pickle=True)
        self.skin_v = nimble_pm_dict["vert"][nimble_pm_dict["skin_v_sep"] :, :]
        self.skin_f = nimble_pm_dict["skin_f"]
        self.nimble_mano_vreg_fidx = nimble_mano_vreg["lmk_faces_idx"]
        self.nimble_mano_vreg_bc = nimble_mano_vreg["lmk_bary_coords"]
        self.nimble_jreg_mano = nimble_pm_dict["jreg_mano"]

        # smplx
        with open(opt.hand.smplx_model, "rb") as f:
            self.smplx_model = pickle.load(f, encoding="latin1")
        self.smplx_joint_idx = self.smplx_model["joint2num"].tolist()
        v_template = self.smplx_model["v_template"]
        J_regressor = self.smplx_model["J_regressor"]
        self.smplx_joints = torch.from_numpy(J_regressor @ v_template).to(device)
        # batch_joint_transform(rot=,joints=self.smplx_joints,parent=self.smplx_model["kintree_table"][0])
        with open(opt.hand.smplx_arm_corr, "rb") as f:
            self.smplx_arm_corr = pickle.load(f)

        # nimblearm preprocess
        (
            self.mano_a_verts,
            self.nimble_ha_verts_idx,
            self.mano_num_a_faces,
        ) = self.nimblearm(device)

        return

    def nimblearm(self, device):
        """
        Return
            verts: only arm (mano)
            faces: hand + wrist + arm (nimble+mano)
            number of arm faces (mano)
        """

        def closest_nimble_idx(device, faces, lmk_faces_idx, lmk_bary_coords):
            """
            For 778 mano vertices, find the closest nimble vertex indices
            """
            lmk_faces = torch.index_select(
                faces.to(device), 0, lmk_faces_idx.view(-1).to(device)
            ).view(-1, 3)
            lmk_bary_minidx = torch.argmin(lmk_bary_coords.to(device), dim=1)
            closest_idx = lmk_faces[torch.arange(lmk_faces.shape[0]), lmk_bary_minidx]
            return closest_idx.float()
            # return lmk_faces[...,0].float()

        nimble_h_faces = self.skin_f.to(device)

        # smplx       -> hand + arm
        smplx_verts = torch.tensor(self.smplx_model["v_template"], device=device)
        mano_ha_idx = self.smplx_arm_corr["arm_vert"]
        mano_ha_verts = smplx_verts[mano_ha_idx]
        mano_ha_faces = torch.tensor(self.smplx_arm_corr["face"], device=device)

        # hand + arm  -> arm
        mano_h_idx = self.smplx_arm_corr["mano_vert_from_arm"]
        mano_a_idx = list(set(range(mano_ha_verts.shape[0])) - set(mano_h_idx))
        mano_h_idx = torch.from_numpy(mano_h_idx).to(device)
        mano_a_idx = torch.tensor(mano_a_idx, dtype=torch.int64, device=device)
        mano_a_verts = mano_ha_verts[mano_a_idx]

        # mano - nimble mapping
        closest_idx = torch.cat(
            [
                closest_nimble_idx(
                    device,
                    nimble_h_faces,
                    self.nimble_mano_vreg_fidx[i],
                    self.nimble_mano_vreg_bc[i],
                ).unsqueeze(0)
                for i in range(1)  # 원래는 20
            ]
        )
        closest_idx = closest_idx.mean(0).long()

        # wrist + mano arm faces
        mano_a_faces = []
        for mano_ha_face in mano_ha_faces:
            is_a = [False, False, False]
            face_element = []

            # arm
            pack_idx = torch.nonzero(mano_a_idx == mano_ha_face[0])
            if pack_idx.nelement() > 0:
                pack_idx.squeeze_()
                assert mano_a_idx[pack_idx] == mano_ha_face[0]
                v0 = NIMBLE_N_VERTS + pack_idx
                is_a[0] = True
            pack_idx = torch.nonzero(mano_a_idx == mano_ha_face[1]).squeeze()
            if pack_idx.nelement() > 0:
                pack_idx.squeeze_()
                assert mano_a_idx[pack_idx] == mano_ha_face[1]
                v1 = NIMBLE_N_VERTS + pack_idx
                is_a[1] = True
            pack_idx = torch.nonzero(mano_a_idx == mano_ha_face[2]).squeeze()
            if pack_idx.nelement() > 0:
                pack_idx.squeeze_()
                assert mano_a_idx[pack_idx] == mano_ha_face[2]
                v2 = NIMBLE_N_VERTS + pack_idx
                is_a[2] = True

            if sum(is_a) == 0:  # all hand vertices
                continue

            # hand
            if is_a[0]:
                face_element.append(v0)
            else:
                pack_idx = torch.nonzero(mano_h_idx == mano_ha_face[0]).squeeze()
                assert mano_h_idx[pack_idx] == mano_ha_face[0]
                nim_idx = closest_idx[pack_idx]
                face_element.append(nim_idx)
            if is_a[1]:
                face_element.append(v1)
            else:
                pack_idx = torch.nonzero(mano_h_idx == mano_ha_face[1]).squeeze()
                assert mano_h_idx[pack_idx] == mano_ha_face[1]
                nim_idx = closest_idx[pack_idx]
                face_element.append(nim_idx)
            if is_a[2]:
                face_element.append(v2)
            else:
                pack_idx = torch.nonzero(mano_h_idx == mano_ha_face[2]).squeeze()
                assert mano_h_idx[pack_idx] == mano_ha_face[2]
                nim_idx = closest_idx[pack_idx]
                face_element.append(nim_idx)

            face_element = torch.stack(face_element)
            mano_a_faces.append(face_element)
        mano_a_faces = torch.stack(mano_a_faces)

        # nimble hand + wrist + mano arm faces
        nimble_ha_verts_idx = torch.cat([nimble_h_faces, mano_a_faces], dim=0)
        mano_num_a_faces = mano_a_faces.shape[0]

        return mano_a_verts, nimble_ha_verts_idx, mano_num_a_faces

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # image
        image = torch.from_numpy(
            cv2.cvtColor(cv2.imread(self.image.path), cv2.COLOR_BGR2RGB)
        )
        if os.path.exists(self.image.inpainted_path):
            inpainted_image = torch.from_numpy(
                cv2.cvtColor(cv2.imread(self.image.inpainted_path), cv2.COLOR_BGR2RGB)
            )
        else:
            inpainted_image = None
        intrinsics = torch.from_numpy(np.load(self.image.intrinsics))
        light = torch.load(self.image.light)
        handarm_seg = torch.from_numpy(np.array(Image.open(self.image.handarm_seg)))
        object_seg = torch.from_numpy(np.array(Image.open(self.image.object_seg)))

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
        mano_pose = torch.load(self.hand.mano_pose)

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
            images=image,
            intrinsics=intrinsics,
            light=light,
            handarm_segs=handarm_seg,
            object_segs=object_seg,
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            mano_pose=mano_pose,
            object_verts=object_verts,
            object_faces=object_faces,
            sampled_verts=sampled_verts,
            contacts=contacts,
            partitions=partitions,
        )

        # add only if not None
        if hand_aux is not None and object_aux is not None:
            hand_aux = {k: v for k, v in hand_aux._asdict().items() if v is not None}
            object_aux = {
                k: v for k, v in object_aux._asdict().items() if v is not None
            }
            return_dict["hand_aux"] = hand_aux
            return_dict["object_aux"] = object_aux
        if inpainted_image is not None:
            return_dict["inpainted_images"] = inpainted_image

        return return_dict
