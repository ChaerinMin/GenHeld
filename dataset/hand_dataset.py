import logging
from torch import Tensor
import torch
from typing import NamedTuple
from typing import Dict
from .base_dataset import _P3DFaces
import cv2
import os 
import numpy as np
from PIL import Image
from dataclasses import dataclass
from .base_dataset import HandDataset
from pytorch3d.io import load_obj, load_ply
from .base_dataset import NIMBLE_N_VERTS

logger = logging.getLogger(__name__)

@dataclass
class HandData:
    images: Tensor
    intrinsics: Tensor
    light: Dict[str, Tensor]
    handarm_segs: Tensor
    hand_verts: Tensor
    hand_faces: NamedTuple
    mano_pose: Tensor = None
    inpainted_images: Tensor = None
    hand_aux: NamedTuple = None

    def to(self, device):
        self.handarm_segs = self.handarm_segs.to(device)
        self.hand_verts = self.hand_verts.to(device)

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


class FreiHANDDataset(HandDataset):
    def __init__(self):
        super().__init__()

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
        return
    
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

        return_dict = dict(
            images=image,
            object_segs=object_seg, 
            intrinsics=intrinsics,
            light=light,
            handarm_segs=handarm_seg,
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            mano_pose=mano_pose,
        )

        # add only if not None
        if hand_aux is not None:
            hand_aux = {k: v for k, v in hand_aux._asdict().items() if v is not None}
            return_dict["hand_aux"] = hand_aux
        if inpainted_image is not None:
            return_dict["inpainted_images"] = inpainted_image

        return return_dict