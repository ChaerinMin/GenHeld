from submodules.NIMBLE_model.utils import vertices2landmarks
from torch.utils.data import Dataset
import torch
from collections import namedtuple
import logging
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import axis_angle_to_matrix


_P3DFaces = namedtuple(
    "_P3DFaces",
    ["verts_idx", "normals_idx", "textures_idx", "materials_idx"],
    defaults=(None,) * 4,
)  # Python 3.7+

NIMBLE_N_VERTS = 5990
ROOT_JOINT_IDX = 9
# NIMBLE_ROOT_ID = 11
logger = logging.getLogger(__name__)

class HandDataset(Dataset):
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

    def nimble_to_nimblearm(self, pose, h_verts, h_faces):
        """
        h_verts: torch.tensor B x V x 3
        """
        batch_size = h_verts.shape[0]

        # batching
        mano_a_verts = self.mano_a_verts
        mano_a_verts = mano_a_verts.unsqueeze(0).expand(batch_size, -1, -1)

        # rotate arm
        smplx_wrist_loc = self.smplx_joints[self.smplx_joint_idx["R_Wrist"]][None, :]
        smplx_middle1_loc = self.smplx_joints[self.smplx_joint_idx["R_Middle1"]][
            None, :
        ]
        # hand_joints = torch.bmm(self.nimble_jreg_mano.unsqueeze(0), h_verts)
        # hand_wrist_loc = hand_joints[:, 0]
        # hand_middle1_loc = hand_joints[:, 9]
        hand_wrist_loc = torch.tensor(self.xyz[0]).unsqueeze(0).to(h_verts.device)
        hand_middle1_loc = torch.tensor(self.xyz[9]).unsqueeze(0).to(h_verts.device)
        smplx_rot = smplx_middle1_loc - smplx_wrist_loc
        smplx_rot = smplx_rot / smplx_rot.norm(dim=1, keepdim=True)
        hand_rot = hand_middle1_loc - hand_wrist_loc
        hand_rot = hand_rot / hand_rot.norm(dim=1, keepdim=True)
        rel_rot_axis = torch.cross(hand_rot, smplx_rot, dim=1)
        rel_rot_angle = torch.acos(torch.sum(hand_rot * smplx_rot, dim=1, keepdim=True))
        rel_rot = rel_rot_axis * rel_rot_angle
        # root_rot = pose[:, 0].to(h_verts.device)
        root_rot = axis_angle_to_matrix(rel_rot)
        t = Transform3d(device=mano_a_verts.device).rotate(root_rot)
        mano_a_verts = t.transform_points(
            mano_a_verts - smplx_wrist_loc.unsqueeze(1)
        ) + hand_wrist_loc.unsqueeze(1)
        #+ self.root_xyz[None, None, :].to(mano_a_verts.device)

        # verts, faces
        nimble_ha_verts = torch.cat([h_verts, mano_a_verts], dim=1)
        nimble_ha_vt = torch.cat(
            [
                h_faces.textures_idx,
                torch.ones(
                    batch_size,
                    self.mano_num_a_faces,
                    3,
                    dtype=h_faces.textures_idx.dtype,
                    device=h_verts.device,
                ),
            ],
            dim=1,
        )
        nimble_ha_faces = _P3DFaces(
            verts_idx=self.nimble_ha_verts_idx.unsqueeze(0),
            textures_idx=nimble_ha_vt,
        )
        return nimble_ha_verts, nimble_ha_faces
    
class ObjectDataset(Dataset):
    def __init__(self) -> None:
        return

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

    def nimble_to_nimblearm(self, pose, h_verts, h_faces):
        """
        h_verts: torch.tensor B x V x 3
        """
        batch_size = h_verts.shape[0]

        # batching
        mano_a_verts = self.mano_a_verts
        mano_a_verts = mano_a_verts.unsqueeze(0).expand(batch_size, -1, -1)

        # rotate arm
        smplx_wrist_loc = self.smplx_joints[self.smplx_joint_idx["R_Wrist"]][None, :]
        smplx_middle1_loc = self.smplx_joints[self.smplx_joint_idx["R_Middle1"]][
            None, :
        ]
        # hand_joints = torch.bmm(self.nimble_jreg_mano.unsqueeze(0), h_verts)
        # hand_wrist_loc = hand_joints[:, 0]
        # hand_middle1_loc = hand_joints[:, 9]
        hand_wrist_loc = torch.tensor(self.xyz[0]).unsqueeze(0).to(h_verts.device)
        hand_middle1_loc = torch.tensor(self.xyz[9]).unsqueeze(0).to(h_verts.device)
        smplx_rot = smplx_middle1_loc - smplx_wrist_loc
        smplx_rot = smplx_rot / smplx_rot.norm(dim=1, keepdim=True)
        hand_rot = hand_middle1_loc - hand_wrist_loc
        hand_rot = hand_rot / hand_rot.norm(dim=1, keepdim=True)
        rel_rot_axis = torch.cross(hand_rot, smplx_rot, dim=1)
        rel_rot_angle = torch.acos(torch.sum(hand_rot * smplx_rot, dim=1, keepdim=True))
        rel_rot = rel_rot_axis * rel_rot_angle
        # root_rot = pose[:, 0].to(h_verts.device)
        root_rot = axis_angle_to_matrix(rel_rot)
        t = Transform3d(device=mano_a_verts.device).rotate(root_rot)
        mano_a_verts = t.transform_points(
            mano_a_verts - smplx_wrist_loc.unsqueeze(1)
        ) + hand_wrist_loc.unsqueeze(1)
        #+ self.root_xyz[None, None, :].to(mano_a_verts.device)

        # verts, faces
        nimble_ha_verts = torch.cat([h_verts, mano_a_verts], dim=1)
        nimble_ha_vt = torch.cat(
            [
                h_faces.textures_idx,
                torch.ones(
                    batch_size,
                    self.mano_num_a_faces,
                    3,
                    dtype=h_faces.textures_idx.dtype,
                    device=h_verts.device,
                ),
            ],
            dim=1,
        )
        nimble_ha_faces = _P3DFaces(
            verts_idx=self.nimble_ha_verts_idx.unsqueeze(0),
            textures_idx=nimble_ha_vt,
        )
        return nimble_ha_verts, nimble_ha_faces
