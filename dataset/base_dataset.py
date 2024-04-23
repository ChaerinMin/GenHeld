import glob
import logging
import os
import pickle
import random

import cv2
import mediapipe
import numpy as np
import open3d as o3d
import pymeshlab as ml
import torch
from iopath.common.file_io import PathManager
from matplotlib import pyplot as plt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pytorch3d.io import load_obj, load_ply, save_obj
from pytorch3d.io.experimental_gltf_io import load_meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix
from rich.console import Console
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize

from dataset import _P3DFaces
from submodules.HiFiHR.models_res_nimble import Model as HiFiHRModel
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer
from submodules.HiFiHR.utils.NIMBLE_model.utils import save_hifihr_mesh
from submodules.HiFiHR.utils.train_utils import load_hifihr
from submodules.NIMBLE_model.utils import vertices2landmarks
from utils.joints import mediapipe_to_kp
from visualization import bones
from visualization.keypoints import vis_keypoints

NIMBLE_N_VERTS = 5990
ROOT_JOINT_IDX = 9
# NIMBLE_ROOT_ID = 11
logger = logging.getLogger(__name__)
console = Console()


class HandDataset(Dataset):
    def __init__(self, opt, cfg, device) -> None:
        self.cfg = cfg
        self.device = device
        self.opt = opt
        self.image = opt.image
        self.hand = opt.hand
        self.cached = opt.cached

        # hifihr
        hand_type = "nimble" if self.hand.nimble else "mano"
        self.hifihr_model = HiFiHRModel(
            ifRender=False,
            device=self.device,
            if_4c=False,
            hand_model=hand_type,
            use_mean_shape=False,
            pretrain="effb3",
            root_id=9,
            root_id_nimble=11,
            ifLight=True,
        )
        self.hifihr_model.to(device)
        self.hifihr_model = load_hifihr(
            self.device, self.hifihr_model, self.hand.hifihr_pretrained
        )
        self.hifihr_model.eval()
        self.manolayer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side="right", use_pca=False
        )

        # mano
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

        # smplx
        with open(opt.hand.smplx_model, "rb") as f:
            self.smplx_model = pickle.load(f, encoding="latin1")
        self.smplx_joint_idx = self.smplx_model["joint2num"].tolist()
        v_template = self.smplx_model["v_template"]
        J_regressor = self.smplx_model["J_regressor"]
        self.smplx_joints = torch.from_numpy(J_regressor @ v_template).to(self.device)
        with open(opt.hand.smplx_arm_corr, "rb") as f:
            self.smplx_arm_corr = pickle.load(f)

        # nimblearm preprocess
        (
            self.mano_a_verts,
            self.nimble_ha_verts_idx,
            self.mano_num_a_faces,
        ) = self.nimblearm(self.device)

        # mediapipe
        base_options = mp_python.BaseOptions(model_asset_path=opt.image.mediapipe)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.16,
            min_hand_presence_confidence=0.16,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            options=options
        )

        # SAM
        with console.status("Setting up the SAM...", spinner="monkey"):
            sam = sam_model_registry["vit_h"](checkpoint=opt.image.sam)
            sam = sam.to(device)
            self.sam_segmenter = SamPredictor(sam)

        return

    def __len__(self):
        return len(self.fidxs)

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]
        verts_color = plt.cm.viridis(np.linspace(0, 1, 778))[:, :-1]
        joint_color = plt.cm.gist_rainbow(np.linspace(0, 1, 21))[:, :-1]

        # image
        image = cv2.cvtColor(cv2.imread(self.image.path % fidx), cv2.COLOR_BGR2RGB)
        inpainted_image = None
        if self.cfg.vis.where_to_render == "raw":
            pass
        elif self.cfg.vis.where_to_render == "inpainted":
            if os.path.exists(self.cached.image.inpainted % fidx):
                inpainted_image = torch.from_numpy(
                    cv2.cvtColor(
                        cv2.imread(self.cached.image.inpainted % fidx),
                        cv2.COLOR_BGR2RGB,
                    )
                )
        else:
            logger.error(f"Invalid where_to_render {self.cfg.vis.where_to_render}")
            raise ValueError

        # seg
        if not os.path.exists(self.cached.image.seg % fidx) or self.opt.refresh:
            seg_dir = os.path.dirname(self.cached.image.seg % fidx)
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)
            # hand keypoints
            kp = self.hand_landmarker.detect(
                mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image)
            )
            kp = mediapipe_to_kp(kp, image.shape[:2])
            # SAM
            kp_box = np.stack([kp.min(0), kp.max(0)])
            kp_box_center = (kp_box[0] + kp_box[1]) / 2
            kp_box = (kp_box - kp_box_center[None, :]) * 1.3 + kp_box_center[None, :]
            kp_box = kp_box.astype(np.int32).reshape(-1)
            kp_label = np.ones(kp.shape[0])
            self.sam_segmenter.set_image(image)
            seg, *_ = self.sam_segmenter.predict(
                point_coords=kp,
                point_labels=kp_label,
                box=kp_box,
                multimask_output=False)
            seg = seg.squeeze(0)
            vis_seg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            vis_seg[seg] = vis_seg[seg] * 0.7 + np.array([0, 0, 255]) * 0.3
            seg = seg.astype(np.uint8) * 255
            # save
            cv2.imwrite(self.cached.image.seg % fidx, seg)
            logger.info(f"Saved {self.cached.image.seg % fidx}")        
            # visualization
            cv2.imwrite(self.cached.image.seg_vis % fidx, vis_seg)
            vis_kp = vis_keypoints(image, kp)
            vis_box = np.copy(image)
            vis_box = cv2.rectangle(vis_box, kp_box[:2], kp_box[2:], (0, 0, 255), 2)
            vis_kp_path = self.cached.image.seg_vis.replace(".", "_kp.") % fidx
            vis_box_path = self.cached.image.seg_vis.replace(".", "_box.") % fidx
            cv2.imwrite(vis_kp_path, vis_kp[:,:, ::-1])
            cv2.imwrite(vis_box_path, vis_box[:,:,::-1])
        seg = cv2.imread(self.cached.image.seg % fidx, cv2.IMREAD_GRAYSCALE)
        seg = torch.from_numpy(seg)
        if seg.shape[0] != image.shape[0]:
            logger.warning("Resizing seg to match image size")
            seg = Resize(image.shape[0])(seg[None, None, ...])[0][0]
        image = torch.from_numpy(image)


        # image -> hand mesh
        hand_path = self.cached.hand.path % fidx
        mano_r_path = self.cached.mano_r.path % fidx
        mano_joint_r_path = self.cached.mano_r.joint % fidx
        bone_vis_path = self.cached.mano_r.bone_vis % fidx
        joint_vis_path = self.cached.mano_r.joint_vis % fidx
        xyz_save_path = self.cached.hand.xyz % fidx
        if (
            not os.path.exists(hand_path)
            or not os.path.exists(xyz_save_path)
            or not os.path.exists(mano_r_path)
            or not os.path.exists(mano_joint_r_path)
            # or self.opt.refresh
        ):
            image_hifihr = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
            if image_hifihr.shape[2] != self.hifihr_image_size:
                image_hifihr = Resize(self.hifihr_image_size)(image_hifihr)
            image_hifihr = image_hifihr.to(torch.float32) / 255.0
            with torch.no_grad():
                output = self.hifihr_model(
                    "FreiHand", mode_train=False, images=image_hifihr
                )
            hand_mesh = output["skin_meshes"]
            hand_textures = output["textures"]
            mano_verts_r = output["mano_verts_r"].squeeze(0)
            mano_verts_r[:, 2] *= -1
            mano_verts_r[:, [0, 2]] = mano_verts_r[:, [2, 0]]
            mano_verts_r = mano_verts_r.cpu().numpy()
            mano_joints_r = output["mano_joints_r"].squeeze(0)
            mano_joints_r[:, 2] *= -1
            mano_joints_r[:, [0, 2]] = mano_joints_r[:, [2, 0]]
            mano_joints_r = mano_joints_r.cpu().numpy()
            xyz = output["xyz"]

            # save
            save_hifihr_mesh(
                hand_path,
                xyz_save_path,
                self.hand.nimble_tex_fuv,
                hand_mesh.verts_padded()[0].cpu().numpy(),
                hand_textures[0].cpu().numpy(),
                xyz[0].cpu().numpy(),
            )
            vertices = o3d.utility.Vector3dVector(mano_verts_r)
            faces = o3d.utility.Vector3iVector(self.mano_f)
            mesh = o3d.geometry.TriangleMesh(vertices, faces)
            mesh.vertex_colors = o3d.utility.Vector3dVector(verts_color)
            o3d.io.write_triangle_mesh(mano_r_path, mesh)
            np.save(mano_joint_r_path, mano_joints_r)
            logger.info(f"Saved {hand_path}")
            logger.info(f"Saved {xyz_save_path}")
            logger.info(f"Saved {mano_r_path}")
            logger.info(f"Saved {mano_joint_r_path}")

            # visualize
            kp = o3d.utility.Vector3dVector(mano_joints_r)
            lines = o3d.utility.Vector2iVector(bones)
            line_set = o3d.geometry.LineSet(kp, lines)
            keypoints = o3d.geometry.PointCloud(kp)
            keypoints.colors = o3d.utility.Vector3dVector(joint_color)
            o3d.io.write_line_set(bone_vis_path, line_set)
            o3d.io.write_point_cloud(joint_vis_path, keypoints)
            logger.info(f"Saved {bone_vis_path}")
            logger.info(f"Saved {joint_vis_path}")
        hand_verts, hand_faces, hand_aux = load_obj(hand_path, load_textures=True)
        mano_verts_r, _, _ = load_obj(mano_r_path, load_textures=False)
        mano_joints_r = torch.from_numpy(np.load(mano_joint_r_path))
        xyz = np.load(xyz_save_path)

        # return
        return_dict = dict(
            fidxs=fidx,
            images=image,
            seg=seg,
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            mano_verts_r=mano_verts_r,
            mano_joints_r=mano_joints_r,
            xyz=xyz,
        )

        # add only if not None
        if hand_aux is not None:
            hand_aux = {k: v for k, v in hand_aux._asdict().items() if v is not None}
            return_dict["hand_aux"] = hand_aux
        if inpainted_image is not None:
            return_dict["inpainted_images"] = inpainted_image

        return return_dict

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
                for i in range(1)
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

    def nimble_to_nimblearm(self, xyz, h_verts, h_faces):
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
        hand_wrist_loc = xyz[:, 0]
        hand_middle1_loc = xyz[:, 9]
        smplx_rot = smplx_middle1_loc - smplx_wrist_loc
        smplx_rot = smplx_rot / smplx_rot.norm(dim=1, keepdim=True)
        hand_rot = hand_middle1_loc - hand_wrist_loc
        hand_rot = hand_rot / hand_rot.norm(dim=1, keepdim=True)
        rel_rot_axis = torch.cross(hand_rot.to(smplx_rot.device), smplx_rot, dim=1)
        rel_rot_angle = torch.acos(torch.sum(hand_rot * smplx_rot, dim=1, keepdim=True))
        rel_rot = rel_rot_axis * rel_rot_angle
        root_rot = axis_angle_to_matrix(rel_rot)
        t = Transform3d(device=mano_a_verts.device).rotate(root_rot)
        mano_a_verts = t.transform_points(
            mano_a_verts - smplx_wrist_loc.unsqueeze(1)
        ) + hand_wrist_loc.unsqueeze(1)

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
            verts_idx=self.nimble_ha_verts_idx.unsqueeze(0).repeat(batch_size, 1, 1),
            textures_idx=nimble_ha_vt,
        )
        return nimble_ha_verts, nimble_ha_faces


class ObjectDataset(Dataset):
    def __init__(self, opt) -> None:
        self.object = opt
        return

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]

        # object
        object_ext = os.path.splitext(self.object.path % fidx)[1]
        glb_path = os.path.splitext(self.object.path % fidx)[0] + ".glb"
        texture_path = os.path.splitext(self.object.path % fidx)[0] + ".png"
        if object_ext == ".obj":
            if os.path.exists(self.object.path % fidx) and not self.object.refresh:
                # texture size
                if self.object.texture_size is not None:
                    texture_img = cv2.imread(texture_path)
                    if texture_img is None:
                        texture_path = (
                            os.path.splitext(self.object.path % fidx)[0] + ".jpg"
                        )
                        texture_img = cv2.imread(texture_path)
                    if texture_img is None:
                        logger.error(f"Texture image {texture_path} not found")
                        raise FileNotFoundError
                    if texture_img.shape[:2] != self.object.texture_size:
                        texture_img = cv2.resize(texture_img, self.object.texture_size)
                        assert cv2.imwrite(texture_path, texture_img), "Failed to save"
                # load
                try:
                    object_verts_highres, object_faces_highres, object_aux_highres = (
                        load_obj(self.object.path % fidx, load_textures=True)
                    )
                except Exception as e:
                    logger.error(f"Cannot load {self.object.path % fidx}")
                    raise e
            elif os.path.exists(glb_path):
                # try:
                object_meshes = load_meshes(glb_path, PathManager())
                for i in range(len(object_meshes)):
                    object_meshes[i][1].textures.align_corners = False
                object_meshes = join_meshes_as_scene(
                    [mesh[1] for mesh in object_meshes]
                )
                object_verts_highres = object_meshes.verts_padded()[0]
                verts_idx = object_meshes.faces_padded()[0]
                verts_uvs = object_meshes.textures.verts_uvs_padded()[0]
                faces_uvs = object_meshes.textures.faces_uvs_padded()[0]
                assert (
                    len(object_meshes.textures.maps_padded()) == 1
                ), "Only support one texture image for each mesh"
                texture_images = object_meshes.textures.maps_padded()[0]
                resizer = transforms.Resize(self.object.texture_size, antialias=True)
                texture_images = (
                    resizer(texture_images.permute(2, 0, 1))
                    .permute(1, 2, 0)
                    .contiguous()
                )
                normals = object_meshes.verts_normals_padded()[0]
                normals_idx = object_meshes.faces_normals_padded()[0]
                save_obj(
                    self.object.path % fidx,
                    object_verts_highres,
                    verts_idx,
                    verts_uvs=verts_uvs,
                    faces_uvs=faces_uvs,
                    texture_map=texture_images,
                    normals=normals,
                    faces_normals_idx=normals_idx,
                )
                object_verts_highres, object_faces_highres, object_aux_highres = (
                    load_obj(self.object.path % fidx, load_textures=True)
                )
            else:
                logger.error(f"Object file {self.object.path % fidx} not found")
                raise FileNotFoundError
        elif object_ext == ".ply":
            if self.nimble or self.object.deci:
                logger.error(
                    "We only support .obj object when nimble=True or deci=True"
                )
                raise ValueError
            object_verts_highres, object_faces_highres = load_ply(
                self.object.path % fidx
            )
            object_aux_highres = None
        else:
            raise ValueError(f"object file extension {object_ext} not supported")

        # decimation
        if (
            self.object.deci
            and object_faces_highres.verts_idx.shape[0] > self.object.deci_faces
        ):
            deci_path = os.path.splitext(self.object.path % fidx)[0] + "_deci.obj"
            if not os.path.exists(deci_path) or self.object.refresh:
                deci = ml.MeshSet()
                deci.load_new_mesh(self.object.path % fidx)
                deci.meshing_decimation_quadric_edge_collapse_with_texture(
                    targetfacenum=self.object.deci_faces, preserveboundary=True
                )
                perc = 2
                cnt = 1
                while deci.mesh(0).face_number() > self.object.deci_faces:
                    deci.meshing_decimation_clustering(
                        threshold=ml.PercentageValue(perc * cnt)
                    )
                    cnt += 1
                    if cnt > 3:
                        break
                deci.save_current_mesh(deci_path)
                # add texture image
                with open(deci_path + ".mtl", "r") as f:
                    lines = f.readlines()
                if not any("map_Kd" in line for line in lines):
                    with open(deci_path + ".mtl", "a") as f:
                        line = f"map_Kd {os.path.basename(texture_path)}\n"
                        f.write(line)
                logger.info(f"Decimated {self.object.path % fidx} to {deci_path}")
            object_verts, object_faces, object_aux = load_obj(
                deci_path, load_textures=True
            )
            if self.object.deci_as_output:
                object_verts_highres = object_verts
                object_faces_highres = object_faces
                object_aux_highres = object_aux
        else:
            object_verts = object_verts_highres
            object_faces = object_faces_highres
            object_aux = object_aux_highres

        fidx = fidx.replace("/", "_")
        return_dict = dict(
            fidx=fidx,
            object_verts=object_verts,
            object_verts_highres=object_verts_highres,
            object_faces=object_faces,
            object_faces_highres=object_faces_highres,
        )

        # add only if not None
        if object_aux is not None:
            object_aux = {
                k: v for k, v in object_aux._asdict().items() if v is not None
            }
            object_aux_highres = {
                k: v for k, v in object_aux_highres._asdict().items() if v is not None
            }
            return_dict["object_aux"] = object_aux
            return_dict["object_aux_highres"] = object_aux_highres

        # ContactGen
        if self.object.sampled_verts_path:
            contactgen_fidxs = sorted(
                [
                    int(os.path.basename(p).split(".")[0])
                    for p in glob.glob(self.object.sampled_verts_path)
                ]
            )
            contactgen_fidx = random.choice(contactgen_fidxs)
            sampled_verts = torch.from_numpy(
                np.load(self.object.sampled_verts_path % (fidx, contactgen_fidx))
            )
            contacts = torch.from_numpy(
                np.load(self.object.contacts_path % (fidx, contactgen_fidx))
            )
            partitions = torch.from_numpy(
                np.load(self.object.partitions_path % (fidx, contactgen_fidx))
            )
            return_dict["sampled_verts"] = sampled_verts
            return_dict["contacts"] = contacts
            return_dict["partitions"] = partitions

        return return_dict


class SelectorDataset(Dataset):
    def __init__(self, opt, cfg):
        self.opt = opt
        return

    def __len__(self):
        return len(self.fidxs)

    def __getitem__(self, idx):
        raise NotImplementedError


class SelectorTestDataset(Dataset):
    def __init__(self, hand_fidxs, hand_verts_r, hand_joints_r):
        self.hand_fidxs = hand_fidxs
        self.hand_verts_r = hand_verts_r
        self.hand_joints_r = hand_joints_r
        return

    def __len__(self):
        return self.hand_fidxs.shape[0]

    def __getitem__(self, idx):
        return_dict = dict(
            hand_fidxs=self.hand_fidxs[idx],
            hand_verts_r=self.hand_verts_r[idx],
            hand_joints_r=self.hand_joints_r[idx],
        )
        return return_dict
