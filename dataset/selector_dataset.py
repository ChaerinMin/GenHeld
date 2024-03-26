import glob
import logging
import os
import pickle

import numpy as np
import open3d as o3d
import torch
import yaml
from matplotlib import pyplot as plt
from pytorch3d.io import IO, load_obj
from pytorch3d.ops import sample_farthest_points
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Rotate, axis_angle_to_matrix

from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer
from submodules.HiFiHR.utils.NIMBLE_model.myNIMBLELayer import mano_v2j_reg
from utils import get_NN
from visualization import bone_colors, bones

from .base_dataset import SelectorDataset

logger = logging.getLogger(__name__)


class DexYCBDataset(SelectorDataset):
    categories = [
        set([15, 12, 10]),  # extra big box/bowl/bottle
        set([0, 3, 1, 11]),  # big box/bowl/bottle
        set([5, 8, 13, 2, 4, 6, 20, 14]),  # small box/bowl/bottle
        set([7]),  # extra small box/bowl/bottle
        set([9]),  # narrow
        set([16, 17, 18, 19]),  # very narrow
    ]

    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        self.cfg = cfg
        # prepare
        manolayer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side="right", use_pca=True
        )
        self.mano_f = pickle.load(open(opt.mano_model, "rb"), encoding="latin1")["f"]
        self.mano_f = torch.from_numpy(self.mano_f.astype(np.int64))
        verts_color = plt.cm.viridis(np.linspace(0, 1, 778))[:, :-1]
        joint_color = plt.cm.gist_rainbow(np.linspace(0, 1, 21))[:, :-1]
        object_color = plt.cm.gist_rainbow(
            np.linspace(0, 1, self.cfg.select_object.opt.n_obj_points)
        )[:, :-1]

        # save object names
        object_paths = sorted(glob.glob(os.path.join(self.opt.path, "models", "*")))
        object_names_path = os.path.join(cfg.selector_ckpt_dir, "selector_objects.yaml")
        object_names = []
        for cate in DexYCBDataset.categories:
            obj_names = []
            for idx in cate:
                obj_names.append(os.path.basename(object_paths[idx]))
            object_names.append(obj_names)
        with open(object_names_path, "w") as f:
            yaml.safe_dump(object_names, f)
            logger.info(f"Saved {object_names_path}")

        self.fidxs = []
        self.class2fidxs = {}
        self.class_vecs = []
        self.hand_verts_r = []
        self.hand_joints_r = []
        self.object_pcs_r = []
        self.hand_contacts_r = []
        subjects = sorted(glob.glob(os.path.join(self.opt.path, "2020*-subject-*/")))
        for subject in subjects:
            # hand beta
            mano_calibs = glob.glob(
                os.path.join(
                    self.opt.path, "calibration", f"mano*subject-{subject[-3:-1]}*"
                )
            )
            assert len(mano_calibs) == 1
            with open(os.path.join(mano_calibs[0], "mano.yml"), "r") as f:
                hand_beta = torch.tensor(yaml.safe_load(f)["betas"])

            # load
            for sequence in sorted(glob.glob(os.path.join(subject, "2020*"))):
                # meta.yml
                with open(os.path.join(sequence, "meta.yml"), "r") as f:
                    meta = yaml.safe_load(f)
                if meta["mano_sides"][0] == "left":
                    logger.info(f"Skip {sequence} because it is left hand")
                    continue

                # fidxs
                seq = sequence.replace(self.opt.path, "")
                self.fidxs.append(seq)

                # class_vec
                class_vec = [0] * cfg.select_object.opt.n_class
                class_path = os.path.join(sequence, "class_gt.txt")
                if os.path.exists(class_path) and not opt.refresh_data:
                    with open(class_path, "r") as f:
                        cate_idx = int(f.read())
                    logger.info(f"Loaded {class_path}")
                else:
                    object_idx = meta["ycb_ids"][meta["ycb_grasp_ind"]]
                    object_idx -= 1  # 1-index -> 0-index
                    cate_idx = -1
                    for i, cate in enumerate(DexYCBDataset.categories):
                        if object_idx in cate:
                            cate_idx = i
                    if cate_idx < 0:
                        logger.warning(f"Unknown category for {object_idx}. Skipped.")
                        continue
                    with open(class_path, "w") as f:
                        f.write(str(cate_idx))
                    logger.info(f"Saved {class_path}")
                class_vec[cate_idx] = 1
                class_vec = torch.tensor(class_vec)
                self.class_vecs.append(class_vec)
                if cate_idx not in self.class2fidxs:
                    self.class2fidxs[cate_idx] = [seq]
                else:
                    self.class2fidxs[cate_idx].append(seq)

                # hand_verts, joints
                hand_mesh_r_path = os.path.join(sequence, "hand_mesh_r.obj")
                hand_joints_r_path = os.path.join(sequence, "hand_joint_r.npy")
                bone_vis_path = os.path.join(sequence, "hand_bone_r_vis.ply")
                joint_vis_path = os.path.join(sequence, "hand_joint_r_vis.ply")
                if os.path.exists(hand_mesh_r_path) and not opt.refresh_data:
                    hand_verts_r, _, _ = load_obj(hand_mesh_r_path, load_textures=False)
                    hand_joints_r = torch.tensor(np.load(hand_joints_r_path))
                    logger.info(f"Loaded {hand_mesh_r_path}")
                    logger.info(f"Loaded {hand_joints_r_path}")
                else:
                    # verts
                    serial = meta["serials"][0]
                    frame = meta["num_frames"] - 1
                    pose_info = np.load(
                        os.path.join(sequence, serial, f"labels_{frame:06d}.npz")
                    )
                    hand_theta = torch.tensor(pose_info["pose_m"][0])
                    hand_theta_r = hand_theta.clone()
                    hand_theta_r[:3] = 0.0
                    hand_verts_r, hand_joints_old = manolayer(
                        th_pose_coeffs=hand_theta_r[None, :48],
                        th_betas=hand_beta[None, :],
                    )
                    # joints
                    hand_joints_r = mano_v2j_reg(hand_verts_r)
                    hand_verts_r = hand_verts_r.squeeze(0)
                    hand_joints_r = hand_joints_r.squeeze(0)
                    # save
                    vertices = o3d.utility.Vector3dVector(hand_verts_r)
                    faces = o3d.utility.Vector3iVector(self.mano_f)
                    mesh = o3d.geometry.TriangleMesh(vertices, faces)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(verts_color)
                    o3d.io.write_triangle_mesh(hand_mesh_r_path, mesh)
                    np.save(hand_joints_r_path, hand_joints_r.numpy())
                    logger.info(f"Saved {hand_mesh_r_path}")
                    logger.info(f"Saved {hand_joints_r_path}")
                    # visualize joints
                    kp = o3d.utility.Vector3dVector(hand_joints_r)
                    lines = o3d.utility.Vector2iVector(bones)
                    colors = np.array(bone_colors).astype(np.float64)
                    line_set = o3d.geometry.LineSet(kp, lines)
                    keypoints = o3d.geometry.PointCloud(kp)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    assert line_set.has_colors()
                    keypoints.colors = o3d.utility.Vector3dVector(joint_color)
                    o3d.io.write_line_set(bone_vis_path, line_set)
                    o3d.io.write_point_cloud(joint_vis_path, keypoints)
                    logger.info(f"Saved {bone_vis_path}")
                    logger.info(f"Saved {joint_vis_path}")
                hand_verts_r = Pointclouds(points=[hand_verts_r])
                hand_verts_r.estimate_normals(assign_to_self=True)
                self.hand_verts_r.append(hand_verts_r)
                self.hand_joints_r.append(hand_joints_r)

                # object_pcs
                object_r_path = os.path.join(sequence, "object_r.ply")
                object_vis_path = os.path.join(sequence, "object_r_vis.ply")
                io = IO()
                if os.path.exists(object_r_path) and not opt.refresh_data:
                    object_pc_r = io.load_pointcloud(object_r_path)
                    logger.info(f"Loaded {object_r_path}")
                else:
                    object_path = object_paths[object_idx]
                    object_pth = os.path.join(object_path, "textured_simple.obj")
                    verts, _, _ = load_obj(object_pth, load_textures=False)
                    # object pose
                    object_6dof = torch.tensor(pose_info["pose_y"])[
                        meta["ycb_grasp_ind"]
                    ]
                    object_6dof = torch.vstack(
                        [
                            object_6dof,
                            torch.tensor(
                                [0, 0, 0, 1], dtype=object_6dof.dtype
                            ).unsqueeze(0),
                        ]
                    )
                    verts = (object_6dof[:3, :3] @ verts.t() + object_6dof[:3, 3:4]).t()
                    # sample points
                    verts = sample_farthest_points(
                        verts[None, ...],
                        torch.tensor([verts.shape[0]]),
                        self.cfg.select_object.opt.n_obj_points,
                    )[0][0]
                    # rotation normalize
                    verts_r = verts - (
                        hand_theta[48:51] + hand_joints_r[0]
                    ).unsqueeze(0)
                    t = Rotate(axis_angle_to_matrix(hand_theta[:3]), dtype=verts.dtype)
                    verts_r = t.transform_points(verts_r.unsqueeze(0)).squeeze(0)
                    verts_r = verts_r + hand_joints_r[0].unsqueeze(0)
                    # sort by z
                    verts_r = torch.index_select(
                        verts_r, 0, torch.sort(verts_r[:, 2])[1]
                    )
                    # save
                    object_pc_r = Pointclouds(points=[verts_r])
                    object_pc_r.estimate_normals(assign_to_self=True)
                    io.save_pointcloud(object_pc_r, object_r_path)
                    logger.info(f"Saved {object_r_path}")
                    # visualize
                    pc = o3d.utility.Vector3dVector(verts_r.numpy())
                    pc = o3d.geometry.PointCloud(pc)
                    pc.colors = o3d.utility.Vector3dVector(object_color)
                    o3d.io.write_point_cloud(object_vis_path, pc)
                    logger.info(f"Saved {object_vis_path}")
                self.object_pcs_r.append(object_pc_r)

                # hand_contacts
                contact_vis_path = os.path.join(sequence, "hand_contact.ply")
                contact_cache_path = os.path.join(sequence, "hand_contact_cache.pt")
                if os.path.exists(contact_cache_path) and not opt.refresh_data:
                    hand_contact_r = torch.load(contact_cache_path)
                    logger.info(f"Loaded {contact_cache_path}")
                else:
                    nn, _ = get_NN(
                        hand_verts_r.points_padded(), object_pc_r.points_padded()
                    )
                    nn = 10.0 * torch.sqrt(nn * 1000)
                    hand_contact_r = 1.0 - 2 * (torch.sigmoid(nn) - 0.5)
                    hand_contact_r = hand_contact_r.squeeze(0)
                    # save
                    torch.save(hand_contact_r, contact_cache_path)
                    logger.info(f"Saved {contact_cache_path}")
                    # visualize
                    hand_contact_vis = hand_verts_r.points_padded()[0][
                        hand_contact_r > 0.5
                    ]
                    assert hand_contact_vis.shape[0] > 0, "No contact points."
                    hand_contact_vis = o3d.utility.Vector3dVector(
                        hand_contact_vis.numpy()
                    )
                    hand_contact_vis = o3d.geometry.PointCloud(hand_contact_vis)
                    hand_contact_vis.paint_uniform_color([1, 0, 0])
                    o3d.io.write_point_cloud(contact_vis_path, hand_contact_vis)
                    logger.info(f"Saved {contact_vis_path}")
                self.hand_contacts_r.append(hand_contact_r)

        # class weights
        self.class_weights = torch.zeros(len(self.class2fidxs))
        for k in np.sort(list(self.class2fidxs.keys())):
            self.class_weights[k] = 1.0 / len(self.class2fidxs[k])

        return

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]
        return_dict = dict(
            fidxs=fidx,
            hand_verts_r=self.hand_verts_r[idx],
            hand_joints_r=self.hand_joints_r[idx],
            hand_contacts_r=self.hand_contacts_r[idx],
            class_vecs=self.class_vecs[idx],
            object_pcs_r=self.object_pcs_r[idx],
        )
        return return_dict
