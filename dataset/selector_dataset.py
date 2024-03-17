import glob
import logging
import os

import numpy as np
import yaml
import torch 
import pickle
from pytorch3d.io import load_obj, save_obj, IO
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import open3d as o3d
from matplotlib import pyplot as plt

from .base_dataset import SelectorDataset
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer
from utils import batch_normalize_mesh, get_NN

logger = logging.getLogger(__name__)


class DexYCBDataset(SelectorDataset):
    categories = [
        set([15]),
        set([0, 3, 10, 1, 11]),
        set([5, 8, 13, 2, 4, 6, 20]),
        set([7]),
        set([9]),
        set([16, 17, 18, 19]),
        # set([12]), 
        # set([14]),
    ]

    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        self.cfg = cfg
        manolayer = ManoLayer(flat_hand_mean=False, ncomps=45, side='right', use_pca=True)
        self.mano_f = torch.from_numpy(
            pickle.load(open(opt.mano_model, "rb"), encoding="latin1")[
                "f"
            ].astype(np.int64),
        )
        ordered_color = plt.cm.gist_rainbow(np.linspace(0, 1, self.cfg.select_object.opt.n_obj_points))[
            :, :-1
        ]  # no alpha channel
        self.class2fidxs = {}

        self.fidxs = []
        self.hand_thetas = []
        # self.hand_verts_n = []
        self.hand_verts_r = []
        self.class_vecs = []
        self.object_pcs_r = []
        self.hand_contacts_r = []
        object_paths = sorted(glob.glob(os.path.join(self.opt.path, "models", "*")))
        subjects = sorted(glob.glob(os.path.join(self.opt.path, "2020*-subject-*/")))

        # save object names
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

        for subject in subjects:
            mano_calibs = glob.glob(os.path.join(self.opt.path, "calibration", f"mano*subject-{subject[-3:-1]}*"))
            assert len(mano_calibs) == 1
            with open(os.path.join(mano_calibs[0], "mano.yml"), "r") as f:
                hand_beta = torch.tensor(yaml.safe_load(f)["betas"])

            for sequence in sorted(glob.glob(os.path.join(subject, "2020*"))):
                with open(os.path.join(sequence, "meta.yml"), "r") as f:
                    meta = yaml.safe_load(f)
                if meta["mano_sides"][0] == 'left':
                    logger.info(f"Skip {sequence} because it is left hand")
                    continue 

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

                # hand_theta
                serial = meta["serials"][0]
                frame = meta["num_frames"] - 1
                pose_info = np.load(os.path.join(sequence, serial, f"labels_{frame:06d}.npz"))
                hand_theta_original = torch.tensor(pose_info["pose_m"][0])
                self.hand_thetas.append(hand_theta_original[:48])

                # hand_verts
                # hand_mesh_path = os.path.join(sequence, "hand_mesh_last.obj")
                hand_mesh_r_path = os.path.join(sequence, "hand_mesh_r_last.obj")
                if os.path.exists(hand_mesh_r_path) and not opt.refresh_data:
                    # hand_verts_n, _, _ = load_obj(hand_mesh_path, load_textures=False)
                    hand_verts_r, _, _ = load_obj(hand_mesh_r_path, load_textures=False)
                    # logger.info(f"Loaded {hand_mesh_path}")
                    logger.info(f"Loaded {hand_mesh_r_path}")
                else:
                    # hand_verts, hand_joints = manolayer(th_pose_coeffs=hand_theta_original[None, :48], th_betas=hand_beta[None, :], th_trans=hand_theta_original[None, 48:])
                    hand_theta_r = hand_theta_original.clone()
                    hand_theta_r[:3] = 0.0
                    hand_verts_r, hand_joints_r = manolayer(th_pose_coeffs=hand_theta_r[None, :48], th_betas=hand_beta[None, :])
                    # hand_verts = hand_verts[0]
                    hand_verts_r = hand_verts_r[0]
                    # normalize
                    # hand_verts_n, hand_center, hand_scale = batch_normalize_mesh(hand_verts[None, ...])
                    # hand_verts_n = hand_verts_n[0]
                    # hand_center = hand_center[0]
                    # hand_scale = hand_scale[0]
                    # save
                    # save_obj(hand_mesh_path, hand_verts_n, self.mano_f)
                    save_obj(hand_mesh_r_path, hand_verts_r, self.mano_f)
                    # logger.info(f"Saved {hand_mesh_path}")
                    logger.info(f"Saved {hand_mesh_r_path}")
                # self.hand_verts_n.append(hand_verts_n)
                hand_verts_r = Pointclouds(points=[hand_verts_r])
                hand_verts_r.estimate_normals(assign_to_self=True)
                self.hand_verts_r.append(hand_verts_r)

                # object_pcs
                # object_cache_path = os.path.join(sequence, "object_cache.ply")
                object_r_path = os.path.join(sequence, "object_r.ply")
                io =IO()
                if os.path.exists(object_r_path) and not opt.refresh_data:
                    # object_pc_n = io.load_pointcloud(object_cache_path)
                    object_pc_r = io.load_pointcloud(object_r_path)
                    # logger.info(f"Loaded {object_cache_path}")
                    logger.info(f"Loaded {object_r_path}")
                else:
                    # load
                    object_path = object_paths[object_idx]
                    object_pth = os.path.join(object_path, "textured_simple.obj")
                    verts, _, _ = load_obj(object_pth, load_textures=False)
                    # object 6dof
                    object_6dof = torch.tensor(pose_info["pose_y"])[meta["ycb_grasp_ind"]]
                    object_6dof = torch.vstack([object_6dof, torch.tensor([0, 0, 0, 1], dtype=object_6dof.dtype).unsqueeze(0)])
                    verts = (object_6dof[:3, :3] @ verts.t() + object_6dof[:3, 3:4]).t()
                    verts = sample_farthest_points(verts[None, ...], torch.tensor([verts.shape[0]]), self.cfg.select_object.opt.n_obj_points)[0][0]
                    # rotation normalize
                    verts_r = verts - (hand_theta_original[48:51]+hand_joints_r[0,0]).unsqueeze(0)
                    t = Rotate(axis_angle_to_matrix(hand_theta_original[:3]), dtype=verts.dtype)
                    verts_r = t.transform_points(verts_r.unsqueeze(0)).squeeze(0)
                    verts_r = verts_r + hand_joints_r[0,0].unsqueeze(0)
                    verts_r = torch.index_select(verts_r, 0, torch.sort(verts_r[:, 2])[1])
                    # normalize
                    # verts_n = (verts - hand_center) / hand_scale
                    # save
                    # object_pc_n = Pointclouds(points=[verts_n])
                    object_pc_r = Pointclouds(points=[verts_r])
                    # object_pc_n.estimate_normals(assign_to_self=True)
                    object_pc_r.estimate_normals(assign_to_self=True)
                    # io.save_pointcloud(object_pc_n, object_cache_path)
                    io.save_pointcloud(object_pc_r, object_r_path)
                    # logger.info(f"Saved {object_cache_path}")
                    logger.info(f"Saved {object_r_path}")
                    if self.cfg.debug:
                        object_ordered_path = os.path.join(sequence, "object_r_ordered.ply")
                        pc = o3d.utility.Vector3dVector(verts_r.numpy())
                        pc = o3d.geometry.PointCloud(pc)
                        pc.colors = o3d.utility.Vector3dVector(ordered_color)
                        o3d.io.write_point_cloud(object_ordered_path, pc)
                        logger.info(f"Saved {object_ordered_path}")
                self.object_pcs_r.append(object_pc_r)

                # hand_contacts
                contact_vis_path = os.path.join(sequence, "hand_contact.ply")
                contact_cache_path = os.path.join(sequence, "hand_contact_cache.pt")
                if os.path.exists(contact_cache_path) and not opt.refresh_data:
                    hand_contact_r = torch.load(contact_cache_path)
                    logger.info(f"Loaded {contact_cache_path}")
                else:
                    nn, _ = get_NN(hand_verts_r.points_padded(), object_pc_r.points_padded())
                    nn = 20.0 * torch.sqrt(nn*1000)
                    hand_contact_r = 1.0 - 2 * (torch.sigmoid(nn) -0.5)
                    hand_contact_r = hand_contact_r.squeeze(0)
                    torch.save(hand_contact_r, contact_cache_path)
                    logger.info(f"Saved {contact_cache_path}")
                    # visualize
                    hand_contact_vis = hand_verts_r.points_padded()[0][hand_contact_r > 0.5]
                    hand_contact_vis = o3d.utility.Vector3dVector(hand_contact_vis.numpy())
                    hand_contact_vis = o3d.geometry.PointCloud(hand_contact_vis)
                    hand_contact_vis.paint_uniform_color([1, 0, 0])
                    o3d.io.write_point_cloud(contact_vis_path, hand_contact_vis)
                    logger.info(f"Saved {contact_vis_path}")
                self.hand_contacts_r.append(hand_contact_r)

                # fidxs
                seq = sequence.replace(self.opt.path, "")
                self.fidxs.append(seq)
                if cate_idx not in self.class2fidxs:
                    self.class2fidxs[cate_idx] = [seq]
                else:
                    self.class2fidxs[cate_idx].append(seq)

        self.class_weights = torch.zeros(len(self.class2fidxs))
        for k in np.sort(list(self.class2fidxs.keys())):
            self.class_weights[k] = 1.0 / len(self.class2fidxs[k])
        return

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]
        return_dict = dict(
            fidxs=fidx,
            hand_theta=self.hand_thetas[idx],
            # hand_verts_n=self.hand_verts_n[idx],
            hand_verts_r=self.hand_verts_r[idx],
            hand_contacts_r=self.hand_contacts_r[idx],
            class_vecs=self.class_vecs[idx],
            object_pcs_r=self.object_pcs_r[idx],
        )
        return return_dict