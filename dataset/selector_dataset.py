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
import open3d as o3d

from .base_dataset import SelectorDataset
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer
from utils import batch_normalize_mesh, get_NN

logger = logging.getLogger(__name__)


class DexYCBDataset(SelectorDataset):
    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        self.cfg = cfg
        manolayer = ManoLayer(flat_hand_mean=False, ncomps=45, side='right', use_pca=True)
        self.mano_f = torch.from_numpy(
            pickle.load(open(opt.mano_model, "rb"), encoding="latin1")[
                "f"
            ].astype(np.int64),
        )

        self.fidxs = []
        self.hand_thetas = []
        self.hand_verts_n = []
        self.hand_verts_r = []
        self.class_vecs = []
        self.object_pcs_n = []
        self.hand_contacts_n = []
        object_paths = sorted(glob.glob(os.path.join(self.opt.path, "models", "*")))
        subjects = sorted(glob.glob(os.path.join(self.opt.path, "2020*-subject-*")))

        # save object names
        object_names = [os.path.basename(p) for p in object_paths]
        object_names_path = os.path.join(cfg.output_dir, "selector_objects.yaml")
        with open(object_names_path, "w") as f:
            yaml.safe_dump(object_names, f)
            logger.info(f"Saved {object_names_path}")

        for subject in subjects:
            mano_calibs = glob.glob(os.path.join(self.opt.path, "calibration", f"mano*subject-{subject[-2:]}*"))
            assert len(mano_calibs) == 1
            with open(os.path.join(mano_calibs[0], "mano.yml"), "r") as f:
                hand_beta = torch.tensor(yaml.safe_load(f)["betas"])

            for sequence in sorted(glob.glob(os.path.join(subject, "2020*"))):
                with open(os.path.join(sequence, "meta.yml"), "r") as f:
                    meta = yaml.safe_load(f)
                if meta["mano_sides"][0] == 'left':
                    logger.info(f"Skip {sequence} because it is left hand")
                    continue 

                # hand_theta
                serial = meta["serials"][0]
                frame = meta["num_frames"] - 1
                pose_info = np.load(os.path.join(sequence, serial, f"labels_{frame:06d}.npz"))
                hand_theta_original = torch.tensor(pose_info["pose_m"][0])
                self.hand_thetas.append(hand_theta_original[:48])

                # hand_verts
                hand_mesh_path = os.path.join(sequence, "hand_mesh_last.obj")
                hand_mesh_r_path = os.path.join(sequence, "hand_mesh_r_last.obj")
                if os.path.exists(hand_mesh_path) and os.path.exists(hand_mesh_r_path):
                    hand_verts_n, _, _ = load_obj(hand_mesh_path, load_textures=False)
                    hand_verts_r, _, _ = load_obj(hand_mesh_r_path, load_textures=False)
                    logger.info(f"Loaded {hand_mesh_path}")
                    logger.info(f"Loaded {hand_mesh_r_path}")
                else:
                    hand_verts, _ = manolayer(th_pose_coeffs=hand_theta_original[None, :48], th_betas=hand_beta[None, :], th_trans=hand_theta_original[None, 48:])
                    hand_theta_r = hand_theta_original.clone()
                    hand_theta_r[:3] = 0.0
                    hand_verts_r, _ = manolayer(th_pose_coeffs=hand_theta_r[None, :48], th_betas=hand_beta[None, :])
                    hand_verts = hand_verts[0]
                    hand_verts_r = hand_verts_r[0]
                    # normalize
                    hand_verts_n, hand_center, hand_scale = batch_normalize_mesh(hand_verts[None, ...])
                    hand_verts_n = hand_verts_n[0]
                    hand_center = hand_center[0]
                    hand_scale = hand_scale[0]
                    # save
                    save_obj(hand_mesh_path, hand_verts_n, self.mano_f)
                    save_obj(hand_mesh_r_path, hand_verts_r, self.mano_f)
                    logger.info(f"Saved {hand_mesh_path}")
                    logger.info(f"Saved {hand_mesh_r_path}")
                self.hand_verts_n.append(hand_verts_n)
                hand_verts_r = Pointclouds(points=[hand_verts_r])
                hand_verts_r.estimate_normals(assign_to_self=True)
                self.hand_verts_r.append(hand_verts_r)

                # class_vec
                object_idx = meta["ycb_ids"][meta["ycb_grasp_ind"]]
                object_idx -= 1  # 1-index -> 0-index
                class_vec = [0] * cfg.select_object.opt.n_class
                class_vec[object_idx] = 1
                class_vec = torch.tensor(class_vec)
                self.class_vecs.append(class_vec)

                # object_pcs
                object_cache_path = os.path.join(sequence, "object_cache.ply")
                io =IO()
                if os.path.exists(object_cache_path):
                    object_pc_n = io.load_pointcloud(object_cache_path)
                    logger.info(f"Loaded {object_cache_path}")
                else:
                    # load
                    object_path = object_paths[object_idx]
                    object_pth = os.path.join(object_path, "textured_simple.obj")
                    verts, _, _ = load_obj(object_pth, load_textures=False)
                    # object 6dof
                    object_6dof = torch.tensor(pose_info["pose_y"])[meta["ycb_grasp_ind"]]
                    object_6dof = torch.vstack([object_6dof, torch.tensor([0, 0, 0, 1], dtype=object_6dof.dtype).unsqueeze(0)])
                    verts = (object_6dof[:3, :3] @ verts.t() + object_6dof[:3, 3:4]).t()
                    # normalize
                    verts_n = (verts - hand_center) / hand_scale
                    verts_n = sample_farthest_points(verts_n[None, ...], torch.tensor([verts_n.shape[0]]), self.cfg.select_object.opt.object.points)[0][0]
                    # save
                    object_pc_n = Pointclouds(points=[verts_n])
                    object_pc_n.estimate_normals(assign_to_self=True)
                    io.save_pointcloud(object_pc_n, object_cache_path)
                    logger.info(f"Saved {object_cache_path}")
                self.object_pcs_n.append(object_pc_n)

                # hand_contacts
                contact_vis_path = os.path.join(sequence, "hand_contact.ply")
                contact_cache_path = os.path.join(sequence, "hand_contact_cache.pt")
                if os.path.exists(contact_cache_path):
                    hand_contact_n = torch.load(contact_cache_path)
                    logger.info(f"Loaded {contact_cache_path}")
                else:
                    nn, _ = get_NN(hand_verts_n.unsqueeze(0), object_pc_n.points_padded())
                    nn = 40.0 * torch.sqrt(nn)
                    hand_contact_n = 1.0 - 2 * (torch.sigmoid(nn) -0.5)
                    hand_contact_n = hand_contact_n.squeeze(0)
                    torch.save(hand_contact_n, contact_cache_path)
                    logger.info(f"Saved {contact_cache_path}")
                    # visualize
                    hand_contact_vis = hand_verts_n[hand_contact_n > 0.5]
                    hand_contact_vis = o3d.utility.Vector3dVector(hand_contact_vis.numpy())
                    hand_contact_vis = o3d.geometry.PointCloud(hand_contact_vis)
                    hand_contact_vis.paint_uniform_color([1, 0, 0])
                    o3d.io.write_point_cloud(contact_vis_path, hand_contact_vis)
                    logger.info(f"Saved {contact_vis_path}")
                self.hand_contacts_n.append(hand_contact_n)

                # fidxs
                seq = sequence.replace(self.opt.path, "")
                self.fidxs.append(seq)

        return

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]
        return_dict = dict(
            fidxs=fidx,
            hand_theta=self.hand_thetas[idx],
            hand_verts_n=self.hand_verts_n[idx],
            hand_verts_r=self.hand_verts_r[idx],
            hand_contacts_n=self.hand_contacts_n[idx],
            class_vecs=self.class_vecs[idx],
            object_pcs_n=self.object_pcs_n[idx],
        )
        return return_dict