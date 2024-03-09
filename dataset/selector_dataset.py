import glob
import logging
import os

import numpy as np
import yaml
import torch 
import pickle
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Pointclouds

from .base_dataset import SelectorDataset
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer
from utils import batch_normalize_mesh, get_NN

logger = logging.getLogger(__name__)


class DexYCBDataset(SelectorDataset):
    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        manolayer = ManoLayer()
        self.mano_f = torch.from_numpy(
            pickle.load(open(opt.mano_model, "rb"), encoding="latin1")[
                "f"
            ].astype(np.int64),
        )

        self.fidxs = []
        self.hand_thetas = []
        self.hand_verts_n = []
        self.class_vecs = []
        self.object_pcs_n = []
        object_paths = sorted(glob.glob(os.path.join(self.opt.path, "models", "*")))
        subjects = sorted(glob.glob(os.path.join(self.opt.path, "2020*-subject-*")))

        # save object names
        object_names = [os.path.basename(p) for p in object_paths]
        object_names_path = os.path.join(self.cfg.output_dir, "selector_objects.yaml")
        with open(object_names_path, "w") as f:
            yaml.safe_dump(object_names, f)
            logger.info(f"Saved {object_names_path}")

        for subject in subjects:
            mano_calibs = glob.glob(os.path.join(self.opt.path, f"calibration", "mano*subject-{subject[-2:]}*"))
            assert len(mano_calibs) == 1
            with open(os.path.join(mano_calibs[0], "mano.yml"), "r") as f:
                hand_beta = yaml.safe_load(f)["betas"]
            for sequence in sorted(glob.glob(os.path.join(subject, "2020*"))):
                with open(os.path.join(sequence, "meta.yaml"), "r") as f:
                    meta = yaml.safe_load(f)

                # hand_theta
                pose_mano = np.load(os.path.join(sequence, "pose.npz"))
                hand_theta_original = pose_mano["pose_m"][-1, 0]  # last frame only
                if meta["mano_sides"] == 'left':
                    hand_theta_right = hand_theta_original.copy()[:48]
                    for i in range(0, len(hand_theta_original), 3):
                        hand_theta_right[i] = -hand_theta_right[i]
                else:
                    hand_theta_right = hand_theta_original[:48]
                self.hand_thetas.append(hand_theta_right)

                # hand_verts, hand_normals
                hand_mesh_path = os.path.join(sequence, "hand_mesh_last.obj")
                hand_rightify_path = os.path.join(sequence, "hand_rightify_last.obj")
                if os.path.exists(hand_mesh_path):
                    if meta["mano_sides"] == 'left':
                        hand_verts, _, _ = load_obj(hand_rightify_path, load_textures=False)
                        logger.info(f"Loaded {hand_rightify_path}")
                    else:
                        hand_verts, _, _ = load_obj(hand_mesh_path, load_textures=False)
                        logger.info(f"Loaded {hand_mesh_path}")
                else:
                    hand_verts = manolayer(th_pose_coeffs=hand_theta_original[:48], th_betas=hand_beta, th_trans=hand_theta_original[48:])
                    save_obj(hand_mesh_path, hand_verts, self.mano_f)
                    logger.info(f"Saved {hand_mesh_path}")
                    if meta["mano_sides"] == 'left':
                        hand_verts[..., 0] = -hand_verts[..., 0]
                    save_obj(hand_rightify_path, hand_verts, self.mano_f)
                    logger.info(f"Saved {hand_rightify_path}")
                hand_verts_n, hand_center, hand_scale = batch_normalize_mesh(hand_verts)
                self.hand_verts_n.append(hand_verts_n)

                # class_vec
                object_idx = meta["ycb_ids"][meta["ycb_grasp_ind"]]
                object_idx -= 1  # 1-index -> 0-index
                class_vec = [0] * self.n_class
                class_vec[object_idx] = 1
                self.class_vecs.append(class_vec)

                # object_pcs
                object_path = object_paths[object_idx]
                object_pth = os.path.join(object_path, "textured_simple.obj")
                verts, _, _ = load_obj(object_pth, load_textures=False)
                if meta["mano_sides"] == 'left':
                    verts[..., 0] = -verts[..., 0]
                verts_n = (verts - hand_center) / hand_scale
                object_pc_n = Pointclouds(points=[verts_n])
                object_pc_n.normals_padded()
                self.object_pcs_n.append(object_pc_n)

                # fidxs
                seq = sequence.replace(self.opt.path, "")
                self.fidxs.append(seq)
        
        # hand_contacts
        self.hand_contacts_n = []
        for i in range(len(self.fidxs)):
            nn, _ = get_NN(self.hand_verts_n[i], self.object_pcs_n[i])
            nn = 100.0 * torch.sqrt(nn)
            hand_contact_n = 1.0 - 2 * (torch.sigmoid(nn*2) -0.5)
            self.hand_contacts_n.append(hand_contact_n)

        return

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]
        return_dict = dict(
            fidxs=fidx,
            hand_theta=self.hand_thetas[idx],
            hand_verts_n=self.hand_verts_n[idx],
            hand_contacts_n=self.hand_contacts_n[idx],
            class_vecs=self.class_vecs[idx],
            object_pcs_n=self.object_pcs_n[idx],
        )
        return return_dict
