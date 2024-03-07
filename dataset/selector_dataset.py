import glob
import logging
import os

import numpy as np
import yaml
import torch 
import pickle
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Pointclouds, Meshes

from .base_dataset import SelectorDataset
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer

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
        self.hand_verts = []
        self.hand_normals = []
        self.class_vecs = []
        self.object_pcs = []
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
                # hand_theta
                pose_mano = np.load(os.path.join(sequence, "pose.npz"))
                hand_theta = pose_mano["pose_m"][
                    -1, 0, :48
                ]  # last frame only, no translation
                self.hand_thetas.append(hand_theta)

                # hand_verts, hand_normals
                hand_mesh_path = os.path.join(sequence, "hand_mesh_last.obj")
                if os.path.exists(hand_mesh_path):
                    hand_verts, faces, aux = load_obj(hand_mesh_path, load_textures=False)
                    hand_normals = aux.normals
                    logger.info(f"Loaded {hand_mesh_path}")
                else:
                    hand_trans = pose_mano["pose_m"][-1, 0, 48:51]
                    hand_verts = manolayer(th_pose_coeffs=hand_theta, th_betas=hand_beta, th_trans=hand_trans)
                    hand_meshes = Meshes(verts=[hand_verts], faces=[self.mano_f])
                    hand_normals = hand_meshes.verts_normals_padded()
                    save_obj(hand_mesh_path, hand_verts, self.mano_f, normals=hand_normals)
                    logger.info(f"Saved {hand_mesh_path}")
                self.hand_verts.append(hand_verts)
                self.hand_normals.append(hand_normals)

                # class_vec
                with open(os.path.join(sequence, "meta.yaml"), "r") as f:
                    meta = yaml.safe_load(f)
                object_idx = meta["ycb_ids"][meta["ycb_grasp_ind"]]
                object_idx -= 1  # 1-index -> 0-index
                class_vec = [0] * self.n_class
                class_vec[object_idx] = 1
                self.class_vecs.append(class_vec)

                # object_pcs
                object_path = object_paths[object_idx]
                object_pth = os.path.join(object_path, "textured_simple.obj")
                verts, faces, aux = load_obj(object_pth, load_textures=False)
                assert verts.shape[0] == aux.normals.shape[0], "DexYCB assumption"
                object_pc = Pointclouds(points=[verts], normals=[aux.normals])
                self.object_pcs.append(object_pc)

                # fidxs
                seq = sequence.replace(self.opt.path, "")
                self.fidxs.append(seq)
        
        # hand_contacts
        self.hand_contacts = []
        for i in range(len(self.fidxs)):
            nn, _ = self.get_NN(self.hand_verts[i], self.object_pcs[i])
            nn = 100.0 * torch.sqrt(nn)
            hand_contact = 1.0 - 2 * (torch.sigmoid(nn*2) -0.5)
            self.hand_contacts.append(hand_contact)

        return

    def __getitem__(self, idx):
        fidx = self.fidxs[idx]
        return_dict = dict(
            fidxs=fidx,
            hand_theta=self.hand_thetas[idx],
            hand_verts=self.hand_verts[idx],
            hand_normals=self.hand_normals[idx],
            hand_contacts=self.hand_contacts[idx],
            class_vecs=self.class_vecs[idx],
            object_pcs=self.object_pcs[idx],
        )
        return return_dict
