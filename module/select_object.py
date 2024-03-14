import logging
import os
import pickle

import numpy as np
import torch
import yaml
import open3d as o3d
from matplotlib import pyplot as plt
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SelectorData
from utils import get_NN
from models.selectnet import SelectObjectNet
from visualization import bones, bone_colors
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer

logger = logging.getLogger(__name__)


class SelectObject(LightningModule):
    def __init__(self, opt, cfg):
        super().__init__()
        self.opt = opt
        self.cfg = cfg

        self.model = SelectObjectNet(opt)
        self.n_class = self.opt.n_class
        self.manolayer = ManoLayer(flat_hand_mean=False, ncomps=45, side='right', use_pca=True)
        return

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            selector_dataset = instantiate(
                self.cfg.selector_dataset, self.cfg, _recursive_=False
            )
            val_size = int(len(selector_dataset) * self.opt.val.ratio)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                selector_dataset, [len(selector_dataset) - val_size, val_size]
            )
            self.mano_f = selector_dataset.mano_f
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opt.train.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=SelectorData.collate_fn,
        )
        del self.train_dataset
        return dataloader

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            selector_dataset = instantiate(
                self.cfg.selector_dataset, cfg=self.cfg, _recursive_=False
            )
            val_size = int(len(selector_dataset) * self.opt.val.ratio)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                selector_dataset, [len(selector_dataset) - val_size, val_size]
            )
            self.mano_f = selector_dataset.mano_f
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.opt.train.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=SelectorData.collate_fn,
        )
        del self.val_dataset
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "interval": "step",
            "frequency": self.opt.val.period,
            "monitor": "val_loss",
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_fit_start(self):
        self.kl_annealing = max(
            1. / self.opt.loss.kl_annealing_epoch, 1.0
        )
        return
    
    def on_predict_start(self):
        object_names_path = os.path.join(self.cfg.output_dir, "selector_objects.yaml")
        with open(object_names_path, "r") as f:
            self.object_names = yaml.safe_load(f)
            logger.info(f"Loaded {object_names_path}")
        self.mano_f = torch.from_numpy(
                pickle.load(open(self.cfg.hand_dataset.opt.hand.mano_model, "rb"), encoding="latin1")[
                "f"
            ].astype(np.int64),
        )
        return

    def on_train_epoch_start(self):
        self.kl_annealing = max(
            (self.current_epoch + 1) / self.opt.loss.kl_annealing_epoch, 1.0
        )
        return

    def forward(self, batch):
        data = SelectorData(**batch)
        fidxs = data.fidxs
        hand_theta = data.hand_theta
        hand_verts_n = data.hand_verts_n
        hand_verts_r = data.hand_verts_r
        hand_contacts_n = data.hand_contacts_n
        class_vecs = data.class_vecs
        object_pcs_n = data.object_pcs_n

        # encode - decode
        (
            contact_pred,
            contact_xyz_n,
            object_base_n,
            hand_theta_r,
        ) = self.model.contact_net(hand_theta, hand_verts_n, hand_verts_r)
        z = self.model.enc(
            class_vecs,
            hand_theta,
            contact_xyz_n,
            object_pcs_n,
            object_pcs_n.normals_padded(),
        )
        z_s = z.rsample()
        class_pred, object_pred_n = self.model.dec(
            z_s, object_base_n, hand_theta
        )
        object_pred_n = object_pred_n.permute(0, 2, 1).contiguous()

        # loss
        loss, loss_dict = self.loss(
            class_pred,
            class_vecs,
            object_pred_n,
            object_pcs_n.points_padded(),
            contact_pred,
            hand_contacts_n,
            contact_xyz_n,
            z,
        )

        # logging & visualization
        for b in range(len(fidxs)):
            loss_dict[f"class_pred_{fidxs[b]}"] = torch.argmax(class_pred[b]).item()
            loss_dict[f"class_gt_{fidxs[b]}"] = torch.argmax(class_vecs[b]).item()
        outputs = dict(
            object_pred_n=object_pred_n,
            contact_xyz_n=contact_xyz_n,
        )
        self.batch = batch
        outputs["hand_theta_r"] = hand_theta_r
        outputs["object_base_n"] = object_base_n[:, :, :3]

        return loss, loss_dict, outputs

    def training_step(self, batch):
        loss, loss_dict, outputs = self(batch)
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        outputs["loss"] = loss
        return outputs

    def validation_step(self, batch):
        loss, loss_dict, outputs = self(batch)
        val_loss_dict = {}
        for k, v in loss_dict.items():
            val_loss_dict[f"val_{k}"] = v
        self.log_dict(
            val_loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        outputs["val_loss"] = loss
        return outputs

    def predict_step(self, batch):
        hand_fidxs = batch["hand_fidxs"]
        hand_theta = batch["hand_theta"]
        hand_verts_n = batch["hand_verts_n"]
        hand_verts_r = batch["hand_verts_r"]
        test_batch_size = hand_theta.shape[0]

        # decode
        (
            contact_pred,
            contact_xyz_n,
            object_base_n,
            hand_theta_r,
        ) = self.model.contact_net(hand_theta, hand_verts_n, hand_verts_r)
        z_gen = np.random.normal(0.0, 1.0, size=(test_batch_size, self.opt.dim_latent))
        z_gen = torch.tensor(z_gen, dtype=hand_theta.dtype).to(hand_theta.device)
        class_pred, object_pred_n = self.model.dec(
            z_gen, object_base_n, hand_theta
        )
        object_pred_n = object_pred_n.permute(0, 2, 1).contiguous()

        # logging & visualization
        consistency_loss, _ = self.loss(
            class_pred=class_pred,
            class_gt=None,
            verts_pred=object_pred_n,
            verts_gt=None,
            contact_pred=contact_pred,
            contact_gt=None,
            contact_xyz=contact_xyz_n,
            z=None,
            unsupervised_only=True,
        )
        logger.warning(
            f"Consistency loss for hand {hand_fidxs}: {consistency_loss:.4f}"
        )
        self.visualize(object_pred_n, contact_xyz_n, hand_fidxs)
        self.visualize_debug(
            hand_theta,
            hand_theta_r,
            hand_verts_n,
            hand_verts_r,
            hand_fidxs,
            object_base_n,
        )

        # output
        class_pred = torch.argmax(class_pred, dim=1)
        class_pred = self.object_names[class_pred]

        return class_pred

    def loss(
        self,
        class_pred,
        class_gt,
        verts_pred,
        verts_gt,
        contact_pred,
        contact_gt,
        contact_xyz,
        z=None,
        unsupervised_only=False,
    ):
        batch_size = class_pred.shape[0]

        loss = 0.0
        loss_dict = {}
        thres = torch.tensor(0.5, dtype=contact_xyz.dtype, device=contact_xyz.device)
        if not unsupervised_only:
            # category
            category_loss = -torch.mean(class_gt * torch.log(class_pred + 1e-8) + (1-class_gt) * torch.log(1-class_pred + 1e-8))
            # category_loss = torch.mean(F.binary_cross_entropy(class_gt, class_pred))
            category_acc = torch.sum(torch.argmax(class_gt, dim=1) == torch.argmax(class_pred, dim=1)) / batch_size

            # object pc
            objectpoint_loss = F.mse_loss(verts_pred.view(batch_size, -1), verts_gt.view(batch_size, -1))

            # contact
            # weight = torch.max(contact_gt, torch.tensor(0.5, dtype=contact_gt.dtype, device=contact_gt.device))
            # contact_loss = F.binary_cross_entropy(
            #     contact_pred, contact_gt, reduction="none"
            # )
            # contact_loss = torch.mean(contact_loss)
            contact_loss = F.mse_loss(contact_pred, contact_gt)
            contact_acc = torch.sum((contact_pred>thres) == (contact_gt>thres)) / contact_gt.numel()

            # KL-divergence
            if z is not None:
                dim_latent = z.mean.shape[1]
                q_z = torch.distributions.normal.Normal(z.mean, z.scale)
                p_z = torch.distributions.normal.Normal(
                    loc=torch.tensor(
                        np.zeros([batch_size, dim_latent]), requires_grad=False
                    )
                    .to(z.mean.device)
                    .type(z.mean.dtype),
                    scale=torch.tensor(
                        np.ones([batch_size, dim_latent]), requires_grad=False
                    )
                    .to(z.mean.device)
                    .type(z.mean.dtype),
                )
                kl_loss = torch.mean(
                    torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])
                )

            loss = (
                self.opt.loss.category_weight * category_loss
                + self.opt.loss.objectpoint_weight * objectpoint_loss
                + self.opt.loss.contact_weight * contact_loss
                + self.kl_annealing * self.opt.loss.kl_weight * kl_loss
            )
            loss_dict["category_loss"] = category_loss
            loss_dict["category_acc"] = category_acc
            loss_dict["objectpoint_loss"] = objectpoint_loss
            loss_dict["contact_loss"] = contact_loss
            loss_dict["contact_acc"] = contact_acc
            loss_dict["kl_loss"] = kl_loss

        # unsupervised loss
        nn, _ = get_NN(contact_xyz, verts_pred)
        nn = 100.0 * torch.sqrt(nn)
        hand_contact = 1.0 - 2 * (torch.sigmoid(nn * 2) - 0.5)
        consistency_loss = torch.sum(hand_contact < thres) / hand_contact.numel()
        consistency_acc = 1.0 - consistency_loss
        loss = loss + self.opt.loss.consistency_weight * consistency_loss
        loss_dict["consistency_acc"] = consistency_acc

        loss_dict["loss"] = loss
        return loss, loss_dict

    def visualize(self, object_pred_n, contact_xyz_n, fidxs, mode=None):
        batch_size = object_pred_n.shape[0]
        save_dir = os.path.join(self.cfg.output_dir, "selector_training")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if mode == "val":
            sign = "val_"
        else:
            sign = ""

        # object pred
        object_pred_n = object_pred_n.detach().cpu().numpy()
        for b in range(batch_size):
            pc = object_pred_n[b]
            pc = o3d.utility.Vector3dVector(pc)
            pc = o3d.geometry.PointCloud(pc)
            if self.trainer.state.fn == "fit":
                assert self.trainer.state.stage not in ["predict", "test"]
                visualize_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_object_pred.ply",
                )
            elif self.trainer.state.fn == "predict":
                visualize_path = os.path.join(
                    self.cfg.results_dir, f"hand{fidxs[b]}_selector_object_pred.ply"
                )
            else:
                logger.error(f"Invalid state for visualizing: {self.trainer.state.fn}")
                raise ValueError
            o3d.io.write_point_cloud(visualize_path, pc)
            logger.info(f"Saved {visualize_path}")
        
        # contact xyz pred
        contact_xyz_n = contact_xyz_n.detach().cpu().numpy()
        for b in range(batch_size):
            pc = contact_xyz_n[b]
            pc = o3d.utility.Vector3dVector(pc)
            pc = o3d.geometry.PointCloud(pc)
            pc.paint_uniform_color([1, 0, 0])  
            if self.trainer.state.fn == "fit":
                assert self.trainer.state.stage not in ["predict", "test"]
                visualize_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_contact_pred.ply",
                )
            elif self.trainer.state.fn == "predict":
                visualize_path = os.path.join(
                    self.cfg.results_dir, f"hand{fidxs[b]}_selector_contact.ply"
                )
            else:
                logger.error(f"Invalid state for visualizing: {self.trainer.state.fn}")
                raise ValueError
            o3d.io.write_point_cloud(visualize_path, pc)
            logger.info(f"Saved {visualize_path}")

        return

    def visualize_debug(
        self,
        hand_theta,
        hand_theta_r,
        hand_verts_n,
        hand_verts_r,
        fidxs,
        object_base_n,
        mode=None,
    ):
        batch_size = hand_theta.shape[0]
        save_dir = os.path.join(self.cfg.output_dir, "selector_training")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if mode == "val":
            sign = "val_"
        else:
            sign = ""

        # hand theta
        _, joints = self.manolayer(hand_theta)
        _, joints_r = self.manolayer(hand_theta_r)
        joints = joints.cpu()
        joints_r = joints_r.cpu()
        for b in range(batch_size):
            points = o3d.utility.Vector3dVector(joints[b])
            lines = o3d.utility.Vector2iVector(bones)
            colors = np.array(bone_colors * 255)
            line_set = o3d.geometry.LineSet(points, lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            save_path = os.path.join(save_dir, f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_hand_theta.ply")
            o3d.io.write_line_set(save_path, line_set)
            logger.info(f"Saved {save_path}")
        for b in range(batch_size):
            points = o3d.utility.Vector3dVector(joints_r[b])
            lines = o3d.utility.Vector2iVector(bones)
            colors = np.array(bone_colors)
            line_set = o3d.geometry.LineSet(points, lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            save_path = os.path.join(save_dir, f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_hand_theta_r.ply")
            o3d.io.write_line_set(save_path, line_set)
            logger.info(f"Saved {save_path}")

        # hand verts (ordered)
        color = plt.cm.viridis(np.linspace(0, 1, hand_verts_n.shape[1]))[
            :, :-1
        ]  # no alpha channel
        hand_verts_n = hand_verts_n.cpu()
        for b in range(batch_size):
            vertices = o3d.utility.Vector3dVector(hand_verts_n[b])
            faces = o3d.utility.Vector3iVector(self.mano_f)
            mesh = o3d.geometry.TriangleMesh(vertices, faces)
            mesh.vertex_colors = o3d.utility.Vector3dVector(color)
            save_path = os.path.join(save_dir, f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_hand_verts_n.ply")
            o3d.io.write_triangle_mesh(save_path, mesh)
            logger.info(f"Saved {save_path}")

        # hand verts nr with normals
        hand_verts_r = hand_verts_r.cpu()
        for b in range(batch_size):
            vertices = o3d.utility.Vector3dVector(hand_verts_r.points_padded()[b])
            faces = o3d.utility.Vector3iVector(self.mano_f)
            mesh = o3d.geometry.TriangleMesh(vertices, faces)
            color = (hand_verts_r.normals_padded()[b] + 1.0) / 2.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(color.detach().cpu().numpy())
            save_path = os.path.join(save_dir, f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_hand_verts_r.ply")
            o3d.io.write_triangle_mesh(save_path, mesh)
            logger.info(f"Saved {save_path}")

        # object base
        object_base_n = object_base_n.cpu()
        for b in range(batch_size):
            pc = o3d.utility.Vector3dVector(object_base_n[b])
            pc = o3d.geometry.PointCloud(pc)
            save_path = os.path.join(save_dir, f"{sign}{self.current_epoch}_{self.global_step:05d}_subject{fidxs[b][17:19]}_{fidxs[b][-6:]}_object_base_n.ply")
            o3d.io.write_point_cloud(save_path, pc)
            logger.info(f"Saved {save_path}")

        return

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch % self.opt.val.vis_epoch == 0:
            self.visualize(
                outputs["object_pred_n"], outputs["contact_xyz_n"], batch["fidxs"], mode="val"
            )
        if self.current_epoch % self.opt.val.debug_vis_epoch == 0:
            self.visualize_debug(
                batch["hand_theta"],
                outputs["hand_theta_r"],
                batch["hand_verts_n"],
                batch["hand_verts_r"],
                batch["fidxs"],
                outputs["object_base_n"],
                mode="val",
            )
        return

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # random pick
        if len(batch["fidxs"]) < self.opt.train.n_vis_samples:
            samples = torch.arange(len(batch["fidxs"]))
        else:
            samples = torch.randperm(len(batch["fidxs"]))[: self.opt.train.n_vis_samples]

        if self.current_epoch % self.opt.train.vis_epoch == 0:
            self.visualize(
                outputs["object_pred_n"][samples], outputs["contact_xyz_n"][samples], batch["fidxs"][samples], mode="train"
            )
        if self.current_epoch % self.opt.train.debug_vis_epoch == 0:
            self.visualize_debug(
                batch["hand_theta"][samples],
                outputs["hand_theta_r"][samples],
                batch["hand_verts_n"][samples],
                batch["hand_verts_r"][samples],
                batch["fidxs"][samples],
                outputs["object_base_n"][samples],
                mode="train",
            )
        return
