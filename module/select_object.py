import logging
import os
import pickle
import random

import numpy as np
import open3d as o3d
import torch
import yaml
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from pytorch3d.ops import iterative_closest_point, knn_points
from pytorch3d.structures import Pointclouds
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SelectorData
from models.selectnet import SelectObjectNet
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer

logger = logging.getLogger(__name__)


def weighted_distance(points, query, weight):
    points_batch = points.unsqueeze(0).repeat(query.shape[0], 1, 1)
    query = query.unsqueeze(1)
    weight = weight.unsqueeze(1)

    points_weighted = points_batch * weight
    query_weighted = query * weight
    distances = knn_points(
        query_weighted, points_weighted, K=points_weighted.shape[1], return_sorted=False
    ).dists
    return distances.squeeze(1) + 1e-8


class SelectObject(LightningModule):
    def __init__(self, opt, cfg, device):
        super().__init__()
        self.opt = opt
        self.cfg = cfg
        self.manual_device = device

        self.model = SelectObjectNet(opt, device_manual=device)
        self.dim_input = self.opt.dim_input
        self.manolayer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side="right", use_pca=True
        )
        return

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            selector_dataset = instantiate(
                self.cfg.selector_dataset, self.cfg, _recursive_=False
            )
            self.class_weights = selector_dataset.class_weights.to(self.manual_device)
            self.cateidx_to_shape = selector_dataset.cateidx_to_shape.to(
                self.manual_device
            )
            self.mano_f = selector_dataset.mano_f

            # train-val split
            fidxs = selector_dataset.fidxs
            val_inds = []
            for cls in selector_dataset.class2fidxs.values():
                if len(cls) < self.opt.val.n_per_class:
                    logger.error(
                        f"Samples in class {cls} has fewer than {self.opt.val.n_per_class} samples"
                    )
                    raise ValueError
                val_names = np.random.choice(cls, self.opt.val.n_per_class)
                val_inds.extend([fidxs.index(name) for name in val_names])
            train_inds = list(set(range(len(selector_dataset))) - set(val_inds))
            val_inds = np.sort(val_inds)
            train_inds = np.sort(train_inds)
            self.train_dataset = torch.utils.data.Subset(selector_dataset, train_inds)
            self.val_dataset = torch.utils.data.Subset(selector_dataset, val_inds)

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
            self.class_weights = selector_dataset.class_weights.to(self.manual_device)
            self.cateidx_to_shape = selector_dataset.cateidx_to_shape.to(
                self.manual_device
            )
            self.mano_f = selector_dataset.mano_f

            # train-val split
            fidxs = selector_dataset.fidxs
            val_inds = []
            for cls in selector_dataset.class2fidxs.values():
                if len(cls) < self.opt.val.n_per_class:
                    logger.error(
                        f"Samples in class {cls} has fewer than {self.opt.val.n_per_class} samples"
                    )
                    raise ValueError
                val_names = np.random.choice(cls, self.opt.val.n_per_class)
                val_inds.extend([fidxs.index(name) for name in val_names])
            train_inds = list(set(range(len(selector_dataset))) - set(val_inds))
            val_inds = np.sort(val_inds)
            train_inds = np.sort(train_inds)
            self.train_dataset = torch.utils.data.Subset(selector_dataset, train_inds)
            self.val_dataset = torch.utils.data.Subset(selector_dataset, val_inds)

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
            self.model.parameters(),
            lr=self.opt.train.lr,
            weight_decay=self.opt.train.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_fit_start(self):
        self.kl_annealing = max(1.0 / self.opt.loss.kl_annealing_epoch, 1.0)
        return

    def on_predict_start(self):
        # object names
        object_names_path = os.path.join(
            self.cfg.selector_ckpt_dir, "selector_objects.yaml"
        )
        with open(object_names_path, "r") as f:
            self.object_names = yaml.safe_load(f)
            logger.info(f"Loaded {object_names_path}")

        # cate_to_shape
        cate_to_shape_path = os.path.join(
            self.cfg.selector_ckpt_dir, "selector_cate2shape.yaml"
        )
        with open(cate_to_shape_path, "r") as f:
            self.cate_to_shape = yaml.safe_load(f)
            logger.info(f"Loaded {cate_to_shape_path}")

        # mano face
        self.mano_f = torch.from_numpy(
            pickle.load(
                open(self.cfg.hand_dataset.opt.hand.mano_model, "rb"), encoding="latin1"
            )["f"].astype(np.int64),
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
        hand_verts_r = data.hand_verts_r
        hand_joints_r = data.hand_joints_r
        hand_contacts_r = data.hand_contacts_r
        class_vecs = data.class_vecs
        shape_codes = data.shape_codes
        object_pcs_r = data.object_pcs_r
        batch_size = class_vecs.shape[0]

        # selector type
        if self.opt.output == "category":
            inputs = class_vecs
        elif self.opt.output == "shapecode":
            inputs = shape_codes
        else:
            logger.error(f"Invalid selector output: {self.opt.output}")
            raise ValueError("Invalid selector output")

        # encode - decode
        joints_encoded = self.model.pos_enc(hand_joints_r.view(-1, 3)).view(
            batch_size, -1
        )
        (
            contact_pred,
            contact_xyz_r,
        ) = self.model.contact_net(joints_encoded, hand_verts_r)
        z = self.model.enc(
            inputs,
            joints_encoded,
            contact_xyz_r,
            object_pcs_r,
            object_pcs_r.normals_padded(),
        )
        z_s = z.rsample()
        output_pred, object_pred_r = self.model.dec(z_s, joints_encoded, contact_xyz_r)

        if self.opt.output == "category":
            shape_code_pred = None
            class_vecs_pred = output_pred
        elif self.opt.output == "shapecode":
            shape_code_pred = output_pred
            shapecode, confidence = (
                shape_code_pred[:, : self.opt.dim_input],
                shape_code_pred[:, self.opt.dim_input :],
            )
            distances = weighted_distance(
                self.cateidx_to_shape, shapecode.detach(), confidence.detach()
            )
            class_vecs_pred = torch.reciprocal(distances)
        class_pred = torch.argmax(class_vecs_pred, dim=1)
        class_gt = torch.argmax(class_vecs, dim=1)

        # loss
        loss, loss_dict = self.loss(
            class_vecs_pred,
            class_vecs,
            shape_code_pred,
            shape_codes,
            object_pred_r,
            object_pcs_r.points_padded(),
            contact_pred,
            hand_contacts_r,
            z,
        )

        # logging & visualization
        for b in range(1):  # len(fidxs)
            loss_dict[f"class_pred_{fidxs[b]}"] = class_pred[b].item()
            loss_dict[f"class_gt_{fidxs[b]}"] = class_gt[b].item()
            if self.opt.output == "shapecode":
                loss_dict[f"shape_code_pred0/{fidxs[b].replace('/','_')}"] = shape_code_pred[b, 0].item()
                loss_dict[f"shape_code_pred1/{fidxs[b].replace('/','_')}"] = shape_code_pred[b, 1].item()
                loss_dict[f"shape_code_pred2/{fidxs[b].replace('/','_')}"] = shape_code_pred[b, 2].item()
                loss_dict[f"shape_confidence0/{fidxs[b].replace('/','_')}"] = shape_code_pred[
                    b, 3
                ].item()
                loss_dict[f"shape_confidence1/{fidxs[b].replace('/','_')}"] = shape_code_pred[
                    b, 4
                ].item()
                loss_dict[f"shape_confidence2/{fidxs[b].replace('/','_')}"] = shape_code_pred[
                    b, 5
                ].item()
                loss_dict[f"shape_gt0/{fidxs[b].replace('/','_')}"] = shape_codes[b, 0].item()
                loss_dict[f"shape_gt1/{fidxs[b].replace('/','_')}"] = shape_codes[b, 1].item()
                loss_dict[f"shape_gt2/{fidxs[b].replace('/','_')}"] = shape_codes[b, 2].item()

        outputs = dict(
            object_pred_r=object_pred_r,
            contact_xyz_r=contact_xyz_r,
            class_pred=class_pred,
            shape_code_pred=shape_code_pred,
            class_gt=class_gt,
            shape_codes=shape_codes,
        )
        self.batch = batch

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
            val_loss_dict[f"val/{k}"] = v
        self.log_dict(
            val_loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        outputs["val_loss"] = loss
        return outputs

    def predict_step(self, batch):
        hand_fidxs = batch["hand_fidxs"]
        hand_verts_r = batch["hand_verts_r"]
        hand_joints_r = batch["hand_joints_r"]
        test_batch_size = hand_fidxs.shape[0]

        # decode
        joints_encoded = self.model.pos_enc(hand_joints_r.reshape(-1, 3)).reshape(
            test_batch_size, -1
        )
        (
            contact_pred,
            contact_xyz_r,
        ) = self.model.contact_net(joints_encoded, hand_verts_r)
        z_gen = np.random.normal(0.0, 1.0, size=(test_batch_size, 512))
        z_gen = torch.tensor(z_gen, dtype=joints_encoded.dtype).to(
            joints_encoded.device
        )
        output_pred, object_pred_r = self.model.dec(
            z_gen, joints_encoded, contact_xyz_r
        )
        object_pred_r = object_pred_r

        # output
        if self.opt.output == "category":
            shape_code_pred = None
            class_pred = torch.argmax(output_pred, dim=1)
            class_pred_name = [
                random.choice(self.object_names[pred]) for pred in class_pred
            ]
        elif self.opt.output == "shapecode":
            shape_code_pred = output_pred
            cateshapes = [pt[1:] for pt in self.cate_to_shape]
            catenames = [pt[0] for pt in self.cate_to_shape]
            distances = weighted_distance(
                cateshapes,
                shape_code_pred[:, : self.opt.dim_input],
                shape_code_pred[:, self.opt.dim_input :],
            )
            class_pred_name = catenames[torch.argmin(distances, dim=1)]
        else:
            logger.error(f"Invalid selector output: {self.opt.output}")
            raise ValueError("Invalid selector output")

        # logging & visualization
        self.visualize(
            object_pred_r,
            contact_xyz_r,
            hand_fidxs,
            class_pred,
            shape_code_pred,
            class_pred_gt=None,
            shape_code_gt=None,
        )

        return class_pred_name

    def loss(
        self,
        class_vecs_pred,
        class_vecs,
        shape_codes_pred,
        shape_codes,
        verts_pred,
        verts_gt,
        contact_pred,
        contact_gt,
        z=None,
    ):
        batch_size = class_vecs_pred.shape[0]
        loss = 0.0
        loss_dict = {}
        thres = torch.tensor(0.5, dtype=contact_pred.dtype, device=contact_pred.device)

        # output loss
        if self.opt.output == "category":
            output_loss = -torch.mean(
                class_vecs
                * torch.log(class_vecs_pred + 1e-8)
                * self.class_weights.unsqueeze(0)
            )
            output_weight = self.opt.loss.category_weight
        elif self.opt.output == "shapecode":
            confidence = shape_codes_pred[:, self.opt.dim_input :]
            output_loss = confidence * (
                shape_codes_pred[:, : self.opt.dim_input] - shape_codes
            )
            output_loss = torch.sqrt(torch.sum(torch.square(output_loss), dim=1))
            output_loss = output_loss - self.opt.loss.confidence_weight * torch.sum(
                confidence, dim=1
            )
            output_loss = torch.mean(output_loss, dim=0)
            output_weight = self.opt.loss.shapecode_weight

        # category acc
        class_pred_rank = torch.argsort(class_vecs_pred, dim=1, descending=True)
        class_gt = torch.argmax(class_vecs, dim=1)
        category_top1 = torch.sum(
            class_gt == class_pred_rank[:, 0]
        ) / batch_size
        category_top3 = []
        for b in range(batch_size):
            if class_gt[b] in class_pred_rank[b, :3]:
                category_top3.append(True)
            else:
                category_top3.append(False)
        category_top3 = torch.sum(torch.tensor(category_top3)) / batch_size

        # object pc
        if self.opt.loss.objectpoint == "mse":
            objectpoint_loss = F.mse_loss(
                verts_pred.view(batch_size, -1), verts_gt.view(batch_size, -1)
            )
        elif self.opt.loss.objectpoint == "icp":
            with torch.no_grad():
                icpsolution = iterative_closest_point(
                    verts_gt, verts_pred, allow_reflection=True
                )
            objectpoint_loss = F.mse_loss(
                verts_pred.view(batch_size, -1), icpsolution.Xt.view(batch_size, -1)
            )
        else:
            logger.error(f"Invalid objectpoint loss: {self.opt.loss.objectpoint}")
            raise ValueError

        # contact
        contact_loss = F.cross_entropy(contact_pred, contact_gt)
        contact_acc = (
            torch.sum((contact_pred > thres) == (contact_gt > thres))
            / contact_gt.numel()
        )

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
            output_weight * output_loss
            + self.opt.loss.objectpoint_weight * objectpoint_loss
            + self.opt.loss.contact_weight * contact_loss
            + self.kl_annealing * self.opt.loss.kl_weight * kl_loss
        )
        loss_dict["output_loss"] = output_loss
        loss_dict["category_top1"] = category_top1
        loss_dict["category_top3"] = category_top3
        loss_dict["objectpoint_loss"] = objectpoint_loss
        loss_dict["contact_loss"] = contact_loss
        loss_dict["contact_acc"] = contact_acc
        loss_dict["kl_loss"] = kl_loss

        loss_dict["loss"] = loss
        return loss, loss_dict

    def visualize(
        self,
        object_pred_r,
        contact_xyz_r,
        fidxs,
        class_pred,
        shape_code_pred=None,
        class_gt=None,
        shape_codes=None,
        mode=None,
    ):
        batch_size = object_pred_r.shape[0]
        save_dir = os.path.join(self.cfg.output_dir, "selector_training")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if mode == "val":
            sign = "val_"
        else:
            sign = ""

        # object pred
        ordered_color = plt.cm.gist_rainbow(
            np.linspace(0, 1, self.cfg.select_object.opt.n_obj_points)
        )[
            :, :-1
        ]  # no alpha channel
        object_pred_r = object_pred_r.detach().cpu().numpy()
        for b in range(batch_size):
            pc = object_pred_r[b]
            pc = o3d.utility.Vector3dVector(pc)
            pc = o3d.geometry.PointCloud(pc)
            pc.colors = o3d.utility.Vector3dVector(ordered_color)
            if self.trainer.state.fn == "fit":
                assert self.trainer.state.stage not in ["predict", "test"]
                visualize_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][18:20]}_{fidxs[b][-6:]}_object_pred.ply",
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
        contact_xyz_r = contact_xyz_r.cpu().points_list()
        for b in range(batch_size):
            pc = contact_xyz_r[b].numpy()
            if pc.shape[0] == 0:
                logger.warning(f"Empty contact xyz for hand {fidxs[b]}")
                continue
            pc = o3d.utility.Vector3dVector(pc)
            pc = o3d.geometry.PointCloud(pc)
            pc.paint_uniform_color([1, 0, 0])
            if self.trainer.state.fn == "fit":
                assert self.trainer.state.stage not in ["predict", "test"]
                visualize_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][18:20]}_{fidxs[b][-6:]}_contact_pred.ply",
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

        # class pred
        class_pred = class_pred.cpu()
        for b in range(batch_size):
            # path
            if self.trainer.state.fn == "fit":
                class_pred_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][18:20]}_{fidxs[b][-6:]}_class_pred.txt",
                )
            elif self.trainer.state.fn == "predict":
                class_pred_path = os.path.join(
                    self.cfg.results_dir, f"hand{fidxs[b]}_selector_class_pred.txt"
                )
            else:
                logger.error(f"Invalid state for visualizing: {self.trainer.state.fn}")
                raise ValueError
            # save
            with open(class_pred_path, "w") as f:
                f.write(str(class_pred[b].item()))

        # class gt
        if class_gt is not None:
            class_gt = class_gt.cpu()
            for b in range(batch_size):
                class_gt_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][18:20]}_{fidxs[b][-6:]}_class_gt.txt",
                )
                with open(class_gt_path, "w") as f:
                    f.write(str(class_gt[b].item()))

        # shape code pred
        if shape_code_pred is not None:
            shape_code_pred = shape_code_pred.detach().cpu().numpy()
            for b in range(batch_size):
                if self.trainer.state.fn == "fit":
                    shape_code_pred_path = os.path.join(
                        save_dir,
                        f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][18:20]}_{fidxs[b][-6:]}_shape_code_pred.txt",
                    )
                elif self.trainer.state.fn == "predict":
                    shape_code_pred_path = os.path.join(
                        self.cfg.results_dir, f"hand{fidxs[b]}_selector_shape_code_pred.txt"
                    )
                with open(shape_code_pred_path, "w") as f:
                    f.write(str(shape_code_pred[b]))

        # shape code gt
        if shape_codes is not None:
            shape_codes = shape_codes.cpu().numpy()
            for b in range(batch_size):
                shape_codes_path = os.path.join(
                    save_dir,
                    f"{sign}{self.current_epoch:04d}_{self.global_step:05d}_subject{fidxs[b][18:20]}_{fidxs[b][-6:]}_shape_code_gt.txt",
                )
                with open(shape_codes_path, "w") as f:
                    f.write(str(shape_codes[b]))
        return

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch % self.opt.val.vis_epoch == 0:
            self.visualize(
                outputs["object_pred_r"],
                outputs["contact_xyz_r"],
                batch["fidxs"],
                outputs["class_pred"],
                outputs["shape_code_pred"],
                outputs["class_gt"],
                outputs["shape_codes"],
                mode="val",
            )
        return

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # randomly pick fidxs
        if len(batch["fidxs"]) < self.opt.train.vis_samples:
            samples = np.arange(len(batch["fidxs"]))
        else:
            samples = np.random.choice(
                len(batch["fidxs"]), self.opt.train.vis_samples, replace=False
            )

        if self.current_epoch % self.opt.train.vis_epoch == 0:
            contact_xyz_r = []
            for s in samples:
                contact_xyz_r.append(outputs["contact_xyz_r"].points_list()[s])
            contact_xyz_r = Pointclouds(contact_xyz_r)
            self.visualize(
                outputs["object_pred_r"][samples],
                contact_xyz_r,
                np.array(batch["fidxs"])[samples],
                outputs["class_pred"],
                outputs["shape_code_pred"],
                outputs["class_gt"],
                outputs["shape_codes"],
                mode="train",
            )
        return
