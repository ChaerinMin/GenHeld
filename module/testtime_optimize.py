import glob
import json
import logging
import os
import random
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import (
    Transform3d,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
)
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
from trimesh import Trimesh

from dataset import DummyDataset, ObjectData, PaddedTensor, SelectorTestData
from dataset.base_dataset import SelectorTestDataset
from loss import ContactLoss
from metric import penetration_volume, penetration_vox
from utils import batch_normalize_mesh, merge_ho
from visualization import blend_images

mpl.rcParams["figure.dpi"] = 80
logger = logging.getLogger(__name__)


def plot_pointcloud(points, title=""):
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(x, z, -y)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.show()


class TestTimeOptimize(LightningModule):
    def __init__(self, cfg, device, accelerator, handresult):
        super().__init__()
        self.cfg = cfg
        self.device_manual = device
        self.accelerator = accelerator
        self.opt = cfg.testtime_optimize
        self.handresult = handresult
        self.contact_loss = ContactLoss(cfg, self.opt.contactloss, device=device)
        self.object_dataset = instantiate(self.cfg.object_dataset)
        self.object_dataloader = DataLoader(
            self.object_dataset,
            batch_size=handresult.batch_size,
            shuffle=True,
            collate_fn=ObjectData.collate_fn,
            pin_memory=True,
        )

        # parameters
        self.s_params = torch.ones(
            self.handresult.batch_size,
            requires_grad=True,
            device=self.device_manual,
        )
        if self.opt.scale_clip[0] == self.opt.scale_clip[1]:
            assert self.opt.scale_clip[0] == 1.0
        if self.opt.scale_clip[0] != self.opt.scale_clip[1]:
            assert self.opt.scale_clip[0] <= 1.0 <= self.opt.scale_clip[1]
            self.s_params = nn.Parameter(self.s_params)
        self.t_params = nn.Parameter(
            torch.zeros(
                self.handresult.batch_size,
                3,
                requires_grad=True,
                device=self.device_manual,
            )
        )
        self.R_params = nn.Parameter(
            torch.zeros(
                self.handresult.batch_size,
                3,
                requires_grad=True,
                device=self.device_manual,
            )
        )

        # monitoring
        self.min_losses = (
            torch.ones(self.handresult.batch_size, device=self.device_manual) * 1e10
        )

        # self.tip_idxs = torch.tensor([745, 317, 444, 556, 673], device=self.device_manual)
        self.tip_idxs = torch.tensor([763, 328, 438, 566, 683], device=self.device_manual)
        return

    def on_train_start(self):
        batch_size = self.handresult.batch_size

        # Selector data
        hand_verts_r = Pointclouds(points=self.handresult.mano_verts_r)
        hand_verts_r.estimate_normals(assign_to_self=True)
        predict_dataset = SelectorTestDataset(
            hand_fidxs=self.handresult.fidxs,
            hand_verts_r=hand_verts_r,
            hand_joints_r=self.handresult.mano_joints_r,
        )
        predict_dataloader = DataLoader(
            predict_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=SelectorTestData.collate_fn,
        )

        # Selector inference
        object_selection = instantiate(
            self.cfg.select_object,
            cfg=self.cfg,
            device=hand_verts_r.device,
            _recursive_=False,
        )
        selector = pl.Trainer(
            devices=len(self.cfg.devices), accelerator=self.accelerator
        )
        class_preds = selector.predict(
            object_selection,
            dataloaders=predict_dataloader,
            ckpt_path=self.cfg.selector_ckpt,
        )[0]

        # load object data
        train_batch = []
        for b in range(batch_size):
            class_pick = random.choice(class_preds[b])
            idx = self.object_dataset.fidxs.index(class_pick)
            train_batch.append(self.object_dataset[idx])
        train_batch = ObjectData.collate_fn(train_batch)
        data = ObjectData(**train_batch)
        data = data.to(self.device_manual)
        self.object_fidx = data.fidx
        object_verts = data.object_verts
        self.object_faces = data.object_faces
        self.object_aux = data.object_aux
        sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        # normalize to center
        self.object_verts_n, self.sampled_verts_n = self.normalize_ho(
            object_verts, sampled_verts
        )

        # force closure loss
        self.fc_contact_ind = None
        self.force_losses = None
        self.rejection_count = torch.zeros([batch_size, 1], device=self.device_manual, dtype=torch.long)
        return

    def on_test_start(self):
        batch_size = self.handresult.batch_size

        pattern = r"object_[0-9]{3}[a-z_\-]+best"
        test_batch = []
        for b in range(batch_size):
            # find checkpoint
            ckpts = glob.glob(
                os.path.join(
                    self.cfg.ckpt_dir, f"hand_{self.handresult.fidxs[b]:08d}_*.json"
                )
            )
            if len(ckpts) == 0:
                logger.error(
                    f"No checkpoint found for hand {self.handresult.fidxs[b]:08d}"
                )
                raise FileNotFoundError
            ckpt = sorted(ckpts)[-1]

            # find object name
            match = re.search(pattern, ckpt)
            if match is None:
                logger.error(f"Invalid checkpoint name: {ckpt}")
                raise FileNotFoundError
            object_fidx = match.group(0)[7:-5]

            # find object data
            idx = self.object_dataset.fidxs.index(object_fidx)
            test_batch.append(self.object_dataset[idx])

            # load checkpoint
            with open(ckpt, "r") as f:
                checkpoint = json.load(f)
                state_dict = checkpoint["state_dict"]
                self.s_params[b] = torch.tensor(state_dict["s_params"])
                self.t_params[b] = torch.tensor(state_dict["t_params"])
                self.R_params[b] = torch.tensor(state_dict["R_params"])
        self.s_params = self.s_params.to(self.device_manual)
        self.t_params = self.t_params.to(self.device_manual)
        self.R_params = self.R_params.to(self.device_manual)

        # load object data
        test_batch = ObjectData.collate_fn(test_batch)
        data = ObjectData(**test_batch)
        data = data.to(self.device_manual)
        self.object_fidx = data.fidx
        object_verts = data.object_verts
        self.object_faces = data.object_faces
        self.object_aux = data.object_aux
        sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        # normalize to center
        self.object_verts_n, self.sampled_verts_n = self.normalize_ho(
            object_verts, sampled_verts
        )

        # metric logging
        self.metric_results = None

        return

    def normalize_ho(self, object_verts, sampled_verts):
        object_verts_n, object_center, object_max_norm = batch_normalize_mesh(
            object_verts
        )
        for b in range(self.handresult.batch_size):
            logger.debug(
                f"hand {self.handresult.fidxs[b]}, [object] center: {object_center[b]}, max_norm: {object_max_norm[b]:.3f}"
            )
        if sampled_verts is not None:
            sampled_verts_n = (sampled_verts.padded - object_center) / object_max_norm
            for i in range(sampled_verts_n.shape[0]):
                sampled_verts_n[i, sampled_verts_n.split_sizes[i] :] = 0.0
            sampled_verts_n.padded = sampled_verts_n
        else:
            sampled_verts_n = None
        return object_verts_n, sampled_verts_n

    def train_dataloader(self):
        # loop Niters times
        dummy_dataset = DummyDataset(self.opt.Niters)
        loop = DataLoader(dummy_dataset, batch_size=1, shuffle=False, pin_memory=True)
        return loop

    def test_dataloader(self):
        dummy_dataset = DummyDataset(1)
        loop = DataLoader(dummy_dataset, batch_size=1, shuffle=False, pin_memory=True)
        return loop

    def configure_optimizers(self):
        if self.opt.scale_clip[0] != self.opt.scale_clip[1]:
            optimizer = torch.optim.Adam(
                [self.s_params, self.t_params, self.R_params], lr=self.opt.lr
            )
        else:
            optimizer = torch.optim.Adam(
                [self.t_params, self.R_params], lr=self.opt.lr
            )
        return optimizer

    def on_after_batch_transfer(self, batch, dataloader_idx):
        self.min_losses = self.min_losses.to(self.device_manual)
        return batch

    def forward(self, batch, batch_idx):
        self.batch_idx = batch_idx

        # transform the object verts
        # s_sigmoid = torch.sigmoid(self.s_params) * 2.2 - 1.1
        # min_s = 1.0 - self.opt.scale_clip
        # max_s = 1.0 + self.opt.scale_clip
        min_s = self.opt.scale_clip[0]
        max_s = self.opt.scale_clip[1]
        # shift = ((1-min_s) / (max_s - min_s)) * 2.0 - 2
        s_clipped = min_s + (torch.tanh(self.s_params) + 1) * 0.5 * (max_s - min_s)
        R_matrix = axis_angle_to_matrix(self.R_params)
        t = (
            Transform3d(device=self.device_manual)
            .scale(s_clipped)
            .rotate(R_matrix)
            .translate(self.t_params)
        )
        new_obj_verts_n = t.transform_points(self.object_verts_n.padded)
        new_obj_verts_n = PaddedTensor.from_padded(
            new_obj_verts_n, self.object_verts_n.split_sizes
        )
        if self.sampled_verts_n is not None:
            new_sampled_verts_n = t.transform_points(self.sampled_verts_n.padded)
            new_sampled_verts_n = PaddedTensor.from_padded(
                new_sampled_verts_n, self.sampled_verts_n.split_sizes
            )
        else:
            new_sampled_verts_n = None

        return new_obj_verts_n, new_sampled_verts_n, R_matrix, s_clipped

    def training_step(self, batch, batch_idx):
        batch_size = self.handresult.batch_size
        new_obj_verts_n, new_sampled_verts_n, R_matrix, s_clipped = self(
            batch, batch_idx
        )
        
        # force closure contact
        if self.opt.loss.force_closure_weight > 0:
            if self.fc_contact_ind is None:
                thumb = torch.zeros([batch_size, 1], device=self.device_manual, dtype=torch.long)
                new_fc_contact_ind = torch.randint(1, 5, size=[batch_size, self.opt.contactloss.fc_n_contacts-1], device=self.device_manual, dtype=torch.long)
                new_fc_contact_ind = torch.cat([thumb, new_fc_contact_ind], dim=1)
                new_fc_contact_ind = self.tip_idxs[new_fc_contact_ind]
            else:
                new_fc_contact_ind = self.fc_contact_ind.clone()
                update_ind = torch.randint(1, self.opt.contactloss.fc_n_contacts, size=[batch_size], device=self.device_manual)
                update_to = torch.randint(1, 5, size=[batch_size], device=self.device_manual)
                update_to = self.tip_idxs[update_to]
                new_fc_contact_ind[torch.arange(batch_size).to(self.device_manual), update_ind] = update_to
                switch = torch.rand([batch_size, 1], device='cuda')
                update_H = ((switch < 0.85) * (self.rejection_count < 2))  # langevin probability: 0.85
                new_fc_contact_ind = new_fc_contact_ind * (~update_H) + self.fc_contact_ind * update_H
        else:
            new_fc_contact_ind = None

        # loss
        (
            attr_loss,
            repul_loss,
            new_fc_loss,
            contact_info,
            metrics,
        ) = self.contact_loss(
            self.handresult.verts_n,
            self.handresult.faces,
            new_obj_verts_n,
            self.object_faces,
            sampled_verts=new_sampled_verts_n,
            contact_object=self.contact_object,
            partition_object=self.partition_object,
            fc_contact_ind=new_fc_contact_ind,
        )
        attraction_loss = attr_loss.loss
        repulsion_loss = repul_loss.loss
        new_force_loss = new_fc_loss.loss
        loss = (
            self.opt.loss.attraction_weight * attraction_loss
            + self.opt.loss.repulsion_weight * repulsion_loss
            + self.opt.loss.force_closure_weight * new_force_loss
        )

        # force closure Metropolis-Hasting
        new_force_losses = new_fc_loss.losses
        if self.opt.loss.force_closure_weight > 0:
            if self.force_losses is None:
                self.fc_contact_ind = new_fc_contact_ind
                self.force_losses = new_force_losses
            else:
                with torch.no_grad():
                    alpha = torch.rand(batch_size, device=new_force_losses.device, dtype=new_force_losses.dtype)
                    temperature = 0.02600707 + self.force_losses * 0.03950357  # Tengyu Liu RAL 2021
                    accept = alpha < torch.exp((new_force_losses - self.force_losses) / temperature)
                    self.fc_contact_ind[accept] = new_fc_contact_ind[accept]
                    self.force_losses[accept] = new_force_losses[accept]
                    self.rejection_count[accept] = 0
                    self.rejection_count[~accept] += 1

        # logging
        R_euler = matrix_to_euler_angles(R_matrix, "XYZ")
        logs = {
            "loss": loss.item(),
            "attraction_loss": attraction_loss.item(),
            "repulsion_loss": repulsion_loss.item(),
            "iter": batch_idx,
        }
        if self.opt.loss.force_closure_weight > 0:
            logs["new_force_loss"] = new_force_loss.item()
        for b in range(batch_size):
            logs[f"scale_{self.handresult.fidxs[b]}"] = s_clipped[b].item()
            logs[f"translate_x_{self.handresult.fidxs[b]}"] = self.t_params[b, 0].item()
            logs[f"translate_y_{self.handresult.fidxs[b]}"] = self.t_params[b, 1].item()
            logs[f"translate_z_{self.handresult.fidxs[b]}"] = self.t_params[b, 2].item()
            logs[f"rotate_x_{self.handresult.fidxs[b]}"] = R_euler[b, 0].item()
            logs[f"rotate_y_{self.handresult.fidxs[b]}"] = R_euler[b, 1].item()
            logs[f"rotate_z_{self.handresult.fidxs[b]}"] = R_euler[b, 2].item()
        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        # monitoring
        self.current_losses = (
            self.opt.loss.attraction_weight * attr_loss.losses
            + self.opt.loss.repulsion_weight * repul_loss.losses
            + self.opt.loss.force_closure_weight * new_fc_loss.losses
        )

        outputs = dict(loss=loss, new_obj_verts_n=new_obj_verts_n)
        return outputs

    def test_step(self, batch, batch_idx):
        scale_cm = (
            self.handresult.max_norm[0].item() * 100
        )  # should be half the hand length
        new_obj_verts_n, new_sampled_verts_n, *_ = self(batch, batch_idx)
        if self.handresult.verts_n.shape[0] > 1:
            logger.error(f"Batch size > 1 is not supported for evaluation.")
            raise ValueError

        # contact ratio
        *_, contact_info, metrics = self.contact_loss(
            self.handresult.verts_n,
            self.handresult.faces,
            new_obj_verts_n,
            self.object_faces,
            sampled_verts=new_sampled_verts_n,
            contact_object=self.contact_object,
            partition_object=self.partition_object,
        )
        min_dists = contact_info["min_dists"]
        min_dists = min_dists[0]  # batch = 1
        contact_zones = []
        for finger in contact_info["contact_zones"].values():
            contact_zones.extend(finger)
        num_zones = len(contact_zones)
        min_dists = min_dists[contact_zones]
        contact_ratio = (
            (min_dists * scale_cm) < self.cfg.metric.contact.thresh
        ).sum().float() / num_zones

        # to cpu
        hand_fidx = self.handresult.fidxs.cpu().numpy()[0]
        hand_verts_n_cpu = self.handresult.verts_n.cpu().numpy()[0]
        hand_faces_cpu = self.handresult.faces.verts_idx.cpu().numpy()[0]
        obj_fidx = self.object_fidx[0]
        obj_verts_n_cpu = new_obj_verts_n.padded.cpu().numpy()[0]
        obj_faces_cpu = self.object_faces.verts_idx.padded.cpu().numpy()[0]

        # penetration
        pene_depth = metrics["max_penetr"] * scale_cm
        hand_mesh_n = Trimesh(
            vertices=hand_verts_n_cpu,
            faces=hand_faces_cpu,
        )
        obj_mesh_n = Trimesh(
            vertices=obj_verts_n_cpu,
            faces=obj_faces_cpu,
        )
        pene_vox = penetration_vox(
            hand_mesh_n,
            obj_mesh_n,
            pitch=(self.cfg.metric.penetration.vox.pitch / scale_cm),
        ) * (scale_cm**3)
        pene_volume = penetration_volume(
            hand_mesh_n, obj_mesh_n, engine=self.cfg.metric.penetration.volume.engine
        ) * (scale_cm**3)
        hand_volume = hand_mesh_n.volume * (scale_cm**3)

        # simulation
        sim = instantiate(self.cfg.metric.simulation, cfg=self.cfg, _recursive_=False)
        sd = (
            sim.simulation_displacement(
                hand_fidx,
                hand_verts_n_cpu,
                hand_faces_cpu,
                obj_fidx,
                obj_verts_n_cpu,
                obj_faces_cpu,
            )
            * scale_cm
        )
        sr = sd < self.cfg.metric.simulation.opt.success_thresh

        # diversity (not for YCB)

        # logging
        logs = {
            "Hand fidx": int(hand_fidx),
            "Object fidx": -1,  # placeholder
            "Contact ratio": contact_ratio.item(),
            "Pene depth [cm]": pene_depth.item(),
            "Pene vox [cm3]": float(pene_vox),
            "Pene volume [cm3]": float(pene_volume),
            "Hand volume [cm3]": float(hand_volume),
            "SD [cm]": float(sd),
            "SR": int(sr),
        }
        self.log_dict(
            logs, prog_bar=True, on_step=True, on_epoch=True, logger=False
        )  # console logging
        logs["Hand fidx"] = str(hand_fidx)
        logs["Object fidx"] = str(obj_fidx)
        self.metric_results = logs

        return

    def on_before_optimizer_step(self, optimizer):
        # save checkpoint
        if self.batch_idx > self.opt.plot.tolerance_step:
            update_mask = self.current_losses < (
                self.min_losses - self.opt.plot.tolerance_difference
            )
            self.min_losses[update_mask] = self.current_losses[update_mask]
            for b in range(self.handresult.batch_size):
                if update_mask[b]:
                    out_ckpt_path = os.path.join(
                        self.cfg.ckpt_dir,
                        "hand_{:08d}_object_{}_best.json".format(
                            self.handresult.fidxs[b], self.object_fidx[b]
                        ),
                    )
                    checkpoint = {
                        "state_dict": {
                            "s_params": self.s_params[b].item(),
                            "t_params": self.t_params[b].tolist(),
                            "R_params": self.R_params[b].tolist(),
                        },
                        "metadata": {
                            "iter": self.batch_idx,
                        },
                    }
                    with open(out_ckpt_path, "w") as f:
                        json.dump(checkpoint, f)
            self.update_mask = update_mask
        return

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = self.handresult.batch_size
        new_obj_verts_n = outputs["new_obj_verts_n"]

        if batch_idx > self.opt.plot.tolerance_step:
            # save mesh
            merged_meshes = None
            if self.update_mask.any() or batch_idx % self.opt.plot.mesh_period == 0:
                is_textured = self.handresult.dataset.nimble
                new_obj_verts = PaddedTensor.from_padded(
                    new_obj_verts_n.padded.detach(), new_obj_verts_n.split_sizes
                )
                new_obj_verts.padded = (
                    new_obj_verts.padded * self.handresult.max_norm[:, None, None]
                ) + self.handresult.center
                for b in range(batch_size):
                    new_obj_verts.padded[b, new_obj_verts_n.split_sizes[b] :] = 0.0
                merged_meshes = merge_ho(
                    is_textured,
                    self.handresult.original_verts,
                    self.handresult.original_faces,
                    self.handresult.aux,
                    new_obj_verts,
                    self.object_faces,
                    self.object_aux,
                )
                p3d_io = IO()
                for b in range(batch_size):
                    if (
                        self.update_mask[b]
                        or batch_idx % self.opt.plot.mesh_period == 0
                    ):
                        if self.update_mask[b]:
                            tag = "best"
                        else:
                            tag = f"iter_{batch_idx:05d}"
                        out_obj_path = os.path.join(
                            self.cfg.results_dir,
                            "hand_{}_object_{}_{}.obj".format(
                                self.handresult.fidxs[b], self.object_fidx[b], tag
                            ),
                        )  # assume object batch size is 1
                        p3d_io.save_mesh(
                            merged_meshes[b], out_obj_path, include_textures=is_textured
                        )
                        logger.info(f"Saved {out_obj_path}")

            # render hand
            if self.update_mask.any() or batch_idx % self.opt.plot.render_period == 0:
                if merged_meshes is None:
                    is_textured = self.handresult.dataset.nimble
                    new_obj_verts = new_obj_verts_n.padded.detach()
                    new_obj_verts = (
                        new_obj_verts * self.handresult.max_norm
                    ) + self.handresult.center
                    for b in range(batch_size):
                        new_obj_verts[b, new_obj_verts_n.split_sizes[b] :] = 0.0
                    merged_meshes = merge_ho(
                        is_textured,
                        self.handresult.original_verts,
                        self.handresult.original_faces,
                        self.handresult.aux,
                        new_obj_verts,
                        self.object_faces,
                        self.object_aux,
                    )

                merged_meshes.verts_normals_packed()
                rendered_images = self.handresult.renderer.render(merged_meshes)
                rendered_images = (rendered_images * 255).cpu().numpy().astype(np.uint8)
                for b in range(batch_size):
                    if (
                        self.update_mask[b]
                        or batch_idx % self.opt.plot.mesh_period == 0
                    ):
                        if self.update_mask[b]:
                            tag = "best"
                        else:
                            tag = f"iter_{batch_idx:05d}"
                        # save rendered hand
                        out_rendered_path = os.path.join(
                            self.cfg.results_dir,
                            "hand_{}_object_{}_{}_rendering.png".format(
                                self.handresult.fidxs[b], self.object_fidx[b], tag
                            ),
                        )  # assume object batch size is 1
                        Image.fromarray(rendered_images[b]).save(out_rendered_path)
                        logger.info(f"Saved {out_rendered_path}")

                        # original image + rendered hand
                        blended_image = blend_images(
                            rendered_images[b],
                            self.handresult.inpainted_images[b],
                            blend_type=self.cfg.vis.blend_type,
                        )
                        out_blended_path = os.path.join(
                            self.cfg.results_dir,
                            "hand_{}_object_{}_{}_blended.png".format(
                                self.handresult.fidxs[b], self.object_fidx[b], tag
                            ),
                        )  # assume object batch size is 1
                        Image.fromarray(blended_image).save(out_blended_path)
                        logger.info(f"Saved {out_blended_path}")

            # plot force closure normal
            if self.opt.loss.force_closure_weight > 0:
                is_period = batch_idx % self.opt.plot.force_closure_period == 0
                if self.update_mask.any() or is_period:
                    self.contact_loss.plot_fc_normal(
                        hand_fidxs=self.handresult.fidxs,
                        object_fidxs=self.object_fidx,
                        iter=batch_idx,
                        denorm_center=self.handresult.center,
                        denorm_scale=self.handresult.max_norm,
                        save_mask=self.update_mask,
                        is_period=is_period,
                    )

        # plot point cloud
        if batch_idx % self.opt.plot.pc_period == self.opt.plot.pc_period - 1:
            for b in range(self.handresult.batch_size):
                new_object_size = new_obj_verts_n.split_sizes[b]
                verts_n = torch.cat(
                    [
                        self.handresult.verts_n[b],
                        new_obj_verts_n.padded[b, :new_object_size],
                    ],
                    dim=0,
                )
                plot_pointcloud(
                    verts_n,
                    title=f"hand: {self.handresult.fidxs[b]}, iter: {batch_idx}",
                )
        
        # save contact point cloud
        if batch_idx % self.opt.plot.contact_period == 0:
            self.contact_loss.plot_contact(
                hand_fidxs=self.handresult.fidxs,
                object_fidxs=self.object_fidx,
                iter=batch_idx,
            )

        return
