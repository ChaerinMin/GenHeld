import glob
import json
import logging
import os
import re
from typing import Union

import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    Transform3d,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
)
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader
from trimesh import Trimesh

from dataset import DummyDataset, ObjectData, PaddedTensor, SelectorTestDataset
from loss import ContactLoss
from metric import penetration_volume, penetration_vox
from utils import merge_ho
from visualization import blend_images, plot_pointcloud

mpl.rcParams["figure.dpi"] = 80
logger = logging.getLogger(__name__)


class TestTimeOptimize(LightningModule):
    def __init__(self, cfg, device, accelerator, handresult):
        super().__init__()
        self.cfg = cfg
        self.device_manual = device
        self.accelerator = accelerator
        self.opt = cfg.testtime_optimize
        self.handresult = handresult
        self.contact_loss = ContactLoss(cfg, self.opt.contactloss)
        self.object_dataset = instantiate(self.cfg.object_dataset)
        self.object_dataloader = DataLoader(
            self.object_dataset,
            batch_size=handresult.batch_size,
            shuffle=True,
            collate_fn=ObjectData.collate_fn,
            pin_memory=True,
        )

        # parameters
        self.s_params = nn.Parameter(
            torch.ones(
                self.handresult.batch_size,
                requires_grad=True,
                device=self.device_manual,
            )
        )
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

        return

    def on_train_start(self):
        # randomly select objects
        # for data in self.object_dataloader:
        #     data = ObjectData(**data)
        #     break
        batch_size = self.handresult.batch_size

        # Selector data
        hand_mesh = Meshes(verts=self.handresult.verts, faces=self.handresult.faces)
        hand_normals = hand_mesh.verts_normals_padded()
        predict_dataset = SelectorTestDataset(
            hand_theta=self.handresult.theta,
            hand_verts=self.handresult.verts,
            hand_normals=hand_normals,
        )
        predict_dataloader = DataLoader(
            predict_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        # Selector inference
        object_selection = instantiate(
            self.cfg.object_selector, self.cfg, recursive=False
        )
        selector = pl.Trainer(
            devices=len(self.cfg.devices), accelerator=self.accelerator
        )
        class_preds = selector.predict(
            object_selection,
            dataloaders=predict_dataloader,
            ckpt_path=self.cfg.selector_ckpt,
        )

        # load object data
        train_batch = []
        for b in range(batch_size):
            idx = self.object_dataset.fidxs.index(class_preds[b])
            train_batch.append(self.object_dataset[idx])
        train_batch = ObjectData.collate_fn(train_batch)
        data = ObjectData(**train_batch)
        data = data.to(self.device_manual)
        self.object_fidx = data.fidx
        self.object_verts = data.object_verts
        self.object_faces = data.object_faces
        self.object_aux = data.object_aux
        self.sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        # normalize to center
        self.normalize_ho()
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
        self.object_verts = data.object_verts
        self.object_faces = data.object_faces
        self.object_aux = data.object_aux
        self.sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        # normalize to center
        self.normalize_ho()

        # metric logging
        self.metric_results = None

        return

    def normalize_ho(self):
        self.object_verts, object_center, object_max_norm = (
            TestTimeOptimize.batch_normalize_mesh(self.object_verts)
        )
        for b in range(self.handresult.batch_size):
            logger.debug(
                f"hand {self.handresult.fidxs[b]}, [object] center: {object_center[b]}, max_norm: {object_max_norm[b]:.3f}"
            )
        if self.sampled_verts is not None:
            sampled_verts = (
                self.sampled_verts.padded - object_center
            ) / object_max_norm
            for i in range(sampled_verts.shape[0]):
                sampled_verts[i, self.sampled_verts.split_sizes[i] :] = 0.0
            self.sampled_verts.padded = sampled_verts
        return

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
        optimizer = torch.optim.Adam(
            [self.s_params, self.t_params, self.R_params], lr=self.opt.lr
        )
        return optimizer

    def on_after_batch_transfer(self, batch, dataloader_idx):
        self.min_losses = self.min_losses.to(self.device_manual)
        return batch

    def forward(self, batch, batch_idx):
        self.batch_idx = batch_idx

        # transform the object verts
        s_sigmoid = torch.sigmoid(self.s_params) * 3.0 - 1.5
        R_matrix = axis_angle_to_matrix(self.R_params)
        t = (
            Transform3d(device=self.device_manual)
            .scale(s_sigmoid)
            .rotate(R_matrix)
            .translate(self.t_params)
        )
        new_object_verts = t.transform_points(self.object_verts.padded)
        new_object_verts = PaddedTensor.from_padded(
            new_object_verts, self.object_verts.split_sizes
        )
        if self.sampled_verts is not None:
            new_sampled_verts = t.transform_points(self.sampled_verts.padded)
            new_sampled_verts = PaddedTensor.from_padded(
                new_sampled_verts, self.sampled_verts.split_sizes
            )
        else:
            new_sampled_verts = None

        return new_object_verts, new_sampled_verts, R_matrix, s_sigmoid

    def training_step(self, batch, batch_idx):
        new_object_verts, new_sampled_verts, R_matrix, s_sigmoid = self(
            batch, batch_idx
        )

        # loss
        (
            attr_loss,
            repul_loss,
            contact_info,
            metrics,
        ) = self.contact_loss(
            self.handresult.verts,
            self.handresult.faces,
            new_object_verts,
            self.object_faces,
            sampled_verts=new_sampled_verts,
            contact_object=self.contact_object,
            partition_object=self.partition_object,
        )
        attraction_loss = attr_loss.loss
        repulsion_loss = repul_loss.loss
        loss = (
            self.opt.loss.attraction_weight * attraction_loss
            + self.opt.loss.repulsion_weight * repulsion_loss
        )

        # logging
        R_euler = matrix_to_euler_angles(R_matrix, "XYZ")
        logs = {
            "loss": loss.item() / self.handresult.batch_size,
            "attraction_loss": attraction_loss.item() / self.handresult.batch_size,
            "repulsion_loss": repulsion_loss.item() / self.handresult.batch_size,
            "iter": batch_idx,
        }
        for b in range(self.handresult.batch_size):
            logs[f"scale_{self.handresult.fidxs[b]}"] = s_sigmoid[b].item()
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
        )

        outputs = dict(loss=loss, new_object_verts=new_object_verts)
        return outputs

    def test_step(self, batch, batch_idx):
        scale_cm = (
            self.handresult.max_norm[0].item() * 100
        )  # should be half the hand length
        new_object_verts, new_sampled_verts, *_ = self(batch, batch_idx)
        if self.handresult.verts.shape[0] > 1:
            logger.error(f"Batch size > 1 is not supported for evaluation.")
            raise ValueError

        # contact ratio
        *_, contact_info, metrics = self.contact_loss(
            self.handresult.verts,
            self.handresult.faces,
            new_object_verts,
            self.object_faces,
            sampled_verts=new_sampled_verts,
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
        hand_verts_cpu = self.handresult.verts.cpu().numpy()[0]
        hand_faces_cpu = self.handresult.faces.verts_idx.cpu().numpy()[0]
        obj_fidx = self.object_fidx[0]
        obj_verts_cpu = new_object_verts.padded.cpu().numpy()[0]
        obj_faces_cpu = self.object_faces.verts_idx.padded.cpu().numpy()[0]

        # penetration
        pene_depth = metrics["max_penetr"] * scale_cm
        hand_mesh = Trimesh(
            vertices=hand_verts_cpu,
            faces=hand_faces_cpu,
        )
        obj_mesh = Trimesh(
            vertices=obj_verts_cpu,
            faces=obj_faces_cpu,
        )
        pene_vox = penetration_vox(
            hand_mesh,
            obj_mesh,
            pitch=(self.cfg.metric.penetration.vox.pitch / scale_cm),
        ) * (scale_cm**3)
        pene_volume = penetration_volume(
            hand_mesh, obj_mesh, engine=self.cfg.metric.penetration.volume.engine
        ) * (scale_cm**3)
        hand_volume = hand_mesh.volume * (scale_cm**3)

        # simulation
        sim = instantiate(self.cfg.metric.simulation, cfg=self.cfg, _recursive_=False)
        sd = (
            sim.simulation_displacement(
                hand_fidx,
                hand_verts_cpu,
                hand_faces_cpu,
                obj_fidx,
                obj_verts_cpu,
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
        new_object_verts = outputs["new_object_verts"]

        if batch_idx > self.opt.plot.tolerance_step:
            # save mesh
            merged_meshes = None
            if self.update_mask.any() or batch_idx % self.opt.plot.mesh_period == 0:
                is_textured = self.handresult.dataset.nimble
                object_aligned_verts = PaddedTensor.from_padded(
                    new_object_verts.padded.detach(), new_object_verts.split_sizes
                )
                object_aligned_verts.padded = (
                    object_aligned_verts.padded
                    * self.handresult.max_norm[:, None, None]
                ) + self.handresult.center
                for b in range(batch_size):
                    object_aligned_verts.padded[
                        b, new_object_verts.split_sizes[b] :
                    ] = 0.0
                merged_meshes = merge_ho(
                    is_textured,
                    self.handresult.original_verts,
                    self.handresult.original_faces,
                    self.handresult.aux,
                    object_aligned_verts,
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
                    object_aligned_verts = new_object_verts.padded.detach()
                    object_aligned_verts = (
                        object_aligned_verts * self.handresult.max_norm
                    ) + self.handresult.center
                    for b in range(batch_size):
                        object_aligned_verts[b, new_object_verts.split_sizes[b] :] = 0.0
                    merged_meshes = merge_ho(
                        is_textured,
                        self.handresult.original_verts,
                        self.handresult.original_faces,
                        self.handresult.aux,
                        object_aligned_verts,
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

        # plot point cloud
        if batch_idx % self.opt.plot.pc_period == self.opt.plot.pc_period - 1:
            for b in range(self.handresult.batch_size):
                new_object_size = new_object_verts.split_sizes[b]
                verts = torch.cat(
                    [
                        self.handresult.verts[b],
                        new_object_verts.padded[b, :new_object_size],
                    ],
                    dim=0,
                )
                plot_pointcloud(
                    verts, title=f"hand: {self.handresult.fidxs[b]}, iter: {batch_idx}"
                )

        # save contact point cloud
        if batch_idx % self.opt.plot.contact_period == 0:
            self.contact_loss.plot_contact(
                hand_fidxs=self.handresult.fidxs,
                object_fidxs=self.object_fidx,
                iter=batch_idx,
            )

        return

    @staticmethod
    def batch_normalize_mesh(vertices: Union[Tensor, PaddedTensor]):
        # Tensor or PaddedTensor
        if isinstance(vertices, Tensor):
            verts = vertices
            padded_split = torch.tensor(
                [verts.shape[1]] * verts.shape[0], device=verts.device
            )
        elif isinstance(vertices, PaddedTensor):
            verts = vertices.padded
            padded_split = vertices.split_sizes
        else:
            logger.error(
                f"verts should be torch.Tensor or PaddedTensor, got {type(verts)}"
            )
            raise ValueError

        # batch normalize mesh
        center = verts.sum(dim=1, keepdim=True) / padded_split[:, None, None]
        verts = verts - center
        for i in range(verts.shape[0]):
            verts[i, padded_split[i] :] = 0.0
        max_norm = verts.norm(dim=2).max(dim=1)[0]
        verts = verts / max_norm.unsqueeze(1).unsqueeze(2)

        # Tensor or PaddedTensor
        if isinstance(vertices, Tensor):
            vertices = verts
        elif isinstance(vertices, PaddedTensor):
            vertices.padded = verts
        else:
            logger.error(
                f"verts should be torch.Tensor or PaddedTensor, got {type(verts)}"
            )
            raise ValueError

        return vertices, center, max_norm
