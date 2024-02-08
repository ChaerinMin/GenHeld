import logging
import os
from typing import Union

import matplotlib as mpl
import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.transforms import (
    Transform3d,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
)
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import DummyDataset, ObjectData, PaddedTensor
from loss import ContactLoss
from utils import merge_ho
from visualization import blend_images, plot_pointcloud

mpl.rcParams["figure.dpi"] = 80
logger = logging.getLogger(__name__)


class OptimizeObject(LightningModule):
    def __init__(self, cfg, handresult):
        super().__init__()
        self.cfg = cfg
        self.opt = cfg.optimize_object
        self.handresult = handresult
        self.contact_loss = ContactLoss(cfg, self.opt.contactloss)
        object_dataset = instantiate(self.cfg.object_dataset)
        self.object_dataloader = DataLoader(
            object_dataset,
            batch_size=handresult.batch_size,
            shuffle=True,
            collate_fn=ObjectData.collate_fn,
        )
        return

    def on_train_start(self):
        # randomly select objects
        for data in self.object_dataloader:
            data = ObjectData(**data)
            break
        data = data.to(self.device)
        self.object_fidx = data.fidx
        self.object_verts = data.object_verts
        self.object_faces = data.object_faces
        self.object_aux = data.object_aux
        self.sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        # normalize to center
        self.object_verts, object_center, object_max_norm = (
            OptimizeObject.batch_normalize_mesh(self.object_verts)
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
        loop = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
        return loop

    def configure_optimizers(self):
        # parameters
        self.s_params = torch.ones(
            self.handresult.batch_size, requires_grad=True, device=self.device
        )
        self.t_params = torch.zeros(
            self.handresult.batch_size, 3, requires_grad=True, device=self.device
        )
        self.R_params = torch.zeros(
            self.handresult.batch_size, 3, requires_grad=True, device=self.device
        )

        optimizer = torch.optim.Adam(
            [self.s_params, self.t_params, self.R_params], lr=self.opt.lr
        )

        return optimizer

    def training_step(self, batch, batch_idx):
        # transform the object verts
        s_sigmoid = torch.sigmoid(self.s_params) * 3.0 - 1.5
        R_matrix = axis_angle_to_matrix(self.R_params)
        t = (
            Transform3d(device=self.device)
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

        # loss
        (
            attraction_loss,
            repulsion_loss,
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

        outputs = dict(loss=loss, new_object_verts=new_object_verts)
        return outputs

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = self.handresult.batch_size
        new_object_verts = outputs["new_object_verts"]

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

        # save mesh
        if batch_idx % self.opt.plot.mesh_period == 0:
            is_textured = self.handresult.dataset.nimble
            object_aligned_verts = PaddedTensor.from_padded(
                new_object_verts.padded.detach(), new_object_verts.split_sizes
            )
            object_aligned_verts.padded = (
                object_aligned_verts.padded * self.handresult.max_norm[:, None, None]
            ) + self.handresult.center
            for b in range(batch_size):
                object_aligned_verts.padded[b, new_object_verts.split_sizes[b] :] = 0.0
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
                out_obj_path = os.path.join(
                    self.cfg.results_dir,
                    "hand_{}_object_{}_iter_{}.obj".format(
                        self.handresult.fidxs[b], self.object_fidx[b], batch_idx
                    ),
                )  # assume object batch size is 1
                p3d_io.save_mesh(
                    merged_meshes[b], out_obj_path, include_textures=is_textured
                )
                logger.info(f"Saved {out_obj_path}")

        # render hand
        if batch_idx % self.opt.plot.render_period == 0:
            if batch_idx % self.opt.plot.mesh_period != 0:
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

            # if len(merged_meshes.verts_list()) > 1:  # support only one mesh
            #     logger.error("Only support one mesh for rendering")

            merged_meshes.verts_normals_packed()
            rendered_images = self.handresult.renderer.render(merged_meshes)
            rendered_images = (rendered_images * 255).cpu().numpy().astype(np.uint8)
            for b in range(batch_size):
                # save rendered hand
                out_rendered_path = os.path.join(
                    self.cfg.results_dir,
                    "hand_{}_object_{}_iter_{}_rendering.png".format(
                        self.handresult.fidxs[b], self.object_fidx[b], batch_idx
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
                    "hand_{}_object_{}_iter_{}_blended.png".format(
                        self.handresult.fidxs[b], self.object_fidx[b], batch_idx
                    ),
                )  # assume object batch size is 1
                Image.fromarray(blended_image).save(out_blended_path)
                logger.info(f"Saved {out_blended_path}")

        # save contact point cloud
        if batch_idx % self.opt.plot.contact_period == 0:
            self.contact_loss.plot_contact(hand_fidxs=self.handresult.fidxs, object_fidxs=self.object_fidx, iter=batch_idx)

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

        return vertices, center, max_norm
