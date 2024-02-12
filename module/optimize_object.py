import json
import logging
import os
from typing import Union
import glob
import re

import matplotlib as mpl
import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.transforms import (Transform3d, axis_angle_to_matrix,
                                  matrix_to_euler_angles)
from pytorch_lightning import LightningModule
from torch import Tensor, nn
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
        self.object_dataset = instantiate(self.cfg.object_dataset)
        self.object_dataloader = DataLoader(
            self.object_dataset,
            batch_size=handresult.batch_size,
            shuffle=True,
            collate_fn=ObjectData.collate_fn,
        )

        # parameters
        self.s_params = nn.Parameter(
            torch.ones(
                self.handresult.batch_size, requires_grad=True, device=self.device
            )
        )
        self.t_params = nn.Parameter(
            torch.zeros(
                self.handresult.batch_size, 3, requires_grad=True, device=self.device
            )
        )
        self.R_params = nn.Parameter(
            torch.zeros(
                self.handresult.batch_size, 3, requires_grad=True, device=self.device
            )
        )

        # monitoring
        self.min_losses = torch.ones(self.handresult.batch_size, device=self.device) * 1e10

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
        self.normalize_ho()
        return
    
    def on_test_start(self):
        batch_size = self.handresult.batch_size

        pattern = r"object_[0-9]{3}[a-z_\-]+best"
        test_batch = []
        for b in range(batch_size):
            # find checkpoint
            ckpts = glob.glob(os.path.join(self.cfg.ckpt_dir, f"hand_{self.handresult.fidxs[b]:08d}_*.json"))
            if len(ckpts) == 0:
                logger.error(f"No checkpoint found for hand {self.handresult.fidxs[b]:08d}")
            ckpt = sorted(ckpts)[-1]

            # find object name
            match = re.search(pattern, ckpt)
            if match is None:
                logger.error(f"Invalid checkpoint name: {ckpt}")
            object_fidx = match.group(0)[7:-5]

            # find object data
            idx = self.object_dataset.fidxs.index(object_fidx)            
            test_batch.append(self.object_dataset[idx])

            # load checkpoint
            with open(ckpt, "r") as f:
                checkpoint = json.load(f)
                state_dict = checkpoint['state_dict']
                self.s_params[b] = state_dict['s_params']
                self.t_params[b] = torch.tensor(state_dict['t_params'])
                self.R_params[b] = torch.tensor(state_dict['R_params'])
        
        # load object data
        test_batch = ObjectData.collate_fn(test_batch)
        data = ObjectData(**test_batch)
        data = data.to(self.device)
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

    def normalize_ho(self):
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
    
    def test_dataloader(self):
        dummy_dataset = DummyDataset(1)
        loop = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
        return loop        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [self.s_params, self.t_params, self.R_params], lr=self.opt.lr
        )
        return optimizer
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        self.min_losses = self.min_losses.to(self.device)
        return batch
    
    def forward(self, batch, batch_idx):
        self.batch_idx = batch_idx 

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

        return new_object_verts, new_sampled_verts, R_matrix, s_sigmoid
    
    def training_step(self, batch, batch_idx):
        new_object_verts, new_sampled_verts, R_matrix, s_sigmoid = self(batch, batch_idx)

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
        new_object_verts, new_sampled_verts, R_matrix, s_sigmoid = self(batch, batch_idx)

        # compute metrics with the new object verts and self (handresult and the loaded parameters).
        logger.warning("Evaluation is not implemented yet.")
        return 
    
    def on_before_optimizer_step(self, optimizer):
        # save checkpoint
        update_mask = self.current_losses < self.min_losses
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
                    }
                }
                with open(out_ckpt_path, "w") as f:
                    json.dump(checkpoint, f)  
        self.update_mask = update_mask             
        return

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
        merged_meshes = None
        if self.update_mask.any() or batch_idx % self.opt.plot.mesh_period == 0:
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
                if self.update_mask[b] or batch_idx % self.opt.plot.mesh_period == 0:
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
                if self.update_mask[b] or batch_idx % self.opt.plot.mesh_period == 0:
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
