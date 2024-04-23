import glob
import json
import logging
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix
from pytorch3d.renderer import TexturesUV
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
from trimesh import Trimesh

from dataset import DummyDataset, ObjectData, PaddedTensor, SelectorTestData
from dataset.base_dataset import SelectorTestDataset
from loss import ContactLoss
from metric import penetration_volume, penetration_vox
from module.select_object import SelectObject
from utils import batch_normalize_mesh, get_hand_size, merge_ho, scale_to_bbox
from visualization import blend_images, warp_object, warp_occ

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

        self.tip_idxs = torch.tensor(
            [763, 328, 438, 566, 683], device=self.device_manual
        )
        return

    def on_train_start(self):
        batch_size = self.handresult.batch_size

        # Selector data
        hand_verts_r = Pointclouds(points=self.handresult.mano_verts_r)
        hand_verts_r.estimate_normals(assign_to_self=True)
        hand_joints_r = self.handresult.mano_joints_r
        predict_dataset = SelectorTestDataset(
            hand_fidxs=self.handresult.fidxs,
            hand_verts_r=hand_verts_r,
            hand_joints_r=hand_joints_r,
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
        class_preds, shape_code = selector.predict(
            object_selection,
            dataloaders=predict_dataloader,
            ckpt_path=self.cfg.selector.ckpt_path,
        )[0]

        # load object data
        train_batch = []
        for b in range(batch_size):
            class_prediction = class_preds[b]
            idx = self.object_dataset.get_idx(class_prediction, "trainset")
            train_batch.append(self.object_dataset[idx])
        train_batch = ObjectData.collate_fn(train_batch)
        data = ObjectData(**train_batch)
        data = data.to(self.device_manual)
        self.obj_fidx = data.fidx
        obj_verts = data.object_verts
        obj_verts_hires = data.object_verts_highres
        self.obj_faces = data.object_faces
        self.obj_faces_hires = data.object_faces_highres
        self.obj_aux_hires = data.object_aux_highres
        sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        self.obj_verts_n, self.sampled_verts_n, self.obj_verts_n_hires = (
            self.initialize_object(
                shape_code, obj_verts, sampled_verts, obj_verts_hires
            )
        )
        self.shape_code = shape_code.cpu()

        # force closure loss
        self.fc_contact_ind = None
        self.force_losses = None
        self.rejection_count = torch.zeros(
            [batch_size, 1], device=self.device_manual, dtype=torch.long
        )
        return

    def on_test_start(self):
        batch_size = self.handresult.batch_size

        pattern = r"object_.+_best"
        test_batch = []
        shape_code = []
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
            idx = self.object_dataset.get_idx(object_fidx, "testset")
            test_batch.append(self.object_dataset[idx])

            # load checkpoint
            with open(ckpt, "r") as f:
                checkpoint = json.load(f)
                state_dict = checkpoint["state_dict"]
                self.s_params[b] = torch.tensor(state_dict["s_params"])
                self.t_params[b] = torch.tensor(state_dict["t_params"])
                self.R_params[b] = torch.tensor(state_dict["R_params"])
                if "shape_code" in state_dict:
                    shape_code.append(state_dict["shape_code"])
                else:
                    shape_code = None
        self.s_params = self.s_params.to(self.device_manual)
        self.t_params = self.t_params.to(self.device_manual)
        self.R_params = self.R_params.to(self.device_manual)
        shape_code = torch.tensor(shape_code, device=self.device_manual)

        # load object data
        test_batch = ObjectData.collate_fn(test_batch)
        data = ObjectData(**test_batch)
        data = data.to(self.device_manual)
        self.obj_fidx = data.fidx
        obj_verts = data.object_verts
        obj_verts_hires = data.object_verts_highres
        self.obj_faces = data.object_faces
        self.obj_faces_hires = data.object_faces_highres
        self.obj_aux_hires = data.object_aux_highres
        sampled_verts = data.sampled_verts
        self.contact_object = data.contacts
        self.partition_object = data.partitions

        # normalize to center
        self.obj_verts_n, self.sampled_verts_n, self.obj_verts_n_hires = (
            self.initialize_object(
                shape_code, obj_verts, sampled_verts, obj_verts_hires
            )
        )

        # metric logging
        self.metric_results = None

        return

    def initialize_object(
        self, shape_code, object_verts, sampled_verts, obj_verts_hires
    ):
        if shape_code is None:
            # object verts
            object_verts_n, object_center, object_max_norm = batch_normalize_mesh(
                object_verts
            )
            for b in range(self.handresult.batch_size):
                logger.debug(
                    f"hand {self.handresult.fidxs[b]}, [object] center: {object_center[b]}, max_norm: {object_max_norm[b]:.3f}"
                )

            # sampled verts (ContactGen)
            if sampled_verts is not None:
                sampled_verts_n = (
                    sampled_verts.padded - object_center
                ) / object_max_norm
                for i in range(sampled_verts_n.shape[0]):
                    sampled_verts_n[i, sampled_verts_n.split_sizes[i] :] = 0.0
                sampled_verts_n.padded = sampled_verts_n
            else:
                sampled_verts_n = None

            # object verts hires
            obj_verts_n_hires = (
                obj_verts_hires.padded - object_center
            ) / object_max_norm.unsqueeze(1).unsqueeze(2)
            for i in range(obj_verts_n_hires.shape[0]):
                obj_verts_n_hires[i, obj_verts_hires.split_sizes[i] :] = 0.0
            obj_verts_hires.padded = obj_verts_n_hires
            obj_verts_n_hires = obj_verts_hires
        else:
            # verts 124 37
            hand_size = get_hand_size(
                (self.handresult.mano_joints_r - self.handresult.center)
                / self.handresult.max_norm.unsqueeze(1).unsqueeze(2)
            )
            # hand_size = torch.norm(self.handresult.verts_n[:, 124] - self.handresult.verts_n[:, 37])
            bbox_lengths = SelectObject.decompose_shape_code(shape_code, hand_size)
            object_verts_n, scaling_fn, _ = scale_to_bbox(
                object_verts, goal_scale=bbox_lengths
            )
            if sampled_verts is not None:
                sampled_verts_n, *_ = scale_to_bbox(
                    sampled_verts, scaling_fn=scaling_fn
                )
            else:
                sampled_verts_n = None
            obj_verts_n_hires, *_ = scale_to_bbox(
                obj_verts_hires, scaling_fn=scaling_fn
            )

        return object_verts_n, sampled_verts_n, obj_verts_n_hires

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
                [self.s_params, self.t_params, self.R_params], lr=self.opt.lr_init
            )
        else:
            optimizer = torch.optim.Adam(
                [self.t_params, self.R_params], lr=self.opt.lr_init
            )
        lr_scheduler = {
            "scheduler": getattr(torch.optim.lr_scheduler, self.opt.lr_type)(
                optimizer, **self.opt.lr_params
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def on_after_batch_transfer(self, batch, dataloader_idx):
        self.min_losses = self.min_losses.to(self.device_manual)
        return batch

    def transform_padded(self, transform, padded_tensor):
        new_padded = transform.transform_points(padded_tensor.padded)
        new_padded = PaddedTensor.from_padded(new_padded, padded_tensor.split_sizes)
        return new_padded

    def forward(self, batch, batch_idx):
        self.batch_idx = batch_idx

        # transform the object verts
        # s_sigmoid = torch.sigmoid(self.s_params) * 3.0 - 1.5  # all4000
        # self.s_sigmoid = torch.sigmoid(self.s_params) * 2.4 - 1.2  # selector4000
        min_s = self.opt.scale_clip[0]
        max_s = self.opt.scale_clip[1]
        self.s_sigmoid = min_s + (torch.tanh(self.s_params) + 1) * 0.5 * (
            max_s - min_s
        )  # differ_sim4000
        self.R_matrix = axis_angle_to_matrix(self.R_params)
        transform = (
            Transform3d(device=self.device_manual)
            .scale(self.s_sigmoid)
            .rotate(self.R_matrix)
            .translate(self.t_params)
        )
        new_obj_verts_n = self.transform_padded(transform, self.obj_verts_n)
        if self.sampled_verts_n is not None:
            new_sampled_verts_n = self.transform_padded(transform, self.sampled_verts_n)
        else:
            new_sampled_verts_n = None

        return new_obj_verts_n, new_sampled_verts_n, transform

    def training_step(self, batch, batch_idx):
        batch_size = self.handresult.batch_size
        new_obj_verts_n, new_sampled_verts_n, transform = self(batch, batch_idx)

        # force closure contact
        if self.opt.loss.force_closure_weight > 0:
            if self.fc_contact_ind is None:
                thumb = torch.zeros(
                    [batch_size, 1], device=self.device_manual, dtype=torch.long
                )
                new_fc_contact_ind = torch.randint(
                    1,
                    5,
                    size=[batch_size, self.opt.contactloss.fc_n_contacts - 1],
                    device=self.device_manual,
                    dtype=torch.long,
                )
                new_fc_contact_ind = torch.cat([thumb, new_fc_contact_ind], dim=1)
                new_fc_contact_ind = self.tip_idxs[new_fc_contact_ind]
            else:
                new_fc_contact_ind = self.fc_contact_ind.clone()
                update_ind = torch.randint(
                    1,
                    self.opt.contactloss.fc_n_contacts,
                    size=[batch_size],
                    device=self.device_manual,
                )
                update_to = torch.randint(
                    1, 5, size=[batch_size], device=self.device_manual
                )
                update_to = self.tip_idxs[update_to]
                new_fc_contact_ind[
                    torch.arange(batch_size).to(self.device_manual), update_ind
                ] = update_to
                switch = torch.rand([batch_size, 1], device="cuda")
                update_H = (switch < 0.85) * (
                    self.rejection_count < 2
                )  # langevin probability: 0.85
                new_fc_contact_ind = (
                    new_fc_contact_ind * (~update_H) + self.fc_contact_ind * update_H
                )
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
            self.obj_faces,
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
                    alpha = torch.rand(
                        batch_size,
                        device=new_force_losses.device,
                        dtype=new_force_losses.dtype,
                    )
                    temperature = (
                        0.02600707 + self.force_losses * 0.03950357
                    )  # Tengyu Liu RAL 2021
                    accept = alpha < torch.exp(
                        (new_force_losses - self.force_losses) / temperature
                    )
                    self.fc_contact_ind[accept] = new_fc_contact_ind[accept]
                    self.force_losses[accept] = new_force_losses[accept]
                    self.rejection_count[accept] = 0
                    self.rejection_count[~accept] += 1

        # logging
        # R_euler = matrix_to_euler_angles(self.R_matrix, "XYZ")
        logs = {
            f"loss_{self.handresult.fidxs[0].cpu().numpy()}": loss.item(),
            f"attraction_loss_{self.handresult.fidxs[0].cpu().numpy()}": attraction_loss.item(),
            f"repulsion_loss_{self.handresult.fidxs[0].cpu().numpy()}": repulsion_loss.item(),
            "iter": batch_idx,
        }
        if self.opt.loss.force_closure_weight > 0:
            logs[f"force_loss_{self.handresult.fidxs[0].cpu().numpy()}"] = (
                new_force_loss.item()
            )
        # for b in range(batch_size):
        #     logs[f"scale_{self.handresult.fidxs[b]}"] = self.s_sigmoid[b].item()
        #     logs[f"translate_x_{self.handresult.fidxs[b]}"] = self.t_params[b, 0].item()
        #     logs[f"translate_y_{self.handresult.fidxs[b]}"] = self.t_params[b, 1].item()
        #     logs[f"translate_z_{self.handresult.fidxs[b]}"] = self.t_params[b, 2].item()
        #     logs[f"rotate_x_{self.handresult.fidxs[b]}"] = R_euler[b, 0].item()
        #     logs[f"rotate_y_{self.handresult.fidxs[b]}"] = R_euler[b, 1].item()
        #     logs[f"rotate_z_{self.handresult.fidxs[b]}"] = R_euler[b, 2].item()
        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # monitoring
        self.current_losses = (
            self.opt.loss.attraction_weight * attr_loss.losses
            + self.opt.loss.repulsion_weight * repul_loss.losses
            + self.opt.loss.force_closure_weight * new_fc_loss.losses
        )

        outputs = dict(loss=loss, transform=transform)
        return outputs

    def test_step(self, batch, batch_idx):
        if self.handresult.verts_n.shape[0] > 1:
            logger.error(f"Batch size > 1 is not supported for evaluation.")
            raise ValueError
        scale_cm = (
            self.handresult.max_norm[0].item() * 100
        )  # should be half the hand length

        # transform hand, object
        obj_verts_n, new_sampled_verts_n, transform = self(batch, batch_idx)
        obj_verts_n_hires = self.transform_padded(transform, self.obj_verts_n_hires)

        # to cpu
        hand_fidx = self.handresult.fidxs.cpu().numpy()[0]
        obj_fidx = self.obj_fidx[0]

        # render image if wanted
        if self.opt.plot.render_eval:
            rendered, blen_proj, blen_warp = self.render_blend(obj_verts_n_hires)
            # save
            save_render_dir = os.path.join(self.cfg.output_dir, "evaluations")
            if not os.path.exists(save_render_dir):
                os.makedirs(save_render_dir)
            for b in range(self.handresult.batch_size):
                save_render_path = os.path.join(
                    save_render_dir,
                    f"hand_{hand_fidx:08d}_object_{obj_fidx}_rendering.png",
                )
                save_proj_path = save_render_path.replace("rendering", "blended")
                save_warp_path = save_render_path.replace("rendering", "warped")
                if rendered[b] is not None:
                    Image.fromarray(rendered[b]).save(save_render_path)
                Image.fromarray(blen_proj[b]).save(save_proj_path)
                if blen_warp[b] is not None:
                    Image.fromarray(blen_warp[b]).save(save_warp_path)

        # contact ratio
        *_, contact_info, metrics = self.contact_loss(
            self.handresult.verts_n,
            self.handresult.faces,
            obj_verts_n,
            self.obj_faces,
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
        hand_verts_n = self.handresult.verts_n.cpu().numpy()[0]
        hand_faces = self.handresult.faces.verts_idx.cpu().numpy()[0]
        obj_verts_n_hires = obj_verts_n_hires.padded.cpu().numpy()[0]
        obj_faces_hires = self.obj_faces_hires.verts_idx.padded.cpu().numpy()[0]

        # penetration
        pene_depth = metrics["max_penetr"] * scale_cm
        hand_mesh_n = Trimesh(
            vertices=hand_verts_n,
            faces=hand_faces,
        )
        obj_mesh_n = Trimesh(
            vertices=obj_verts_n_hires,
            faces=obj_faces_hires,
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
                hand_verts_n,
                hand_faces,
                obj_fidx,
                obj_verts_n_hires,
                obj_faces_hires,
            )
            * scale_cm
        )
        sr = sd < self.cfg.metric.simulation.opt.success_thresh

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
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True, logger=False)
        logs["Hand fidx"] = str(hand_fidx)
        logs["Object fidx"] = str(obj_fidx)
        self.metric_results = logs

        return

    def on_before_optimizer_step(self, optimizer):
        # save checkpoint
        if self.batch_idx > self.opt.plot.tolerance_step or self.batch_idx == 0:
            update_mask = self.current_losses < (
                self.min_losses - self.opt.plot.tolerance_difference
            )
            self.min_losses[update_mask] = self.current_losses[update_mask]
            for b in range(self.handresult.batch_size):
                if update_mask[b]:
                    out_ckpt_path = os.path.join(
                        self.cfg.ckpt_dir,
                        "hand_{:08d}_object_{}_best.json".format(
                            self.handresult.fidxs[b], self.obj_fidx[b]
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
                    if self.shape_code is not None:
                        checkpoint["state_dict"]["shape_code"] = self.shape_code[
                            b
                        ].tolist()
                    with open(out_ckpt_path, "w") as f:
                        json.dump(checkpoint, f)
            self.update_mask = update_mask
        return

    def merge_mesh(self, obj_verts_n_hires: PaddedTensor):
        """
        Hand comes from self
        """
        is_textured = self.handresult.dataset.nimble
        # denormalize
        obj_verts_highres = PaddedTensor.from_padded(
            obj_verts_n_hires.padded.detach(), obj_verts_n_hires.split_sizes
        )
        obj_verts_highres.padded = (
            obj_verts_highres.padded * self.handresult.max_norm[:, None, None]
        ) + self.handresult.center
        obj_verts_highres.clean(obj_verts_n_hires.split_sizes)
        # merge
        merged_meshes = merge_ho(
            is_textured,
            self.handresult.original_verts,
            self.handresult.original_faces,
            self.handresult.aux,
            obj_verts_highres,
            self.obj_faces_hires,
            self.obj_aux_hires,
        )
        return merged_meshes, obj_verts_highres

    def render_blend(
        self,
        obj_verts_n_hires: PaddedTensor,
        merged_meshes: Meshes = None,
        obj_verts_hires: PaddedTensor = None,
    ):
        """
        Hand comes from self
        """
        # merge hand + object if not done
        if merged_meshes is None:
            assert obj_verts_hires is None
            merged_meshes, obj_verts_hires = self.merge_mesh(obj_verts_n_hires)
        else:
            assert obj_verts_n_hires is None
            assert obj_verts_hires is not None
        merged_meshes.verts_normals_packed()

        # project hand + object
        projected, depth = self.handresult.renderer.render(merged_meshes)
        projected = projected.cpu().numpy()
        projected = projected * 255
        projected = projected.astype(np.uint8)
        projected = [projected[b] for b in range(self.handresult.batch_size)]
        if self.cfg.vis.what_to_render == "hand_object":
            out_rendered = projected
        elif self.cfg.vis.what_to_render == "warped_object":
            # original image, hand mask
            img = self.handresult.images
            img_handmask = self.handresult.seg
            # project hand
            material_name = list(self.handresult.aux["texture_images"].keys())[0]
            texture_images = self.handresult.aux["texture_images"][material_name]
            hand_textures = TexturesUV(
                texture_images,
                self.handresult.original_faces.textures_idx,
                self.handresult.aux["verts_uvs"],
            )
            hand_meshes = Meshes(
                self.handresult.original_verts,
                self.handresult.original_faces.verts_idx,
                hand_textures,
            )
            hand_meshes.verts_normals_packed()
            proj_hand, depth_hand = self.handresult.renderer.render(hand_meshes)
            proj_hand = proj_hand.cpu().numpy()
            proj_hand = (proj_hand * 255).astype(np.uint8)
            # projected hand mask
            proj_handmask, depth_hand = self.handresult.renderer.render_wo_texture(
                hand_meshes
            )
            proj_handmask = proj_handmask[..., 3]
            proj_handmask = proj_handmask.cpu().numpy()
            # projected hand occlusion
            proj_handocc = depth != depth_hand
            proj_handocc = proj_handocc.squeeze(3)
            proj_handocc = proj_handocc.cpu().numpy().astype(np.float32)
            # project object
            assert (
                len(self.obj_aux_hires["texture_images"]) == 1
            ), "Only one texture image is supported"
            material_name = list(self.obj_aux_hires["texture_images"].keys())[0]
            texture_images = self.obj_aux_hires["texture_images"][material_name].to(
                obj_verts_hires.device
            )
            obj_textures = TexturesUV(
                texture_images,
                self.obj_faces_hires.textures_idx.padded,
                self.obj_aux_hires["verts_uvs"].padded,
            )
            obj_meshes = Meshes(
                obj_verts_hires.padded,
                self.obj_faces_hires.verts_idx.padded,
                obj_textures,
            )
            obj_meshes.verts_normals_packed()
            proj_object, _ = self.handresult.renderer.render(obj_meshes)
            proj_object = proj_object.cpu().numpy()
            proj_object = (proj_object * 255).astype(np.uint8)
            # warp object & occlusion
            warped = []
            for b in range(self.handresult.batch_size):
                warp, M = warp_object(proj_object[b], img[b], proj_hand[b, :, :, :3])
                if warp is None:
                    warped.append(None)
                    continue
                img_handocc = warp_occ(
                    proj_handocc[b], proj_handmask[b], M, img_handmask[b]
                )
                assert img_handocc.max() <= 1.0
                img_valid = np.sum(img, axis=3) > 15
                warp[..., 3] = warp[..., 3] * img_handocc * img_valid
                warp = warp.astype(np.uint8)
                warped.append(warp)
            out_rendered = warped
        else:
            logger.error(f"Invalid what_to_render: {self.cfg.vis.what_to_render}")
            raise ValueError

        # blend
        if self.cfg.vis.where_to_render == "raw":
            background = self.handresult.images
        elif self.cfg.vis.where_to_render == "inpainted":
            background = self.handresult.inpainted_images
        blended_projected = []
        for b in range(self.handresult.batch_size):
            blended = blend_images(
                projected[b],
                background[b],
                blend_type=self.cfg.vis.blend_type,
            )
            blended_projected.append(blended)
        if self.cfg.vis.what_to_render == "warped_object":
            blended_warped = []
            for b in range(self.handresult.batch_size):
                if warped[b] is None:
                    blended_warped.append(None)
                    continue
                blended = blend_images(
                    warped[b],
                    background[b],
                    blend_type=self.cfg.vis.blend_type,
                )
                blended_warped.append(blended)
        else:
            blended_warped = None
        return out_rendered, blended_projected, blended_warped

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = self.handresult.batch_size
        transform = outputs["transform"]
        obj_verts_n_hires = self.transform_padded(transform, self.obj_verts_n_hires)

        if batch_idx > self.opt.plot.tolerance_step or batch_idx == 0:
            # mesh
            merged_meshes = None
            if self.update_mask.any() or batch_idx % self.opt.plot.mesh_period == 0:
                merged_meshes, obj_verts_hires = self.merge_mesh(obj_verts_n_hires)
                p3d_io = IO()
                for b in range(batch_size):
                    if (
                        self.update_mask[b]
                        or batch_idx % self.opt.plot.mesh_period == 0
                    ):
                        if batch_idx == 0:
                            tag = "init"
                        elif self.update_mask[b]:
                            tag = "best"
                        else:
                            tag = f"iter_{batch_idx:05d}"
                        out_obj_path = os.path.join(
                            self.cfg.results_dir,
                            "hand_{}_object_{}_{}.obj".format(
                                self.handresult.fidxs[b], self.obj_fidx[b], tag
                            ),
                        )
                        p3d_io.save_mesh(merged_meshes[b], out_obj_path)
                        logger.info(f"Saved {out_obj_path}")

            # rendering
            if self.update_mask.any() or batch_idx % self.opt.plot.render_period == 0:
                ren_images, blen_proj, blen_warp = self.render_blend(
                    None, merged_meshes, obj_verts_hires
                )
                for b in range(batch_size):
                    if (
                        self.update_mask[b]
                        or batch_idx % self.opt.plot.mesh_period == 0
                    ):
                        if batch_idx == 0:
                            tag = "init"
                        elif self.update_mask[b]:
                            tag = "best"
                        else:
                            tag = f"iter_{batch_idx:05d}"

                        out_ren_path = os.path.join(
                            self.cfg.results_dir,
                            "hand_{}_object_{}_{}_rendering.png".format(
                                self.handresult.fidxs[b], self.obj_fidx[b], tag
                            ),
                        )
                        out_proj_path = out_ren_path.replace("rendering", "blended")
                        out_warp_path = out_ren_path.replace("rendering", "warped")
                        if ren_images[b] is not None:
                            Image.fromarray(ren_images[b]).save(out_ren_path)
                            logger.info(f"Saved {out_ren_path}")
                        Image.fromarray(blen_proj[b]).save(out_proj_path)
                        logger.info(f"Saved {out_proj_path}")
                        if blen_warp[b] is not None:
                            Image.fromarray(blen_warp[b]).save(out_warp_path)
                            logger.info(f"Saved {out_warp_path}")

            # plot force closure normal
            if self.opt.loss.force_closure_weight > 0:
                is_period = batch_idx % self.opt.plot.force_closure_period == 0
                if self.update_mask.any() or is_period:
                    self.contact_loss.plot_fc_normal(
                        hand_fidxs=self.handresult.fidxs,
                        object_fidxs=self.obj_fidx,
                        iter=batch_idx,
                        denorm_center=self.handresult.center,
                        denorm_scale=self.handresult.max_norm,
                        save_mask=self.update_mask,
                        is_period=is_period,
                    )

        # plot point cloud
        if batch_idx % self.opt.plot.pc_period == self.opt.plot.pc_period - 1:
            for b in range(self.handresult.batch_size):
                new_object_size = obj_verts_n_hires.split_sizes[b]
                verts_n = torch.cat(
                    [
                        self.handresult.verts_n[b],
                        obj_verts_n_hires.padded[b, :new_object_size],
                    ],
                    dim=0,
                )
                plot_pointcloud(
                    verts_n,
                    title=f"hand: {self.handresult.fidxs[b]}, iter: {batch_idx}",
                )

        # save contact point cloud
        if batch_idx % self.opt.plot.contact_period == self.opt.plot.contact_period - 1:
            self.contact_loss.plot_contact(
                hand_fidxs=self.handresult.fidxs,
                object_fidxs=self.obj_fidx,
                iter=batch_idx,
            )

        return
