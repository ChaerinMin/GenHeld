import logging
import os

import numpy as np
import torch
import yaml
import open3d as o3d
from matplotlib import pyplot as plt
from hydra.utils import instantiate
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import RotateAxisAngle, axis_angle_to_matrix
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SelectorData
from utils import get_NN
from submodules.HiFiHR.utils.manopth.manolayer import ManoLayer

from .pointnet import PointNetFeaturePropagation, PointNetSetAbstraction, STNkd

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, dim_hidden=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)

        self.fc2 = nn.Linear(dim_hidden, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class CrossAttentionModule(nn.Module):
    def __init__(self, query_dim, key_value_dim):
        super().__init__()
        hidden_dim = max(query_dim, key_value_dim)

        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_value_dim, hidden_dim)
        self.value_projection = nn.Linear(key_value_dim, hidden_dim)

        self.scale = hidden_dim**0.5  # Scaling factor for dot products

    def forward(self, query, key_value):
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key_value)
        value_projected = self.value_projection(key_value)

        scores = (
            torch.matmul(query_projected, key_projected.transpose(-2, -1)) / self.scale
        )
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value_projected)

        return attended_values


class PointNetObject(nn.Module):
    def __init__(self, hc, in_feature):
        super().__init__()
        self.hc = hc
        self.in_feature = in_feature

        self.enc_sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=self.in_feature,
            mlp=[self.hc, self.hc * 2],
            group_all=False,
        )
        self.enc_sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.25,
            nsample=64,
            in_channel=self.hc * 2 + 3,
            mlp=[self.hc * 2, self.hc * 4],
            group_all=False,
        )
        self.enc_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=self.hc * 4 + 3,
            mlp=[self.hc * 4, self.hc * 8],
            group_all=True,
        )

    def forward(self, l0_xyz, l0_points):
        l1_xyz, l1_points = self.enc_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.enc_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.enc_sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, self.hc * 8)

        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, x


class PointNetContact(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        x = x.transpose(2, 1)  # [B, N, D]
        if D > 3:
            x, feature = x.split(3, dim=2)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # [B, N, 64]

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # global feature: [B, 1024]
        if self.global_feat:
            return x, None, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)  # N  [B, 1024, N]
            return torch.cat([x, pointfeat], 1), None, trans_feat


class ContactNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet = PointNetContact(
            global_feat=False, feature_transform=False, channel=6
        )
        self.convfuse = torch.nn.Conv1d(3778, 3000, 1)
        self.bnfuse = nn.BatchNorm1d(3000)

        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = torch.nn.Conv1d(128, 1, 1)
        return

    def forward(self, hand_theta_r, hand_verts_nr, hand_normals_nr):
        batch_size = hand_verts_nr.shape[0]
        n_pts = hand_verts_nr.shape[1]
        x_pc = torch.cat([hand_verts_nr, hand_normals_nr], dim=1)
        x_pc, *_ = self.pointnet(x_pc)
        x = torch.cat([x_pc, hand_theta_r], dim=1).permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse(self.convfuse(x)))
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = torch.sigmoid(x)
        x = x.view(batch_size, n_pts)
        return (x,)


class SelectObjectNet(nn.Module):
    def __init__(self, opt):
        dim_R = 3

        # category
        self.cate_en_bn1 = nn.BatchNorm1d(opt.n_class + dim_R)
        self.cate_en_res1 = ResBlock(
            opt.n_class + dim_R + opt.dim_hand_pose, opt.category.dim_hidden
        )
        self.cate_en_res2 = ResBlock(
            opt.category.dim_hidden + opt.n_class + dim_R + opt.dim_hand_pose,
            opt.category.dim_hidden,
        )
        self.cate_de_res2 = ResBlock(
            opt.category.dim_latent + opt.dim_hand_pose, opt.category.dim_hidden
        )
        self.cate_de_res1 = ResBlock(
            opt.category.dim_hidden + opt.dim_latent + opt.dim_hand_pose + dim_R,
            opt.category.dim_hidden,
        )
        self.cate_de_mlp = nn.Linear(opt.category.dim_hidden, opt.n_class)
        self.cate_softmax = nn.Softmax()

        # object pc
        obj_dim_feat = 3 + opt.dim_hand_pose + opt.dim_hand_verts
        self.obj_en_pointnet = PointNetObject(opt.object.dim_hidden, obj_dim_feat)
        # self.obj_de_mlp4 = nn.Linear(opt.object.dim_hidden * 8 + opt.dim_latent, opt.object.dim_hidden * 8)
        # self.obj_de_bn4 = nn.BatchNorm1d(opt.object.dim_hidden * 8)
        # self.obj_de_drop4 = nn.Dropout(0.1)
        # self.obj_de_mlp3 = nn.Linear(opt.object.dim_hidden * 4, opt.object.dim_hidden * 8)
        # self.obj_de_bn3 = nn.BatchNorm1d(opt.object.dim_hidden * 8)
        # self.obj_de_drop3 = nn.Dropout(0.1)
        # self.obj_de_mlp2 = nn.Linear(opt.object.dim_hidden * 2, opt.object.dim_hidden * 4)
        # self.obj_de_bn2 = nn.BatchNorm1d(opt.object.dim_hidden * 4)
        # self.obj_de_drop2 = nn.Dropout(0.1)
        self.obj_de_mlp1 = nn.Linear(opt.dim_latent, opt.object.dim_hidden * 2)
        self.obj_de_bn1 = nn.BatchNorm1d(opt.object.dim_hidden * 2)
        self.obj_de_drop1 = nn.Dropout(0.1)
        self.obj_de_pointnet3 = PointNetFeaturePropagation(
            in_channel=opt.object.dim_hidden * 8 + opt.object.dim_hidden * 4,
            mlp=[opt.object.dim_hidden * 8, opt.object.dim_hidden * 4],
        )
        self.obj_de_pointnet2 = PointNetFeaturePropagation(
            in_channel=opt.object.dim_hidden * 4 + opt.object.dim_hidden * 2,
            mlp=[opt.object.dim_hidden * 4, opt.object.dim_hidden * 2],
        )
        self.obj_de_pointnet1 = PointNetFeaturePropagation(
            in_channel=opt.object.dim_hidden * 2 + obj_dim_feat,
            mlp=[opt.object.dim_hidden * 2, opt.object.dim_hidden * 2],
        )
        self.obj_de_conv1 = nn.Conv1d(
            opt.object.dim_hidden * 2, opt.object.dim_hidden * 2, 1
        )
        self.obj_de_conv_bn1 = nn.BatchNorm1d(opt.object.dim_hidden * 2)
        self.obj_de_conv_drop1 = nn.Dropout(0.1)
        self.obj_de_conv2 = nn.Conv1d(opt.object.dim_hidden * 2, 1, 1)

        # cross attention
        self.en_ca1 = CrossAttentionModule(
            query_dim=opt.object.dim_hidden * 4, key_value_dim=opt.category.dim_hidden
        )
        self.en_ca2 = CrossAttentionModule(
            query_dim=opt.object.dim_hidden * 8, key_value_dim=opt.category.dim_hidden
        )
        self.de_ca2 = CrossAttentionModule(
            query_dim=opt.object.dim_hidden * 2, key_value_dim=opt.category.dim_hidden
        )
        self.de_ca1 = CrossAttentionModule(
            query_dim=opt.object.dim_hidden * 2 + obj_dim_feat,
            key_value_dim=opt.category.dim_hidden,
        )

        # contact
        self.contact_net = ContactNet()

        # fusion
        self.fusion_en = ResBlock(
            opt.category.dim_hidden + opt.object.dim_hidden * 8, opt.fusion.dim_hidden
        )
        self.fusion_en_mu = nn.Linear(opt.fusion.dim_hidden, opt.dim_latent)
        self.fusion_en_var = nn.Linear(opt.fusion.dim_hidden, opt.dim_latent)

        return

    def enc(self, class_vec, hand_theta, contact_xyz, object_pcs, object_normals):
        # object pc
        obj_f0 = torch.cat([object_normals, hand_theta[:, 3:], contact_xyz], 1)
        *_, obj_x1 = self.obj_en_pointnet(object_pcs[:, :3, :], obj_f0)

        # category
        cate_x0 = torch.cat([class_vec, hand_theta[:, :3]], dim=1).float()
        cate_x0 = self.cate_en_bn1(cate_x0)
        cate_x0 = torch.cat([cate_x0, hand_theta[:, 3:]], dim=1)
        cate_x1 = self.cate_en_res1(cate_x0)
        cate_x1 = self.en_ca1(object_pcs[:, :3, :], cate_x1)
        cate_x1 = self.cate_en_res2(torch.cat([cate_x1, cate_x0], dim=1))

        # fusion
        cate_x1 = self.en_ca2(obj_x1, cate_x1)
        z = self.fusion_en(torch.cat([cate_x1, obj_x1], dim=1))
        z = torch.distributions.normal.Normal(
            self.fusion_en_mu(z), F.softplus(self.fusion_en_var(z))
        )
        return z

    def dec(self, z, hand_theta, contact_xyz):
        # object pc
        obj_x2 = torch.cat([z, hand_theta[:, 3:], contact_xyz], dim=1)
        obj_x2 = F.relu(self.obj_de_bn1(self.obj_de_mlp1(obj_x2)))
        obj_x2 = self.obj_de_drop1(obj_x2, inplace=True)
        obj_x1 = obj_x2.view(obj_x2.size()[0], obj_x2.size()[1], 1)
        obj_x1 = self.obj_de_pointnet3(hand_theta[:, 3:], obj_x1)
        obj_x1 = self.obj_de_pointnet2(hand_theta[:, 3:], obj_x1)
        object_pred = self.obj_de_pointnet1(hand_theta[:, 3:], obj_x1)
        object_pred = F.relu(
            self.obj_de_conv_bn1(self.obj_de_conv1(object_pred)), inplace=True
        )
        object_pred = self.obj_de_conv_drop1(object_pred)
        object_pred = self.obj_de_conv2(object_pred)

        # category
        cate_x = torch.cat([z, hand_theta[:, 3:]], dim=1)
        category_pred = self.cate_de_res2(cate_x)
        category_pred = self.de_ca2(obj_x2, category_pred)
        category_pred = torch.cat([category_pred, cate_x, hand_theta[:, :3]], dim=1)
        category_pred = self.de_ca1(obj_x1, category_pred)
        category_pred = self.cate_de_res1(category_pred)
        category_pred = self.cate_de_mlp(category_pred)
        category_pred = self.cate_softmax(category_pred)

        return category_pred, object_pred


class SelectObject(LightningModule):
    def __init___(self, opt, cfg):
        super().__init__()
        self.opt = opt
        self.cfg = cfg

        self.model = SelectObjectNet(opt)
        self.n_class = self.opt.n_class
        self.manolayer = ManoLayer()
        return

    def train_dataloader(self):
        selector_dataset = instantiate(
            self.cfg.selector_dataset, self.cfg, recursive=False
        )
        val_size = int(len(selector_dataset) * self.opt.val.ratio)
        train_dataset, self.val_dataset = torch.utils.data.random_split(
            selector_dataset, [len(selector_dataset) - val_size, val_size]
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.opt.train_batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=SelectorData.collate_fn,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.opt.val_batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=SelectorData.collate_fn,
        )
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

    def on_predict_start(self):
        object_names_path = os.path.join(self.cfg.output_dir, "selector_objects.yaml")
        with open(object_names_path, "r") as f:
            self.object_names = yaml.safe_load(f)
            logger.info(f"Loaded {object_names_path}")
        return

    def on_train_epoch_start(self):
        self.kl_annealing = max(
            (self.current_epoch + 1) / self.opt.loss.kl_annealing_epoch, 1.0
        )
        return

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
        loss_dict = dict(class_pred=class_pred)
        if not unsupervised_only:
            # category
            category_loss = -torch.sum(class_gt * torch.log(class_pred + 1e-8))

            # object pc
            verts_pred = verts_pred.view(batch_size, -1)
            verts_gt = verts_gt.view(batch_size, -1)
            objectpoint_loss = F.mse_loss(verts_pred, verts_gt)

            # contact
            contact_loss = F.binary_cross_entropy(
                contact_pred, contact_gt, weight=max(contact_gt, 0.5), reduction="none"
            )
            contact_loss = torch.mean(contact_loss)

            # KL-divergence
            if z is not None:
                q_z = torch.distributions.normal.Normal(z.mean, z.scale)
                p_z = torch.distributions.normal.Normal(
                    loc=torch.tensor(
                        np.zeros([batch_size, self.opt.dim_latent]), requires_grad=False
                    )
                    .to(z.device)
                    .type(z.dtype),
                    scale=torch.tensor(
                        np.ones([batch_size, self.opt.dim_latent]), requires_grad=False
                    )
                    .to(z.device)
                    .type(z.dtype),
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
            loss_dict["class_gt"] = class_gt
            loss_dict["category_loss"] = category_loss
            loss_dict["objectpoint_loss"] = objectpoint_loss
            loss_dict["contact_loss"] = contact_loss
            loss_dict["kl_loss"] = kl_loss

        # unsupervised loss
        nn, _ = get_NN(contact_xyz, verts_pred)
        nn = 100.0 * torch.sqrt(nn)
        hand_contact = 1.0 - 2 * (torch.sigmoid(nn * 2) - 0.5)
        consistency_loss = torch.mean(hand_contact < 0.5)
        loss += consistency_loss
        loss_dict["consistency_loss"] = (
            self.opt.loss.consistency_weight * consistency_loss
        )

        return loss, loss_dict

    def forward(self, batch):
        data = SelectorData(**batch)
        fidxs = data.fidxs
        hand_theta = data.hand_theta
        hand_verts_n = data.hand_verts_n
        hand_contacts_n = data.hand_contacts_n
        class_vecs = data.class_vecs
        object_pcs_n = data.object_pcs_n

        # contact_net
        hand_theta_r = hand_theta.clone()
        hand_theta_r[:, :3] = 0.0
        t = RotateAxisAngle(
            axis_angle_to_matrix(-hand_theta[:, :3]),
            dtype=hand_theta.dtype,
            device=hand_theta.device,
        )
        hand_verts_nr = t.transform_points(hand_verts_n)
        hand_normals_nr = Pointclouds([hand_verts_nr]).normals_padded()
        contact_pred = self.model.contact_net(
            hand_theta_r, hand_verts_nr, hand_normals_nr
        )

        # encode - decode
        contact_xyz_n = hand_verts_n[contact_pred > 0.5]
        z = self.model.enc(
            class_vecs,
            hand_theta,
            contact_xyz_n,
            object_pcs_n,
            object_pcs_n.normals_padded(),
        )
        z_s = z.rsample()
        class_pred, object_pred_n = self.model.dec(z_s, hand_theta, contact_xyz_n)

        # loss
        loss, loss_dict = self.loss(
            class_pred,
            class_vecs,
            object_pred_n,
            object_pcs_n,
            contact_pred,
            hand_contacts_n,
            contact_xyz_n,
            z,
        )

        # logging & visualization
        outputs = dict(
            object_pred_n=object_pred_n,
            contact_xyz_n=contact_xyz_n,
        )
        if self.cfg.debug:
            self.batch = batch
            outputs["hand_theta_r"] = hand_theta_r
            outputs["hand_verts_nr"] = hand_verts_nr
            outputs["hand_normals_nr"] = hand_normals_nr

        return loss, loss_dict, outputs

    def training_step(self, batch):
        loss, loss_dict, outputs = self(batch)
        loss_dict["loss"] = loss
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return outputs

    def validation_step(self, batch):
        loss, loss_dict, outputs = self(batch)
        loss_dict["val_loss"] = loss
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return outputs

    def predict_step(self, batch):
        hand_fidxs = batch["hand_fidxs"]
        hand_theta = batch["hand_theta"]
        hand_verts_n = batch["hand_verts_n"]
        test_batch_size = hand_theta.shape[0]

        # z
        z_gen = np.random.normal(0.0, 1.0, size=(test_batch_size, self.opt.dim_latent))
        z_gen = torch.tensor(z_gen, dtype=hand_theta.dtype).to(hand_theta.device)

        # contact_net
        hand_theta_r = hand_theta.clone()
        hand_theta_r[:, :3] = 0.0
        t = RotateAxisAngle(
            axis_angle_to_matrix(-hand_theta[:, :3]),
            dtype=hand_theta.dtype,
            device=hand_theta.device,
        )
        hand_verts_nr = t.transform_points(hand_verts_n)
        hand_normals_nr = Pointclouds([hand_verts_nr]).normals_padded()
        contact_pred = self.model.contact_net(
            hand_theta_r, hand_verts_nr, hand_normals_nr
        )

        # decode
        contact_xyz_n = hand_verts_n[contact_pred > 0.5]
        class_pred, object_pred_n = self.model.dec(z_gen, hand_theta, contact_xyz_n)

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
        if self.cfg.debug:
            self.visualize_debug(
                hand_theta,
                hand_theta_r,
                hand_verts_n,
                hand_verts_nr,
                hand_normals_nr,
                hand_fidxs,
            )

        # output
        class_pred = torch.argmax(class_pred, dim=1)
        class_pred = self.object_names[class_pred]

        return class_pred

    def visualize(self, object_pred_n, contact_xyz_n, fidxs):
        batch_size = object_pred_n.shape[0]
        object_pred_n = object_pred_n.detach().cpu().numpy()
        contact_xyz_n = contact_xyz_n.detach().cpu().numpy()
        for b in batch_size:
            pc = o3d.utility.Vector3dVector(np.concatenate(object_pred_n[b], contact_xyz_n[b]))
            pc = o3d.geometry.PointCloud(pc)
            color = np.zeros((pc.shape[0], 3))
            color[object_pred_n.shape[1]:] = [1, 0, 0]
            pc.colors = o3d.utility.Vector3dVector(color)

            # save
            if self.trainer.state.fn == "fit":
                assert self.trainer.state.stage not in ["predict", "test"]
                visualize_path = os.path.join(self.cfg.selector_ckpt_dir, f"hand_{fidxs[b]}_selector_pc.ply")
            elif self.trainer.state.fn == "predict":
                visualize_path = os.path.join(self.cfg.results_dir, f"{self.current_epoch}_{self.global_step}_hand_{fidxs[b]}.ply")
            else:
                logger.error(f"Invalid state for visualizing: {self.trainer.state.fn}")
            o3d.io.write_point_cloud(visualize_path, pc)
            logger.info(f"Visualized the selector results at {visualize_path}")

        return

    def visualize_debug(
        self,
        hand_theta,
        hand_theta_r,
        hand_verts_n,
        hand_verts_nr,
        hand_normals_nr,
        fidxs,
        hand_contacts_n=None,
        object_pcs_n=None,
    ):
        batch_size = hand_theta.shape[0]
        save_dir = os.path.join(self.cfg.output_dir, "selector_debug")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # hand theta
        _, joints = self.manolayer(hand_theta)
        _, joints_r = self.manolayer(hand_theta_r)
        bones = [
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 1], [1, 2], [2, 3], [3, 4],
        ]
        bone_colors = [
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        ]
        for b in batch_size:
            points = o3d.utility.Vector3dVector(joints[b])
            lines = o3d.utility.Vector2iVector(bones)
            colors = np.array(bone_colors)
            line_set = o3d.geometry.LineSet(points, lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            save_path = os.path.join(save_dir, f"{fidxs[b]}_hand_theta.ply")
            o3d.io.write_line_set(save_path, line_set)
            logger.info(f"Saved {save_path}")
        for b in batch_size:
            points = o3d.utility.Vector3dVector(joints_r[b])
            lines = o3d.utility.Vector2iVector(bones)
            colors = np.array(bone_colors)
            line_set = o3d.geometry.LineSet(points, lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            save_path = os.path.join(save_dir, f"{fidxs[b]}_hand_theta_r.ply")
            o3d.io.write_line_set(save_path, line_set)
            logger.info(f"Saved {save_path}")

        # hand verts
        color = plt.cm.viridis(np.linspace(0, 1, hand_verts_n.shape[1]))[:, :-1]  # no alpha channel
        for b in batch_size:
            pc = o3d.utility.Vector3dVector(hand_verts_n[b])
            pc = o3d.geometry.PointCloud(pc)
            pc.colors = o3d.utility.Vector3dVector(color)
            save_path = os.path.join(save_dir, f"{fidxs[b]}_hand_verts_n.ply")
            o3d.io.write_point_cloud(save_path, pc)
            logger.info(f"Saved {save_path}")
        
        # hand verts nr with normals
        for b in batch_size:
            pc = o3d.utility.Vector3dVector(hand_verts_nr[b])
            pc = o3d.geometry.PointCloud(pc)
            color = (hand_normals_nr[b] + 1.0) / 2.0
            pc.colors = o3d.utility.Vector3dVector(color)
            save_path = os.path.join(save_dir, f"{fidxs[b]}_hand_verts_nr.ply")
            o3d.io.write_point_cloud(save_path, pc)
            logger.info(f"Saved {save_path}")
        
        # input object pc, input contacts 
        if hand_contacts_n is not None and object_pcs_n is not None:
            for b in batch_size:
                pc = o3d.utility.Vector3dVector(np.concatenate(object_pcs_n[b], hand_contacts_n[b]))
                pc = o3d.geometry.PointCloud(pc)
                color = np.zeros((pc.shape[0], 3))
                color[object_pcs_n.shape[1]:] = [1, 0, 0]
                pc.colors = o3d.utility.Vector3dVector(color)
                save_path = os.path.join(save_dir, f"{fidxs[b]}_input_obj_contacts.ply")
                o3d.io.write_point_cloud(save_path, pc)
                logger.info(f"Saved {save_path}")    

        return

    def on_validation_batch_end(self, outputs, batch):
        if self.current_epoch % self.opt.visualize_epoch == 0:
            self.visualize(outputs["object_pred_n"], outputs["contact_xyz_n"], batch["fidxs"])
        if self.cfg.debug and self.current_epoch == 0:
            self.visualize_debug(
                batch["hand_theta"],
                outputs["hand_theta_r"],
                batch["hand_verts_n"],
                outputs["hand_verts_nr"],
                outputs["hand_normals_nr"],
                batch["fidxs"],
                batch["hand_contacts_n"],
                batch["object_pcs_n"],
            )
        return
