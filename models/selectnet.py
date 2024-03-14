import math

import torch
from torch.nn import functional as F
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
from pytorch3d.ops import knn_points
from torch import nn

from models.pointnet import PointNetPropagation, PointNetSetAbstraction, STNkd


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


class CrossAttention(nn.Module):
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
    def __init__(self, hc, in_feature, group_all=False):
        super().__init__()
        self.hc = hc
        self.in_feature = in_feature

        self.enc_sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=self.in_feature,
            mlp=[self.hc, self.hc * 2],
            group_all=group_all,
        )
        self.enc_sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.25,
            nsample=64,
            in_channel=self.hc * 2 + 3,
            mlp=[self.hc * 2, self.hc * 4],
            group_all=group_all,
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
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
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
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.pointnet = PointNetContact(
            global_feat=False, feature_transform=False, channel=6
        )
        self.convfuse = nn.Conv1d(778 + 48, 778, 1)
        self.bnfuse = nn.BatchNorm1d(778)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 1, 1)
        return

    @staticmethod
    def create_ellipsoid(center, axes, num_points):
        x0, y0, z0 = center
        a, b, c = axes

        # Parametric angles
        sqrt_n_points = math.sqrt(num_points)
        assert sqrt_n_points % 1 == 0, "num_points must be a perfect square"
        sqrt_n_points = int(sqrt_n_points)
        theta = torch.linspace(0, 2 * math.pi, sqrt_n_points, dtype=center.dtype, device=center.device)
        phi = torch.linspace(0, math.pi, sqrt_n_points, dtype=center.dtype, device=center.device)

        # Meshgrid for theta and phi
        theta, phi = torch.meshgrid(theta, phi)

        # Parametric equations of the ellipsoid
        x = a * torch.cos(theta) * torch.sin(phi) + x0
        y = b * torch.sin(theta) * torch.sin(phi) + y0
        z = c * torch.cos(phi) + z0

        ellipsoid = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
        return ellipsoid

    def forward(self, hand_theta, hand_verts_n, hand_verts_r):
        # normalize rotation
        hand_theta_r = hand_theta.clone()
        hand_theta_r[:, :3] = 0.0
        hand_normals_r = hand_verts_r.normals_padded()
        hand_verts_r = hand_verts_r.points_padded()

        # run
        batch_size = hand_theta_r.shape[0]
        hand_verts_r = hand_verts_r.permute(0, 2, 1).contiguous()
        hand_normals_r = hand_normals_r.permute(0, 2, 1).contiguous()
        x_pc = torch.cat([hand_verts_r, hand_normals_r], dim=1)
        x_pc, *_ = self.pointnet(x_pc)
        x = torch.cat([x_pc, hand_theta_r.view(batch_size, 1, -1).repeat(1, x_pc.shape[1], 1)], dim=2)
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse(self.convfuse(x)))
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = torch.sigmoid(x)
        x = x.squeeze(2)

        # contact xyz, object base
        contact_xyzs_n = []
        object_base_n = []
        hand_verts_r = hand_verts_r.permute(0, 2, 1).contiguous()
        for b in range(batch_size):
            contact_xyz_n = hand_verts_n[b][x[b] >= 0.5]
            if contact_xyz_n.shape[0] == 0:
                contact_xyz_n = hand_verts_n[b][-2:]
            center = torch.mean(contact_xyz_n, dim=0, keepdim=True)
            min_dist = torch.min(torch.norm(contact_xyz_n - center, dim=1))
            obj_base_n = torch.rand(self.opt.object.points, 3, device=contact_xyz_n.device) - 0.5
            obj_base_n = obj_base_n * min_dist  / (torch.norm(obj_base_n, dim=1, keepdim=True) + 1e-6)
            obj_base_n = obj_base_n + center
            object_base_n.append(obj_base_n)
            contact_xyzs_n.append(contact_xyz_n)
        object_base_n = torch.stack(object_base_n, dim=0)
        contact_xyzs_n = Pointclouds(contact_xyzs_n)

        # object base features
        distances, idx_object, _ = knn_points(
            p1=contact_xyzs_n.points_padded(),
            p2=object_base_n,
            lengths1=torch.LongTensor([len(i) for i in contact_xyzs_n.points_list()]).to(contact_xyzs_n.device),
            lengths2=torch.LongTensor([self.opt.object.points] * batch_size).to(contact_xyzs_n.device),
            norm=2,
            K=1,
        )
        object_feat_n = torch.zeros_like(object_base_n[..., 0:1])
        idx_object = idx_object[:, :,0]
        batch_idx = torch.arange(idx_object.shape[0]).view(-1, 1).expand(-1, idx_object.shape[1])
        object_feat_n[batch_idx, idx_object, :] = torch.sigmoid(-distances) * 2.
        object_base_n = torch.cat([object_base_n, object_feat_n], dim=2)

        return (
            x,
            contact_xyzs_n.points_padded(),
            object_base_n,
            hand_theta_r,
        )


class SelectObjectNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        dim_hand_pose = opt.dim_hand_pose
        category_hc = opt.category.hc
        object_hc = opt.object.hc
        fusion_hc = opt.fusion.hc
        # contact
        self.contact_net = ContactNet(opt)

        # encoder
        self.en_pointnet1 = PointNetObject(1, 3)
        self.en_pointnet2 = PointNetObject(object_hc, 11)
        self.en_batchnorm1 = nn.BatchNorm1d(opt.n_class + 3)
        self.en_resnet1 = ResBlock(opt.n_class + dim_hand_pose, category_hc)
        self.en_cross1 = CrossAttention(object_hc * 8, category_hc)
        self.en_resnet2 = ResBlock(
            max(object_hc * 8, category_hc) + opt.n_class + dim_hand_pose, category_hc
        )
        self.en_cross2 = CrossAttention(object_hc * 8, category_hc)

        # fusion
        self.fusion_resnet = ResBlock(
            max(object_hc * 8, category_hc) + object_hc * 8, fusion_hc
        )
        self.fusion_mu = nn.Linear(fusion_hc, opt.dim_latent)
        self.fusion_var = nn.Linear(fusion_hc, opt.dim_latent)

        # decoder
        self.de_mlp1 = nn.Linear(opt.dim_latent + dim_hand_pose - 3, object_hc * 2)
        self.de_batchnorm1 = nn.BatchNorm1d(object_hc * 2)
        self.de_drop1 = nn.Dropout(0.1)
        self.de_conv0 = nn.Conv1d(object_hc * 2, opt.object.points, 1)
        self.de_batchnorm0 = nn.BatchNorm1d(opt.object.points)
        self.de_pointnet0 = PointNetObject(object_hc, 5)
        self.de_pointnet1 = PointNetPropagation(
            in_channel=object_hc * 8 + object_hc * 4, mlp=[object_hc * 8, object_hc * 4]
        )
        self.de_pointnet2 = PointNetPropagation(
            in_channel=object_hc * 4 + object_hc * 2, mlp=[object_hc * 4, object_hc * 2]
        )
        self.de_pointnet3 = PointNetPropagation(
            in_channel=object_hc * 2 + 3,
            mlp=[object_hc * 2, object_hc * 2],
        )
        self.de_conv1 = nn.Conv1d(object_hc * 2, object_hc, 1)
        self.de_batchnorm2 = nn.BatchNorm1d(object_hc)
        self.de_drop2 = nn.Dropout(0.1)
        self.de_conv2 = nn.Conv1d(object_hc, 3, 1)
        self.de_resnet1 = ResBlock(opt.dim_latent + dim_hand_pose - 3, category_hc)
        self.de_cross1 = CrossAttention(object_hc * 8, category_hc)
        self.de_maxpool = nn.MaxPool1d(object_hc * 4)
        self.de_cross2 = CrossAttention(
            object_hc * 2,
            max(category_hc, object_hc * 8) + dim_hand_pose + opt.dim_latent,
        )
        self.de_resnet2 = ResBlock(
            max(category_hc, object_hc * 8) + dim_hand_pose + opt.dim_latent,
            category_hc,
        )
        self.de_mlp2 = nn.Linear(category_hc, opt.n_class)
        self.de_softmax = nn.Softmax()

        return

    def enc(self, class_vec, hand_theta, contact_xyz, object_pcs, object_normals):
        batch_size = class_vec.shape[0]
        object_pcs = object_pcs.points_padded()
        object_pcs = object_pcs.permute(0, 2, 1).contiguous()

        # object pc
        hand_theta_stack = hand_theta.reshape(batch_size, -1, 3)
        obj_f0 = torch.cat([object_normals, hand_theta_stack, contact_xyz], 1)
        *_, obj_f0 = self.en_pointnet1(obj_f0.permute(0, 2, 1).contiguous(), None)
        obj_f0 = obj_f0.unsqueeze(2).repeat(1, 1, object_pcs.shape[2])
        *_, obj_x1 = self.en_pointnet2(object_pcs, obj_f0)

        # category
        cate_x0 = torch.cat([class_vec, hand_theta[:, :3]], dim=1).float()
        cate_x0 = self.en_batchnorm1(cate_x0)
        cate_x0 = torch.cat([cate_x0, hand_theta[:, 3:]], dim=1)
        cate_x1 = self.en_resnet1(cate_x0)
        cate_x1 = self.en_cross1(obj_x1, cate_x1)
        cate_x1 = self.en_resnet2(torch.cat([cate_x1, cate_x0], dim=1))

        # fusion
        cate_x1 = self.en_cross2(obj_x1, cate_x1)
        z = self.fusion_resnet(torch.cat([cate_x1, obj_x1], dim=1))
        z = torch.distributions.normal.Normal(
            self.fusion_mu(z), F.softplus(self.fusion_var(z))
        )
        return z

    def dec(self, z, object_base, hand_theta):
        object_base = object_base.permute(0, 2, 1).contiguous()

        # object pc
        cond = torch.cat([z, hand_theta[:, 3:]], dim=1)
        cond = self.de_mlp1(cond)
        cond = F.relu(self.de_batchnorm1(cond))
        cond = self.de_drop1(cond)
        cond = self.de_conv0(cond.unsqueeze(2))
        cond = F.relu(self.de_batchnorm0(cond))
        cond = cond.permute(0, 2, 1).contiguous()
        cond = torch.cat([object_base[:, 3:4, :], cond], dim=1)
        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points = self.de_pointnet0(
            object_base[:, :3], cond
        )
        l3_points = l3_points.unsqueeze(2)
        l2_points = self.de_pointnet1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.de_pointnet2(l1_xyz, l2_xyz, l1_points, l2_points)
        object_pred = self.de_pointnet3(
            object_base[:, :3], l1_xyz, object_base[:, :3], l1_points
        )
        object_pred = self.de_conv1(object_pred)
        object_pred = F.relu(self.de_batchnorm2(object_pred))
        object_pred = self.de_drop2(object_pred)
        object_pred = self.de_conv2(object_pred)
        object_pred = object_base[:, :3, :] + object_pred

        # category
        cate_x = torch.cat([z, hand_theta[:, 3:]], dim=1)
        category_pred = self.de_resnet1(cate_x)
        category_pred = self.de_cross1(l3_points.squeeze(2), category_pred)
        category_pred = torch.cat([category_pred, cate_x, hand_theta[:, :3]], dim=1)
        l1_feat = self.de_maxpool(l1_points).squeeze(2)
        category_pred = self.de_cross2(l1_feat, category_pred)
        category_pred = self.de_resnet2(category_pred)
        category_pred = self.de_mlp2(category_pred)
        category_pred = self.de_softmax(category_pred)

        return category_pred, object_pred

