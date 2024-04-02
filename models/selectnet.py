import logging

import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from torch import nn
from torch.nn import functional as F

from models.pointnet import PointNetSetAbstraction, STNkd

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


class ConvResBlock(nn.Module):
    def __init__(self, Fin, Fout, dim_hidden=256):

        super(ConvResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.conv1 = nn.Conv1d(Fin, dim_hidden, 1)
        self.bn1 = nn.BatchNorm1d(dim_hidden)

        self.conv2 = nn.Conv1d(dim_hidden, Fout, 1)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.conv3 = nn.Conv1d(Fin, Fout, 1)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.conv3(x))

        Xout = self.conv1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.conv2(Xout)
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
    def __init__(self, opt, dim_pos_enc):
        super().__init__()
        self.opt = opt
        self.pointnet = PointNetContact(
            global_feat=False, feature_transform=False, channel=6
        )
        self.convfuse = nn.Conv1d(778 + dim_pos_enc, 778, 1)
        self.bnfuse = nn.BatchNorm1d(778)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 1, 1)
        return

    def forward(self, joints_encoded, hand_verts_r):
        hand_normals_r = hand_verts_r.normals_padded()
        hand_verts_r = hand_verts_r.points_padded()

        # run
        batch_size = joints_encoded.shape[0]
        hand_verts_r = hand_verts_r.permute(0, 2, 1).contiguous()
        hand_normals_r = hand_normals_r.permute(0, 2, 1).contiguous()
        x_pc = torch.cat([hand_verts_r, hand_normals_r], dim=1)
        x_pc, *_ = self.pointnet(x_pc)
        x = torch.cat(
            [x_pc, joints_encoded.view(batch_size, 1, -1).repeat(1, x_pc.shape[1], 1)],
            dim=2,
        )
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

        # contact xyz
        contact_xyzs_r = torch.zeros(batch_size, 778, 3, device=hand_verts_r.device)
        hand_verts_r = hand_verts_r.permute(0, 2, 1).contiguous()
        num_contacts = []
        for b in range(batch_size):
            n_contacts = int(torch.sum(x[b] >= 0.5))
            contact_xyzs_r[b, :n_contacts] = hand_verts_r[b][x[b] >= 0.5]
            num_contacts.append(n_contacts)
        contact_xyzs_r = Pointclouds(contact_xyzs_r)
        contact_xyzs_r._num_points_per_cloud = torch.tensor(
            num_contacts, device=contact_xyzs_r.device
        )
        contact_xyzs_r.points_list()

        return (
            x,
            contact_xyzs_r,
        )


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=4):
        super().__init__()
        self.out_dim_ = (2 * num_freqs * 3) + 3
        self.coeffs_ = 2.0 ** (np.arange(num_freqs))

    def forward(self, x):
        outputs = [x]
        for i, coeff in enumerate(self.coeffs_):
            outputs.append(torch.sin(x * coeff))
            outputs.append(torch.cos(x * coeff))
        return torch.cat(outputs, -1)

    @property
    def out_dim(self):
        return self.out_dim_


class SelectObjectNet(nn.Module):
    def __init__(self, opt, device_manual):
        super().__init__()
        self.opt = opt
        assert opt.n_obj_points > 2048
        # positional encoding
        self.pos_enc = PositionalEncoding()
        dim_pos_enc = opt.dim_joints * self.pos_enc.out_dim 

        # contact
        self.contact_net = ContactNet(opt, dim_pos_enc=dim_pos_enc)

        # encoder
        self.en_conv1 = nn.Conv1d(int(dim_pos_enc / 3) + 778, 1024, 1, bias=False)
        self.en_upconv1 = nn.ConvTranspose1d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        )  # 1024 -> 2048
        self.en_conv2 = nn.Conv1d(2048, opt.n_obj_points, 1)
        self.en_pointnet1 = PointNetObject(int(512 / 8), 9)

        self.en_resnet1 = ConvResBlock(1, 64)
        self.en_resnet2 = ConvResBlock(64, 128)
        self.en_resnet3 = ConvResBlock(128, 256)
        self.en_resnet4 = ConvResBlock(256, 512)
        self.en_maxpool = nn.MaxPool1d(opt.dim_input + dim_pos_enc)

        # fusion
        self.en_cross1 = CrossAttention(512, 512)
        self.fusion_resnet = ResBlock(512, 512)
        self.fusion_mu = nn.Linear(512, 512)
        self.fusion_var = nn.Linear(512, 512)

        # decoder
        self.de_conv0 = nn.Conv1d(3 + 512, 512, 3, padding=1)
        self.de_batchnorm1 = nn.BatchNorm1d(512)
        self.de_drop1 = nn.Dropout(0.1)
        self.de_conv1 = nn.Conv1d(512, 256, 1)
        self.de_batchnorm2 = nn.BatchNorm1d(256)
        self.de_drop2 = nn.Dropout(0.1)
        self.de_conv2 = nn.Conv1d(256, 128, 3, padding=1)
        self.de_batchnorm3 = nn.BatchNorm1d(128)
        self.de_drop3 = nn.Dropout(0.1)
        self.de_conv3 = nn.Conv1d(128, 64, 3, padding=1)
        self.de_batchnorm4 = nn.BatchNorm1d(64)
        self.de_drop4 = nn.Dropout(0.1)
        self.de_conv4 = nn.Conv1d(64, 3, 1)

        self.de_resnet1 = ResBlock(512 + dim_pos_enc, 256)
        self.de_maxpool = nn.MaxPool1d(opt.n_obj_points)
        self.de_cross1 = CrossAttention(256, 256)
        self.de_resnet2 = ResBlock(256, 128)
        self.de_cross2 = CrossAttention(128, 128)
        self.de_resnet3 = ResBlock(128, 64)
        self.de_mlp2 = nn.Linear(64, opt.dim_output)
        if opt.output == "category":
            self.de_final = nn.Softmax()
        elif opt.output == "shapecode":
            self.de_final = nn.ReLU()
            self.add_final = torch.tensor([0., 0., 0., 1., 1., 1.], device=device_manual)
            assert self.opt.dim_output == 6 and self.opt.dim_input == 3
        else:
            logger.error(f"Unknown output type: {opt.output}")
            raise ValueError(f"Unknown output type: {opt.output}")

        return

    def object_upsample(self, joints_encoded, contact_xyz_n):
        batch_size = joints_encoded.shape[0]
        joints_encoded = joints_encoded.reshape(batch_size, -1, 3)
        obj_x = torch.cat([joints_encoded, contact_xyz_n.points_padded()], 1)
        obj_x = self.en_conv1(obj_x)
        obj_x = obj_x.permute(0, 2, 1).contiguous()
        obj_x = self.en_upconv1(obj_x)
        obj_x = F.relu(obj_x)
        obj_x = obj_x.permute(0, 2, 1).contiguous()
        obj_x = self.en_conv2(obj_x)
        obj_x = F.relu(obj_x)
        obj_x = obj_x.permute(0, 2, 1).contiguous()
        return obj_x

    def enc(self, inputs, joints_encoded, contact_xyz_r, object_pcs, object_normals):
        object_pcs = object_pcs.points_padded()
        object_pcs = object_pcs.permute(0, 2, 1).contiguous()
        object_normals = object_normals.permute(0, 2, 1).contiguous()
        
        # object pc
        obj_x = self.object_upsample(joints_encoded, contact_xyz_r)
        obj_x = torch.cat([obj_x, object_normals], dim=1)
        *_, obj_x = self.en_pointnet1(object_pcs, obj_x)
        
        # category
        input_x = torch.cat(
            [inputs.unsqueeze(1), joints_encoded.unsqueeze(1)], dim=2
        ).float()
        input_x = self.en_resnet1(input_x)
        input_x = self.en_resnet2(input_x)
        input_x = self.en_resnet3(input_x)
        input_x = self.en_resnet4(input_x)
        input_x = self.en_maxpool(input_x).squeeze(2)

        # fusion
        z = self.en_cross1(obj_x, input_x)
        z = self.fusion_resnet(z)
        z = torch.distributions.normal.Normal(
            self.fusion_mu(z), F.softplus(self.fusion_var(z))
        )
        return z

    def dec(self, z, joints_encoded, contact_xyz_r):
        # object pc
        obj_x = self.object_upsample(joints_encoded, contact_xyz_r)
        z_stack = z.unsqueeze(2).repeat(1, 1, obj_x.shape[2])
        obj_x = torch.cat([z_stack, obj_x], dim=1)
        obj_x = self.de_conv0(obj_x)
        obj_x = F.relu(self.de_batchnorm1(obj_x))
        obj_x = self.de_drop1(obj_x)
        obj_x = self.de_conv1(obj_x)
        obj_x = F.relu(self.de_batchnorm2(obj_x))
        obj_x1 = self.de_drop2(obj_x)
        obj_x1 = self.de_conv2(obj_x1)
        obj_x1 = F.relu(self.de_batchnorm3(obj_x1))
        object_pred = self.de_drop3(obj_x1)
        object_pred = self.de_conv3(object_pred)
        object_pred = F.relu(self.de_batchnorm4(object_pred))
        object_pred = self.de_drop4(object_pred)
        object_pred = self.de_conv4(object_pred)
        object_pred = object_pred.permute(0, 2, 1).contiguous()

        # category
        output_pred = torch.cat([z, joints_encoded], dim=1)
        output_pred = self.de_resnet1(output_pred)
        output_pred = self.de_resnet2(output_pred)
        output_pred = self.de_resnet3(output_pred)
        output_pred = self.de_mlp2(output_pred)
        output_pred = self.de_final(output_pred)
        if self.opt.output == "shapecode":
            output_pred = output_pred + self.add_final.unsqueeze(0)

        return output_pred, object_pred
