import logging
import os
from collections import namedtuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from torch.nn import Module

from . import contact_utils

logger = logging.getLogger(__name__)


def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
        losses = (mask * dists).sum(1) / mask.sum(1)
    else:
        loss = torch.tensor(0.0, dtype=dists.dtype, device=dists.device)
        losses = torch.zeros(dists.shape[0], dtype=dists.dtype, device=dists.device)
    return loss, losses


class ContactLoss(Module):
    def __init__(self, cfg, opt, device):
        super(ContactLoss, self).__init__()
        self.cfg = cfg
        for k, v in opt.items():
            setattr(self, k, v)

        self.handpart_lookup = [
            [],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [2, 3],
        ]

        # force closure
        self.fc_skew_sym = (
            torch.tensor(
                np.array(
                    [
                        [0, 0, 0, 0, 0, -1, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, -1, 0, 0],
                        [0, -1, 0, 1, 0, 0, 0, 0, 0],
                    ]
                )
            )
            .float()
            .to(device)
        )
        self.eye3 = torch.tensor(np.eye(3).reshape(1, 1, 3, 3)).float().to(device)
        self.eye6 = torch.tensor(np.eye(6).reshape(1, 6, 6)).float().to(device)
        self.eps = torch.tensor(0.01).float().to(device)
        self.relu = torch.nn.ReLU()
        self.right = torch.tensor([[[self.fc_voxel_size, 0.0, 0.0]]]).to(device)
        self.left = torch.tensor([[[-self.fc_voxel_size, 0.0, 0.0]]]).to(device)
        self.top = torch.tensor([[[0.0, self.fc_voxel_size, 0.0]]]).to(device)
        self.bottom = torch.tensor([[[0.0, -self.fc_voxel_size, 0.0]]]).to(device)
        self.front = torch.tensor([[[0.0, 0.0, self.fc_voxel_size]]]).to(device)
        self.back = torch.tensor([[[0.0, 0.0, -self.fc_voxel_size]]]).to(device)

        # plot contact
        self.penetr_hand = []
        self.hand_part = []
        self.obj_part = []

        # plot fc normal
        self.fc_contact = None
        self.fc_h_normals = None
        self.fc_o_normals = None
        self.fc_exteriors = None

        # loss and losses
        self.LossNLosses = namedtuple("LossNLosses", ["loss", "losses"])

        return

    def forward(
        self,
        hand_verts_pt,
        hand_faces,
        obj_verts_pt,
        obj_faces,
        sampled_verts=None,
        contact_object=None,
        partition_object=None,
        fc_contact_ind=None,
    ):
        batch_size = hand_verts_pt.shape[0]

        # exterior mask
        hand_triangles = []
        for b in range(batch_size):
            hand_triangles.append(hand_verts_pt[b, hand_faces.verts_idx[b]])
        hand_triangles = torch.stack(hand_triangles, dim=0)
        exterior_obj = contact_utils.batch_mesh_contains_points(
            obj_verts_pt.padded.detach(), hand_triangles.detach()
        )
        exterior_hand = []
        obj_triang = []
        for b in range(batch_size):
            obj_triangles = obj_verts_pt.padded[
                b, obj_faces.verts_idx.padded[b, : obj_faces.verts_idx.split_sizes[b]]
            ]
            obj_triangles = obj_triangles.detach()[None, ...]
            ext_hand = contact_utils.batch_mesh_contains_points(
                hand_verts_pt.detach()[b : b + 1], obj_triangles
            )
            exterior_hand.append(ext_hand)
            obj_triang.append(obj_triangles)
        exterior_hand = torch.cat(exterior_hand, dim=0)
        penetr_o_mask = ~exterior_obj
        for b in range(batch_size):
            penetr_o_mask[b, obj_verts_pt.split_sizes[b] :] = 0
        penetr_h_mask = ~exterior_hand

        # plot penetration
        self.penetr_obj = []
        self.penetr_hand = []
        for b in range(batch_size):
            self.penetr_obj.append(obj_verts_pt.padded.detach()[b, penetr_o_mask[b]])
            self.penetr_hand.append(hand_verts_pt.detach()[b, penetr_h_mask[b]])

        # min vertex pairs between hand and object
        dists = (
            torch.ones(
                batch_size,
                hand_verts_pt.shape[1],
                max(obj_verts_pt.split_sizes),
                device=hand_verts_pt.device,
                dtype=hand_verts_pt.dtype,
            )
            * 1e10
        )
        for b in range(batch_size):
            obj_size = obj_verts_pt.split_sizes[b]
            dists[b, :, :obj_size] = batch_pairwise_dist(
                hand_verts_pt[b : b + 1], obj_verts_pt.padded[b : b + 1, :obj_size]
            ).squeeze(0)
        minoh, minoh_idxs = torch.min(dists, 1)
        minho, minho_idxs = torch.min(dists, 2)
        results_close = batch_index_select(obj_verts_pt.padded, 1, minho_idxs)
        results_o_close = batch_index_select(hand_verts_pt, 1, minoh_idxs)

        # d (ObMan)
        if self.contact_target == "all":
            anchor_dists = torch.norm(results_close - hand_verts_pt, 2, 2)
            anchor_o_dist = torch.norm(results_o_close - obj_verts_pt.padded, 2, 2)
        elif self.contact_target == "obj":
            anchor_dists = torch.norm(results_close - hand_verts_pt.detach(), 2, 2)
            anchor_o_dist = torch.norm(
                results_o_close.detach() - obj_verts_pt.padded, 2, 2
            )
        elif self.contact_target == "hand":
            anchor_dists = torch.norm(results_close.detach() - hand_verts_pt, 2, 2)
            anchor_o_dist = torch.norm(
                results_o_close - obj_verts_pt.padded.detach(), 2, 2
            )
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(self.contact_target)
            )

        # l of attraction loss (ObMan)
        if self.contact_mode == "dist_sq":
            contact_vals = anchor_dists**2
            below_dist = minho < (self.contact_thresh**2)
        elif self.contact_mode == "dist":
            contact_vals = anchor_dists
            below_dist = minho < self.contact_thresh
        elif self.contact_mode == "dist_tanh":
            contact_vals = self.contact_thresh * torch.tanh(
                anchor_dists / self.contact_thresh
            )
            below_dist = torch.ones_like(minho).byte()
        else:
            raise ValueError(
                "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(
                    self.contact_mode
                )
            )

        # l of attraction loss when partition is given
        if self.contact_zones == "gen":
            assert sampled_verts is not None
            assert contact_object is not None and partition_object is not None
            _, contact_zones = contact_utils.load_contacts(
                "assets/contact_zones.pkl", display=True
            )  # palm, index, middle, ring, pinky, thumb

            # initializations
            contact_vals_part = torch.zeros_like(minho)
            below_part = torch.ones_like(minho).byte()
            self.hand_part = [[] for _ in range(batch_size)]
            self.obj_part = [[] for _ in range(batch_size)]

            for i, zone_idxs in contact_zones.items():
                handpart = self.handpart_lookup[i]
                if len(handpart) == 0:
                    continue
                hand_part = hand_verts_pt[:, zone_idxs]
                for b in range(batch_size):
                    partmask = torch.zeros_like(
                        partition_object.padded[b], dtype=torch.bool
                    )
                    for part in handpart:
                        partmask = torch.logical_or(
                            partmask, (partition_object.padded[b] == part)
                        )
                    obj_part = sampled_verts.padded[b, partmask]
                    contact_part = contact_object.padded[b, partmask]
                    dists_part = batch_pairwise_dist(hand_part, obj_part.unsqueeze(0))
                    minho_part, minho_part_idxs = torch.min(dists_part, 2)
                    close_part = batch_index_select(
                        obj_part.unsqueeze(0), 1, minho_part_idxs
                    )
                    close_weight = batch_index_select(
                        contact_part.unsqueeze(0), 1, minho_part_idxs
                    )
                    assert self.contact_target == "obj", "Not implemented"
                    anchor_part = torch.norm(close_part - hand_part.detach(), 2, 2)
                    anchor_part = anchor_part * close_weight
                    if self.contact_mode == "dist_sq":
                        contact_vals_part[b : b + 1, zone_idxs] = anchor_part**2
                    elif self.contact_mode == "dist":
                        contact_vals_part[b : b + 1, zone_idxs] = anchor_part
                    elif self.contact_mode == "dist_tanh":
                        contact_vals_part[b : b + 1, zone_idxs] = (
                            self.contact_thresh
                            * torch.tanh(anchor_part / self.contact_thresh)
                        )
                    else:
                        raise ValueError(
                            "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(
                                self.contact_mode
                            )
                        )

                    # plot contact
                    self.hand_part[b].append(hand_part[b])
                    self.obj_part[b].append(obj_part.detach())
            contact_vals = contact_vals_part
            below_dist = below_part

        # l of repulsion loss (ObMan)
        if self.collision_mode == "dist_sq":
            collision_vals = anchor_dists**2
            collision_o_vals = anchor_o_dist**2
        elif self.collision_mode == "dist":
            collision_vals = anchor_dists
            collision_o_vals = anchor_o_dist
        elif self.collision_mode == "dist_tanh":
            collision_vals = self.collision_thresh * torch.tanh(
                anchor_dists / self.collision_thresh
            )
            collision_o_vals = self.collision_thresh * torch.tanh(
                anchor_o_dist / self.collision_thresh
            )
        else:
            raise ValueError(
                "collision_mode {} not in "
                "[dist_sq|dist|dist_tanh]".format(self.collision_mode)
            )
        for b in range(batch_size):
            collision_o_vals[b, obj_verts_pt.split_sizes[b] :] = 0

        # C and Ext(Obj) (ObMan)
        missed_mask_original = below_dist & exterior_hand
        if self.contact_zones == "tips":
            tip_idxs = [745, 317, 444, 556, 673]
            tips = torch.zeros_like(missed_mask_original)
            tips[:, tip_idxs] = 1
            missed_mask = missed_mask_original & tips
        elif self.contact_zones == "zones":
            # For each batch keep the closest point from the contact zone
            _, contact_zones = contact_utils.load_contacts("assets/contact_zones.pkl")
            contact_matching = torch.zeros_like(missed_mask_original)
            for _, zone_idxs in contact_zones.items():
                min_zone_vals, min_zone_idxs = minho[:, zone_idxs].min(1)
                cont_idxs = minoh.new(zone_idxs)[min_zone_idxs]
                contact_matching[
                    [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
                ] = 1
            missed_mask = missed_mask_original & contact_matching
        elif self.contact_zones in ["gen", "all"]:
            missed_mask = missed_mask_original
        else:
            raise ValueError(
                "contact_zones {} not in [tips|zones|all]".format(contact_zones)
            )

        # compute losses
        missed_loss, missed_losses = masked_mean_loss(
            contact_vals, missed_mask
        )  # attraction loss
        penetr_h_loss, penetr_h_losses = masked_mean_loss(
            collision_vals, penetr_h_mask
        )  # repulsion loss
        penetr_o_loss, penetr_o_losses = masked_mean_loss(
            collision_o_vals, penetr_o_mask
        )  # repulsion loss
        penetr_loss = penetr_h_loss + penetr_o_loss
        penetr_losses = penetr_h_losses + penetr_o_losses
        if self.contact_sym:
            obj2hand_dists = torch.sqrt(minoh)
            sym_below_dist = minoh < self.contact_thresh
            sym_loss, sym_losses = masked_mean_loss(obj2hand_dists, sym_below_dist)
            missed_loss = missed_loss + sym_loss
            missed_losses = missed_losses + sym_losses

        # force closure
        if fc_contact_ind is not None:
            fc_contact_point = torch.gather(
                hand_verts_pt, 1, torch.tile(fc_contact_ind.unsqueeze(-1), [1, 1, 3])
            )
            self.fc_contact = fc_contact_point

            # normal alignment
            h_contact_normal = Meshes(
                hand_verts_pt, hand_faces.verts_idx
            ).verts_normals_padded()
            h_contact_normal = torch.gather(
                h_contact_normal, 1, torch.tile(fc_contact_ind.unsqueeze(-1), [1, 1, 3])
            )
            o_contact_normal = self.contact_normal(
                obj_verts_pt, obj_faces, obj_triang, fc_contact_point
            )
            normal_alignments = 2.0 - (
                (h_contact_normal * o_contact_normal).sum(-1) + 1
            )
            self.fc_h_normals = h_contact_normal
            self.fc_o_normals = o_contact_normal

            # force closure
            G = self.x_to_G(fc_contact_point)
            fullranks = self.loss_8a(G)
            force_closures = self.loss_8b(o_contact_normal, G)

            # masking
            # fc_exterior = torch.gather(exterior_hand, 1, fc_contact_ind)
            # normal_alignment, normal_alignments = masked_mean_loss(
            #     normal_alignments, fc_exterior
            # )
            # self.fc_exteriors = fc_exterior

            # fc_attr, fc_pene
            fc_attr = torch.gather(contact_vals, 1, fc_contact_ind)
            fc_attr_mask = torch.gather(missed_mask_original, 1, fc_contact_ind)
            fc_attr, fc_attrs = masked_mean_loss(fc_attr, fc_attr_mask)
            fc_pene = torch.gather(collision_vals, 1, fc_contact_ind)
            fc_pene_mask = torch.gather(penetr_h_mask, 1, fc_contact_ind)
            fc_pene, fc_penes = masked_mean_loss(fc_pene, fc_pene_mask)

            fullrank = torch.mean(fullranks)
            force_closure = torch.mean(force_closures)
            normal_alignments = torch.mean(normal_alignments, dim=1)
            normal_alignment = torch.mean(normal_alignments)
            fc_term = fullrank + force_closure + fc_attr + fc_pene * 3 + normal_alignment 
            fc_terms = fullranks + force_closures + fc_attrs + fc_penes * 3 + normal_alignments 
            fc_loss = self.LossNLosses(fc_term, fc_terms)
        else:
            fc_loss = torch.tensor(
                0.0, dtype=missed_loss.dtype, device=missed_loss.device
            )

        # loss and losses
        missed_loss = self.LossNLosses(missed_loss, missed_losses)
        penetr_loss = self.LossNLosses(penetr_loss, penetr_losses)

        # contact_info, metrics
        max_penetr_depth = (
            (anchor_dists.detach() * penetr_h_mask.float()).max(1)[0].mean()
        )
        mean_penetr_depth = (
            (anchor_dists.detach() * penetr_h_mask.float()).mean(1).mean()
        )
        contact_info = {
            "attraction_masks": missed_mask,
            "repulsion_h_masks": penetr_h_mask,
            "contact_points": results_close,
            "min_dists": minho,
            "contact_zones": contact_zones,
        }
        metrics = {
            "max_penetr": max_penetr_depth,
            "mean_penetr": mean_penetr_depth,
        }

        return missed_loss, penetr_loss, fc_loss, contact_info, metrics

    def plot_contact(self, hand_fidxs, object_fidxs, iter):
        batch_size = len(self.penetr_hand)
        num_parts = len(self.handpart_lookup)

        # penetration
        for b in range(batch_size):
            penetr_hand = self.penetr_hand[b]
            penetr_obj = self.penetr_obj[b]
            penetr = torch.cat([penetr_hand, penetr_obj], dim=0)
            if penetr.shape[0] == 0:
                logger.info(
                    f"Hand {hand_fidxs[b]}, object {object_fidxs[b]}, at iter {iter} has no penetration"
                )
                continue
            penetr = o3d.utility.Vector3dVector(penetr.cpu().numpy())
            penetr = o3d.geometry.PointCloud(penetr)
            penetr.paint_uniform_color([0, 0, 0])
            save_path = os.path.join(
                self.cfg.results_dir,
                f"hand_{hand_fidxs[b]}_object_{object_fidxs[b]}_iter_{iter}_penetr_hand.ply",
            )
            saved = o3d.io.write_point_cloud(save_path, penetr)
            if saved:
                logger.info(f"Saved {save_path}")
            else:
                logger.warning(f"Failed to save {save_path}")

        # color map
        def generate_colors(n, saturation):
            """Generate n colors in HSV space and convert them to RGB."""
            colors = []
            for i in range(n):
                hue = i / n
                color = mcolors.hsv_to_rgb([hue, saturation, 1])
                colors.append(color)
            return colors

        # ContactGen hand & object
        if len(self.hand_part) == 0 or len(self.obj_part) == 0:
            logger.warning(f"No result from ContactGen")
            return

        color_hand = generate_colors(num_parts, saturation=1)
        color_obj = generate_colors(num_parts, saturation=0.5)
        for b in range(batch_size):
            if len(self.hand_part[b]) == 0 or len(self.obj_part[b]) == 0:
                logger.error(
                    f"ContactGen was used, but hand {hand_fidxs[b]}, object {object_fidxs[b]}, at iter {iter} has no ContactGen result"
                )
                raise ValueError
                continue

            # assign colors
            coordinates = []
            colors = []
            for i, hand_finger in enumerate(self.hand_part[b]):
                coordinates.append(hand_finger.cpu().numpy())
                colors.append(np.array([color_hand[i]] * hand_finger.shape[0]))
            for i, obj_finger in enumerate(self.obj_part[b]):
                coordinates.append(obj_finger.cpu().numpy())
                colors.append(np.array([color_obj[i]] * obj_finger.shape[0]))
            coordinates = np.concatenate(coordinates, axis=0)
            colors = np.concatenate(colors, axis=0)

            # to o3d
            coordinates = o3d.utility.Vector3dVector(coordinates)
            colors = o3d.utility.Vector3dVector(colors)
            contact_pc = o3d.geometry.PointCloud()
            contact_pc.points = coordinates
            contact_pc.colors = colors

            # save
            save_path = os.path.join(
                self.cfg.results_dir,
                f"hand_{hand_fidxs[b]}_object_{object_fidxs[b]}_iter_{iter}_contact.ply",
            )
            saved = o3d.io.write_point_cloud(save_path, contact_pc)
            if saved:
                logger.info(f"Saved {save_path}")
            else:
                logger.warning(f"Failed to save {save_path}")
        return

    def plot_fc_normal(self, hand_fidxs, object_fidxs, iter, denorm_center, denorm_scale, save_mask, is_period):
        batch_size = self.fc_contact.shape[0]
        fc_contact = self.fc_contact * denorm_scale[:, None, None] + denorm_center
        for b in range(batch_size):
            if save_mask[b]:
                tag = "best"
            elif is_period:
                tag = f"iter_{iter:05d}"
            else:
                continue
            filename = os.path.join(
                self.cfg.results_dir,
                f"hand_{hand_fidxs[b]}_object_{object_fidxs[b]}_{tag}_fc_normal.ply",
            )
            # num_exteriors = self.fc_exteriors[b].sum()
            with open(filename, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {self.fc_n_contacts*3}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write(f"element edge {self.fc_n_contacts*2}\n")
                f.write("property int vertex1\n")
                f.write("property int vertex2\n")
                f.write("end_header\n")
                # vertex
                for i in range(fc_contact.shape[1]):
                    # if self.fc_exteriors[b, i] == 0:
                    #     continue
                    start = fc_contact[b, i]
                    f.write(f"{start[0]} {start[1]} {start[2]} 0 0 0\n")
                    end = fc_contact[b, i] + self.fc_h_normals[b, i] * 0.1
                    f.write(f"{end[0]} {end[1]} {end[2]} 255 0 0\n")
                    end = fc_contact[b, i] + self.fc_o_normals[b, i] * 0.1
                    f.write(f"{end[0]} {end[1]} {end[2]} 0 255 0\n")
                # edge
                for i in range(fc_contact.shape[1]):
                    # if self.fc_exteriors[b, i] == 0:
                    #     continue
                    f.write(f"{i*3} {i*3+1}\n")
                    f.write(f"{i*3} {i*3+2}\n")
            logger.info(f"Saved {filename}")
        return

    def x_to_G(self, x):
        """
        x: B x N x 3
        G: B x 6 x 3N
        """
        B = x.shape[0]
        N = x.shape[1]
        xi_cross = (
            torch.matmul(x, self.fc_skew_sym)
            .reshape([B, N, 3, 3])
            .transpose(1, 2)
            .reshape([B, 3, 3 * N])
        )
        I = self.eye3.repeat([B, N, 1, 1]).transpose(1, 2).reshape([B, 3, 3 * N])
        G = torch.stack([I, xi_cross], 1).reshape([B, 6, 3 * N])
        return G

    def loss_8a(self, G):
        """
        G: B x 6 x 3N
        Return: B
        """
        Gt = G.transpose(1, 2)
        temp = self.eps * self.eye6
        temp = torch.matmul(G, Gt) - temp
        eigval = torch.linalg.eigh(temp.cpu(), UPLO="U")[0].to(G.device)
        rnev = self.relu(-eigval)
        results = torch.sum(rnev * rnev, 1)
        return results

    def loss_8b(self, f, G):
        """
        G: B x 6 x 3N
        f: B x N x 3
        """
        B = f.shape[0]
        N = f.shape[1]
        results = torch.matmul(G, f.reshape(B, 3 * N, 1))
        assert len(results.shape) == 3
        results = results * results
        results = self.relu(torch.sum(results, (1, 2)))
        return results

    def contact_normal(self, obj_verts, obj_faces, obj_triang, contact_point):
        batch_size = obj_verts.padded.shape[0]
        meshes = Meshes(
            verts=[
                obj_verts.padded[b, : obj_verts.split_sizes[b]]
                for b in range(batch_size)
            ],
            faces=[
                obj_faces.verts_idx.padded[b, : obj_faces.verts_idx.split_sizes[b]]
                for b in range(batch_size)
            ],
        )
        pcls = torch.cat(
            [
                contact_point,
                contact_point + self.right,
                contact_point + self.left,
                contact_point + self.top,
                contact_point + self.bottom,
                contact_point + self.front,
                contact_point + self.back,
            ],
            dim=1,
        )
        pcls = Pointclouds(pcls)

        # distance
        points = pcls.points_packed()
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
        distance = point_face_distance(
            points, points_first_idx, tris, tris_first_idx, max_points
        )
        distance = distance.view(batch_size, -1)

        # signed distance
        exterior = []
        for b in range(batch_size):
            ext = contact_utils.batch_mesh_contains_points(
                pcls.points_padded()[b].unsqueeze(0), obj_triang[b]
            )
            exterior.append(ext)
        exterior = torch.cat(exterior, dim=0)
        sign = exterior.to(torch.int) * 2 - 1  # 1 or -1
        distance = distance * sign

        # normal
        distance = distance.view(batch_size, 7, -1).permute(0, 2, 1).contiguous()
        diff_x = distance[:, :, 1] - distance[:, :, 2]
        diff_y = distance[:, :, 3] - distance[:, :, 4]
        diff_z = distance[:, :, 5] - distance[:, :, 6]
        normal = torch.stack([diff_x, diff_y, diff_z], dim=-1)
        normal = -normal / torch.norm(normal, dim=-1, keepdim=True)

        return normal
