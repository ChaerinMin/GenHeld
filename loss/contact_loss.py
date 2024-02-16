import logging
import os

import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
import torch
from torch.nn import Module
from collections import namedtuple

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
    def __init__(self, cfg, opt):
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

        # plot contact
        self.penetr_hand = []
        self.hand_part = []
        self.obj_part = []

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
        for b in range(batch_size):
            obj_triangles = obj_verts_pt.padded[
                b, obj_faces.verts_idx.padded[b, : obj_faces.verts_idx.split_sizes[b]]
            ]
            ext_hand = contact_utils.batch_mesh_contains_points(
                hand_verts_pt.detach()[b : b + 1], obj_triangles.detach()[None, ...]
            )
            exterior_hand.append(ext_hand)
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
        missed_mask = below_dist & exterior_hand
        if self.contact_zones == "tips":
            tip_idxs = [745, 317, 444, 556, 673]
            tips = torch.zeros_like(missed_mask)
            tips[:, tip_idxs] = 1
            missed_mask = missed_mask & tips
        elif self.contact_zones == "zones":
            # For each batch keep the closest point from the contact zone
            _, contact_zones = contact_utils.load_contacts("assets/contact_zones.pkl")
            contact_matching = torch.zeros_like(missed_mask)
            for _, zone_idxs in contact_zones.items():
                min_zone_vals, min_zone_idxs = minho[:, zone_idxs].min(1)
                cont_idxs = minoh.new(zone_idxs)[min_zone_idxs]
                contact_matching[
                    [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
                ] = 1
            missed_mask = missed_mask & contact_matching
        elif self.contact_zones in ["gen", "all"]:
            missed_mask = missed_mask
        else:
            raise ValueError(
                "contact_zones {} not in [tips|zones|all]".format(contact_zones)
            )

        # compute losses
        missed_loss, missed_losses = masked_mean_loss(contact_vals, missed_mask)  # attraction loss
        penetr_h_loss, penetr_h_losses = masked_mean_loss(collision_vals, penetr_h_mask)  # repulsion loss
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

        return missed_loss, penetr_loss, contact_info, metrics

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
