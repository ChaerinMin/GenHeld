import torch
from . import contact_utils
import numpy as np


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
    else:
        loss = torch.Tensor([0]).cuda()
    return loss


def contact_loss(
    hand_verts_pt,
    hand_faces,
    obj_verts_pt,
    obj_faces,
    contact_thresh=5,
    contact_mode="dist_tanh",
    collision_thresh=10,
    collision_mode="dist_tanh",
    contact_target="obj",
    contact_sym=False,
    contact_zones="gen",
    sampled_verts=None,
    contact_object=None,
    partition_object=None,
):
    batch_size = hand_verts_pt.shape[0]

    # exterior mask
    obj_triangles = []
    for b in range(batch_size):
        obj_triangles.append(obj_verts_pt[b, obj_faces[b]])
    obj_triangles = torch.stack(obj_triangles, dim=0)
    exterior = contact_utils.batch_mesh_contains_points(
        hand_verts_pt.detach(), obj_triangles.detach()
    )
    penetr_mask = ~exterior

    # min vertex pairs between hand and object
    dists = batch_pairwise_dist(hand_verts_pt, obj_verts_pt)
    minoh, minoh_idxs = torch.min(dists, 1)
    minho, minho_idxs = torch.min(dists, 2)
    results_close = batch_index_select(obj_verts_pt, 1, minho_idxs)

    # d (ObMan)
    if contact_target == "all":
        anchor_dists = torch.norm(results_close - hand_verts_pt, 2, 2)
    elif contact_target == "obj":
        anchor_dists = torch.norm(results_close - hand_verts_pt.detach(), 2, 2)
    elif contact_target == "hand":
        anchor_dists = torch.norm(results_close.detach() - hand_verts_pt, 2, 2)
    else:
        raise ValueError(
            "contact_target {} not in [all|obj|hand]".format(contact_target)
        )

    # l of attraction loss (ObMan)
    if contact_mode == "dist_sq":
        if contact_target == "all":
            contact_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            contact_vals = ((results_close - hand_verts_pt.detach()) ** 2).sum(2)
        elif contact_target == "hand":
            contact_vals = ((results_close.detach() - hand_verts_pt) ** 2).sum(2)
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(contact_target)
            )
        below_dist = minho < (contact_thresh**2)
    elif contact_mode == "dist":
        contact_vals = anchor_dists
        below_dist = minho < contact_thresh
    elif contact_mode == "dist_tanh":
        contact_vals = contact_thresh * torch.tanh(anchor_dists / contact_thresh)
        below_dist = torch.ones_like(minho).byte()
    else:
        raise ValueError(
            "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(contact_mode)
        )

    # l of attraction loss when partition is given
    if contact_zones == "gen":
        assert sampled_verts is not None
        assert contact_object is not None and partition_object is not None
        _, contact_zone = contact_utils.load_contacts(
            "assets/contact_zones.pkl", display=True
        )  # palm, index, middle, ring, pinky, thumb
        handpart_lookup = [[], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [2, 3]]
        contact_vals_part = torch.zeros_like(minho)
        below_part = torch.ones_like(minho).byte()
        for i, zone_idxs in contact_zone.items():
            handpart = handpart_lookup[i]
            if len(handpart) == 0:
                continue
            hand_part = hand_verts_pt[:, zone_idxs]
            for b in range(batch_size):
                partmask = torch.zeros_like(partition_object[b], dtype=torch.bool)
                for part in handpart:
                    partmask = torch.logical_or(partmask, (partition_object[b] == part))
                obj_part = sampled_verts[b, partmask]
                contact_part = contact_object[b, partmask]
                dists_part = batch_pairwise_dist(hand_part, obj_part.unsqueeze(0))
                minho_part, minho_part_idxs = torch.min(dists_part, 2)
                close_part = batch_index_select(
                    obj_part.unsqueeze(0), 1, minho_part_idxs
                )
                close_weight = batch_index_select(
                    contact_part.unsqueeze(0), 1, minho_part_idxs
                ) 
                assert contact_target == "obj", "Not implemented"
                anchor_part = torch.norm(close_part - hand_part.detach(), 2, 2)
                anchor_part = anchor_part * close_weight 
                assert contact_mode == "dist_tanh", "Not implemented"
                contact_vals_part[b : b + 1, zone_idxs] = contact_thresh * torch.tanh(
                    anchor_part / contact_thresh
                )
        contact_vals = contact_vals_part
        below_dist = below_part

    # l of repulsion loss (ObMan)
    if collision_mode == "dist_sq":
        if contact_target == "all":
            collision_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            collision_vals = ((results_close - hand_verts_pt.detach()) ** 2).sum(2)
        elif contact_target == "hand":
            collision_vals = ((results_close.detach() - hand_verts_pt) ** 2).sum(2)
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(contact_target)
            )
    elif collision_mode == "dist":
        collision_vals = anchor_dists
    elif collision_mode == "dist_tanh":
        collision_vals = collision_thresh * torch.tanh(anchor_dists / collision_thresh)
    else:
        raise ValueError(
            "collision_mode {} not in "
            "[dist_sq|dist|dist_tanh]".format(collision_mode)
        )

    # C and Ext(Obj) (ObMan)
    missed_mask = below_dist & exterior
    if contact_zones == "tips":
        tip_idxs = [745, 317, 444, 556, 673]
        tips = torch.zeros_like(missed_mask)
        tips[:, tip_idxs] = 1
        missed_mask = missed_mask & tips
    elif contact_zones in ["zones", "gen"]:
        _, contact_zones = contact_utils.load_contacts("assets/contact_zones.pkl")
        contact_matching = torch.zeros_like(missed_mask)
        for _, zone_idxs in contact_zones.items():
            min_zone_vals, min_zone_idxs = minho[:, zone_idxs].min(1)
            cont_idxs = minoh.new(zone_idxs)[min_zone_idxs]
            # For each batch keep the closest point from the contact zone
            contact_matching[
                [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
            ] = 1
        missed_mask = missed_mask & contact_matching
    elif contact_zones == "all":
        missed_mask = missed_mask
    else:
        raise ValueError(
            "contact_zones {} not in [tips|zones|all]".format(contact_zones)
        )

    # compute losses
    missed_loss = masked_mean_loss(contact_vals, missed_mask)  # attraction loss
    penetr_loss = masked_mean_loss(collision_vals, penetr_mask)  # repulsion loss
    if contact_sym:
        obj2hand_dists = torch.sqrt(minoh)
        sym_below_dist = minoh < contact_thresh
        sym_loss = masked_mean_loss(obj2hand_dists, sym_below_dist)
        missed_loss = missed_loss + sym_loss

    # contact_info, metrics
    max_penetr_depth = (anchor_dists.detach() * penetr_mask.float()).max(1)[0].mean()
    mean_penetr_depth = (anchor_dists.detach() * penetr_mask.float()).mean(1).mean()
    contact_info = {
        "attraction_masks": missed_mask,
        "repulsion_masks": penetr_mask,
        "contact_points": results_close,
        "min_dists": minho,
    }
    metrics = {
        "max_penetr": max_penetr_depth,
        "mean_penetr": mean_penetr_depth,
    }

    return missed_loss, penetr_loss, contact_info, metrics
