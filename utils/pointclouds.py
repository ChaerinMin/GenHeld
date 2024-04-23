import torch 
from pytorch3d.ops.knn import knn_points


def get_NN(src_xyz, trg_xyz, k=1):
    """
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    """
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],),
        src_xyz.shape[1],
        dtype=torch.int64,
        device=src_xyz.device,
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],),
        trg_xyz.shape[1],
        dtype=torch.int64,
        device=trg_xyz.device,
    )
    src_nn = knn_points(
        src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k
    )  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx