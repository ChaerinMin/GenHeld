import logging
from typing import Dict, NamedTuple, Union

import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures.meshes import Meshes
from torch import Tensor

from dataset import PaddedTensor

logger = logging.getLogger(__name__)


def merge_ho(
    is_textured: bool,
    h_verts: Tensor,
    h_faces: NamedTuple,
    h_aux: Dict,
    o_verts: PaddedTensor,
    o_faces: NamedTuple,
    o_aux: Dict,
):
    """
    Hands are standard batching.
    Objects are heterogeneous batching.
    Vertex normals should be forgotten and recalculated, because vertex positions could have been modified
    """
    batch_size = h_verts.shape[0]

    verts = torch.cat([h_verts, o_verts.padded], dim=1)
    f_v = torch.cat(
        [h_faces.verts_idx, h_verts.shape[1] + o_faces.verts_idx.padded], dim=1
    )
    for b in range(batch_size):
        f_v[b, h_faces.verts_idx.shape[1] + o_faces.verts_idx.split_sizes[b] :] = -1

    if is_textured:
        # texture images
        if len(h_aux["texture_images"]) > 1:
            logger.error(
                f"Only support one texture image for each mesh, got {len(h_aux['texture_images'])}"
            )
            raise ValueError
        h_material_name = list(h_aux["texture_images"].keys())[0]
        if len(o_aux["texture_images"]) > 1:
            logger.error(
                f"Only support one texture image for each mesh, got {len(o_aux['texture_images'])}"
            )
            raise ValueError
        o_material_name = list(o_aux["texture_images"].keys())[0]
        _, h_H, h_W, h_C = h_aux["texture_images"][h_material_name].shape
        _, o_H, o_W, o_C = o_aux["texture_images"][o_material_name].shape
        assert h_C == o_C == 3
        max_H = max(h_H, o_H)
        texture_images = torch.ones(
            (batch_size, max_H, h_W + o_W, h_C), dtype=torch.float32
        )
        texture_images[:, :h_H, :h_W] = h_aux["texture_images"][h_material_name]
        texture_images[:, :o_H, h_W:] = o_aux["texture_images"][o_material_name]

        # vt
        h_vt, o_vt = (
            h_aux["verts_uvs"].clone(),
            o_aux["verts_uvs"].padded.clone(),
        )
        h_vt[..., 0] = h_vt[..., 0] * h_W / (h_W + o_W)
        h_vt[..., 1] = (h_vt[..., 1] * h_H + max_H - h_H) / max_H
        o_vt[..., 0] = (h_W + o_vt[..., 0] * o_W) / (h_W + o_W)
        o_vt[..., 1] = (o_vt[..., 1] * o_H + max_H - o_H) / max_H
        for b in range(batch_size):
            o_vt[b, o_aux["verts_uvs"].split_sizes[b] :] = 0.0
        vt = torch.cat([h_vt, o_vt], dim=1)
        f_vt = torch.cat(
            [
                h_faces.textures_idx,
                o_faces.textures_idx.padded + h_aux["verts_uvs"].shape[1],
            ],
            dim=1,
        )
        for b in range(batch_size):
            f_vt[
                b, h_faces.textures_idx.shape[1] + o_faces.textures_idx.split_sizes[b] :
            ] = -1

        # pytorch3d Textures
        textures = TexturesUV(
            maps=texture_images.to(verts.device),
            faces_uvs=f_vt,
            verts_uvs=vt,
        )

    else:
        textures = None

    meshes = Meshes(
        verts=verts,
        faces=f_v,
        textures=textures,
    )

    # recalculate normals
    meshes.verts_normals_packed()

    return meshes


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
        logger.error(f"verts should be torch.Tensor or PaddedTensor, got {type(verts)}")
        raise ValueError

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
        logger.error(f"verts should be torch.Tensor or PaddedTensor, got {type(verts)}")
        raise ValueError

    return vertices, center, max_norm
