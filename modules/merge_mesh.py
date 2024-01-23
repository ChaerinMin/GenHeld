import logging
import torch 
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import TexturesUV

logger = logging.getLogger(__name__)


def merge_meshes(is_textured, a_verts, a_faces, a_aux, b_verts, b_faces, b_aux):
    """
    Not support heterogeneous batching. Only support batch size 1.
    Vertex normals should be forgotten and recalculated, because vertex positions could have been modified
    """

    if a_verts.shape[0] > 1:
        logger.error("Only support batch size 1")

    verts = torch.cat([a_verts, b_verts], dim=1)
    f_v = torch.cat([a_faces.verts_idx, b_faces.verts_idx + a_verts.shape[1]], dim=1)

    if is_textured:
        # texture images
        if len(a_aux["texture_images"]) > 1:
            logger.error(
                f"Only support one texture image for each mesh, got {len(a_aux['texture_images'])}"
            )
        if len(b_aux["texture_images"]) > 1:
            logger.error(
                f"Only support one texture image for each mesh, got {len(b_aux['texture_images'])}"
            )
        batch_size, a_H, a_W, a_C = a_aux["texture_images"]["material_0"].shape
        batch_size, b_H, b_W, b_C = b_aux["texture_images"]["material_0"].shape
        assert a_C == b_C == 3
        max_H = max(a_H, b_H)
        texture_images = torch.ones(
            (batch_size, max_H, a_W + b_W, a_C), dtype=torch.float32
        )
        texture_images[:, :a_H, :a_W] = a_aux["texture_images"]["material_0"]
        texture_images[:, :b_H, a_W:] = b_aux["texture_images"]["material_0"]

        # vt
        a_vt, b_vt = a_aux["verts_uvs"].clone(), b_aux["verts_uvs"].clone()
        a_vt[..., 0] = a_vt[..., 0] * a_W / (a_W + b_W)
        a_vt[..., 1] = (a_vt[..., 1] * a_H + max_H - a_H) / max_H
        b_vt[..., 0] = (a_W + b_vt[..., 0] * b_W) / (a_W + b_W)
        b_vt[..., 1] = (b_vt[..., 1] * b_H + max_H - b_H) / max_H
        vt = torch.cat([a_vt, b_vt], dim=1)
        f_vt = torch.cat(
            [a_faces.textures_idx, b_faces.textures_idx + a_aux["verts_uvs"].shape[1]],
            dim=1,
        )

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
