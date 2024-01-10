import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import (
    TexturesUV,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    Materials,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.lighting import PointLights, DirectionalLights
import matplotlib.pyplot as plt
import logging

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


def blend_images(foreground, background, use_alpha=True, blend_type="alpha_blending"):
    # make mask
    if not use_alpha:
        logger.error("Only support use_alpha=True")
    alpha = foreground[..., 3]
    mask = alpha > (255 / 2.0)
    mask = mask * np.array(255, dtype=np.uint8)

    if (
        foreground.shape[0] != background.shape[0]
        or foreground.shape[1] != background.shape[1]
    ):
        logger.error("Only support foreground and background have the same shape")

    if blend_type == "poisson":
        center = (foreground.shape[0] // 2, foreground.shape[1] // 2)
        blended_image = cv2.seamlessClone(
            foreground[..., :3], background, mask, p=center, flags=cv2.MIXED_CLONE
        )
    elif blend_type == "alpha_blending":
        # alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha = alpha.astype(np.float16) / 255.0
        alpha = alpha[..., None]
        blended_image = foreground[..., :3] * alpha + background * (1 - alpha)
        blended_image = blended_image.astype(np.uint8)
    else:
        logger.error(f"Unknown blend_type: {blend_type}")

    return blended_image


class Renderer:
    def __init__(
        self, device, image_size, intrinsics, predicted_light, use_predicted_light=False
    ):
        self.device = device
        self.aa_factor = 3
        raster_settings_soft = RasterizationSettings(
            image_size=image_size * self.aa_factor,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        materials = Materials(
            diffuse_color=((0.8, 0.8, 0.8),),
            specular_color=((0.2, 0.2, 0.2),),
            shininess=30,
            device=device,
        )

        if use_predicted_light:
            self.lighting = DirectionalLights(
                direction=predicted_light["directions"], device=device
            )  # diffuse_color=predicted_light['colors'],
        else:
            self.lighting = PointLights(device=device)

        fxfy, cxcy = Renderer.ndc_fxfy_cxcy(intrinsics, image_size)
        self.cameras = PerspectiveCameras(
            focal_length=-fxfy, principal_point=cxcy, device=self.device
        )

        self.renderer_p3d = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
            shader=HardPhongShader(
                materials=materials,
                device=device,
            ),
        )
        return

    def render(self, meshes):
        rendered_images = self.renderer_p3d(
            meshes, lights=self.lighting, cameras=self.cameras
        )
        rendered_images = rendered_images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        rendered_images = F.avg_pool2d(
            rendered_images, kernel_size=self.aa_factor, stride=self.aa_factor
        )
        rendered_images = rendered_images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        return rendered_images

    @staticmethod
    def ndc_fxfy_cxcy(Ks, image_size):
        ndc_fx = Ks[:, 0, 0] * 2 / image_size
        ndc_fy = Ks[:, 1, 1] * 2 / image_size
        ndc_px = -(Ks[:, 0, 2] - image_size / 2.0) * 2 / image_size
        ndc_py = -(Ks[:, 1, 2] - image_size / 2.0) * 2 / image_size
        focal_length = torch.stack([ndc_fx, ndc_fy], dim=-1)
        principal_point = torch.stack([ndc_px, ndc_py], dim=-1)
        return focal_length, principal_point
