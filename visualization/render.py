import logging

import cv2
import copy
import numpy as np
import torch
from pytorch3d.renderer import (
    HardPhongShader,
    SoftSilhouetteShader,
    Materials,
    MeshRasterizer,
    MeshRendererWithFragments,
    RasterizationSettings,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.lighting import PointLights
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import KDTree

from utils.joints import mediapipe_to_kp

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self, device, image_size, intrinsics):
        self.device = device
        raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        materials = Materials(
            diffuse_color=((0.8, 0.8, 0.8),),
            specular_color=((0.2, 0.2, 0.2),),
            shininess=30,
            device=self.device,
        )

        self.lighting = PointLights(device=self.device)

        fxfy, cxcy = Renderer.ndc_fxfy_cxcy(intrinsics, image_size)
        self.cameras = PerspectiveCameras(
            focal_length=-fxfy, principal_point=cxcy, device=self.device
        )

        self.renderer_p3d = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
            shader=HardPhongShader(
                materials=materials,
                device=self.device,
            ),
        )

        self.renderer_wo_texture = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
            shader=SoftSilhouetteShader(),
        )
        return

    def render(self, meshes):
        rendered_images, fragments = self.renderer_p3d(
            meshes, lights=self.lighting, cameras=self.cameras
        )
        return rendered_images, fragments.zbuf

    def render_wo_texture(self, meshes):
        rendered_images, fragments = self.renderer_wo_texture(
            meshes, cameras=self.cameras
        )
        return rendered_images, fragments.zbuf

    @staticmethod
    def ndc_fxfy_cxcy(Ks, image_size):
        ndc_fx = Ks[:, 0, 0] * 2 / image_size
        ndc_fy = Ks[:, 1, 1] * 2 / image_size
        ndc_px = -(Ks[:, 0, 2] - image_size / 2.0) * 2 / image_size
        ndc_py = -(Ks[:, 1, 2] - image_size / 2.0) * 2 / image_size
        focal_length = torch.stack([ndc_fx, ndc_fy], dim=-1)
        principal_point = torch.stack([ndc_px, ndc_py], dim=-1)
        return focal_length, principal_point


def blend_images(foreground, background, use_alpha=True, blend_type="alpha_blending"):
    assert foreground.shape[-1] == 4, "Foreground must have alpha channel"

    # make mask
    if not use_alpha:
        logger.error("Only support use_alpha=True")
        raise ValueError
    alpha = foreground[..., 3]
    mask = alpha > (255 / 2.0)
    mask = mask * np.array(255, dtype=np.uint8)

    if (
        foreground.shape[0] != background.shape[0]
        or foreground.shape[1] != background.shape[1]
    ):
        logger.error("Only support foreground and background have the same shape")
        raise ValueError

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
        raise ValueError

    return blended_image


def warp_object(src_object, dst, src):
    """
    Translate object by hand root
    No batching
    """
    base_options = mp_python.BaseOptions(
        model_asset_path="assets/mediapipe/hand_landmarker.task"
    )
    options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.16, min_hand_presence_confidence=0.16)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(options=options)
    src = copy.deepcopy(src)
    src_pts = hand_landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=src)
    )
    dst_pts = hand_landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=dst)
    )
    if len(src_pts.hand_landmarks) == 0 or len(dst_pts.hand_landmarks) == 0:
        return None, None
    src_pts = mediapipe_to_kp(src_pts, src.shape[:2])
    dst_pts = mediapipe_to_kp(dst_pts, dst.shape[:2])
    tr2 = dst_pts[2, :] - src_pts[2, :]
    tr4 = dst_pts[4, :] - src_pts[4, :]
    tr = (tr2 + tr4) / 2
    M = np.array([[1, 0, tr[0]], [0, 1, tr[1]]], dtype=np.float32)
    dst_object = cv2.warpAffine(src_object, M, (dst.shape[1], dst.shape[0]))
    return dst_object, M


def warp_occ(src_occ, src_mask, M, dst_mask):
    """
    Warp occlusion mask by transformation matrix
    No batching
    """
    src_occ = cv2.warpAffine(src_occ, M, (dst_mask.shape[1], dst_mask.shape[0]))
    src_mask = cv2.warpAffine(src_mask, M, (dst_mask.shape[1], dst_mask.shape[0]))
    src_coords = np.column_stack((src_mask > 0.5).nonzero())
    dst_coords = np.column_stack((dst_mask > 127).nonzero())
    vis = np.zeros(src_mask.shape, np.uint8)[:, :, None]
    vis = np.repeat(vis, 3, axis=-1)
    vis[src_coords[:, 0], src_coords[:, 1], 0] = 255
    vis[dst_coords[:, 0], dst_coords[:, 1], 1] = 255
    tree = KDTree(src_coords)
    _, indices = tree.query(dst_coords)
    nn_coords = src_coords[indices]
    dst_occ = np.ones_like(src_occ)
    dst_occ[dst_coords[:, 0], dst_coords[:, 1]] = src_occ[
        nn_coords[:, 0], nn_coords[:, 1]
    ]
    return dst_occ
