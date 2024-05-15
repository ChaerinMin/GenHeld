import cv2
from matplotlib.colors import hsv_to_rgb
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt

from visualization.lines import bones


def vis_keypoints(img: np.ndarray, kp, vis_lines=True) -> np.ndarray:
    eps = 0.01
    if kp is None:
        return img
    if img.shape[-1] == 4:
        alpha = img[..., 3]
        img = img[..., :3]
    else:
        alpha = None
    output_canvas = np.copy(img)
    keypoints = np.copy(kp)

    if vis_lines:
        for ie, (e1, e2) in enumerate(bones):
            k1 = keypoints[e1]
            k2 = keypoints[e2]
            if k1 is None or k2 is None:
                continue
            x1 = int(k1[0])
            y1 = int(k1[1])
            x2 = int(k2[0])
            y2 = int(k2[1])
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    output_canvas,
                    (x1, y1),
                    (x2, y2),
                    hsv_to_rgb([ie / float(len(bones)), 1.0, 1.0]) * 255,
                    thickness=2,
                )

    for ik, keypoint in enumerate(keypoints):
        x, y = keypoint[0], keypoint[1]
        x = int(x)
        y = int(y)
        if x > eps and y > eps:
            cv2.circle(
                output_canvas,
                (x, y),
                4,
                hsv_to_rgb([ik / float(len(keypoints)), 1.0, 1.0]) * 255,
                thickness=-1,
            )
    
    if alpha is not None:
        output_canvas = np.concatenate([output_canvas, alpha[..., None]], axis=-1)

    return output_canvas


joint_color = plt.cm.gist_rainbow(np.linspace(0, 1, 21))[:, :-1]

def vis_joints(joints, bone_path, joint_path):
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    if '.' not in bone_path:
        bone_path += '.ply'
    if '.' not in joint_path:
        joint_path += '.ply'
    if joints.ndim == 3:
        joints = joints[0]
        print(f"Warning: joints shape is {joints.shape}, only visualizing first frame.")
    if joints.shape[1] != 3:
        raise ValueError(f"Invalid joints shape: {joints.shape}")

    # joints lines
    kp = o3d.utility.Vector3dVector(joints)
    lines = o3d.utility.Vector2iVector(bones)
    line_set = o3d.geometry.LineSet(kp, lines)
    o3d.io.write_line_set(bone_path, line_set)
    # joints colors
    keypoints = o3d.geometry.PointCloud(kp)
    keypoints.colors = o3d.utility.Vector3dVector(joint_color)
    o3d.io.write_point_cloud(joint_path, keypoints)
    return