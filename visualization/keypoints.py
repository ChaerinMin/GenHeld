import cv2
from matplotlib.colors import hsv_to_rgb
import numpy as np
from matplotlib import pyplot as plt

from visualization.lines import bones


def vis_keypoints(img: np.ndarray, kp, vis_lines=True) -> np.ndarray:
    eps = 0.01
    if kp is None:
        return img
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

    return output_canvas
