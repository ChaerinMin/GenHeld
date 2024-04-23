import torch
import numpy as np


def get_hand_size(mano_joints):
    '''
    mano_joints: torch.Tensor (B, 21, 3)
    Return: torch.Tensor (B, )
    '''
    hand_size = torch.norm(mano_joints[:, 17] - mano_joints[:, 0], dim=1)
    return hand_size

def mediapipe_to_kp(kp_mediapipe, img_shape):
    H, W = img_shape
    keypoints = np.zeros((21, 2))
    if len(kp_mediapipe.hand_landmarks) == 0:
        return keypoints
    for i in range(21):
        data = kp_mediapipe.hand_landmarks[0]
        keypoints[i] = [data[i].x, data[i].y]
    keypoints[:, 0] *= W
    keypoints[:, 1] *= H
    return keypoints.astype(np.int32)