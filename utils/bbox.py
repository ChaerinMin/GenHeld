import torch
import numpy as np
import open3d as o3d 
from pytorch3d.transforms import Transform3d


def get_bbox(points):
    '''
    No batching
    '''
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = o3d.utility.Vector3dVector(points)
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        points
    )
    bbox = bbox.get_minimal_oriented_bounding_box()
    bbox_corners = bbox.get_box_points()
    bbox_size = bbox.extent
    bbox_size = np.sort(bbox_size)[::-1]
    return bbox_size, bbox_corners

def scale_to_bbox(points, goal_scale=None, scaling_fn=None, stay_aligned=True, dealignment_fn=None):
    '''
    Yes batching
    '''
    assert points.padded.dim() == 3
    batch_size = points.padded.shape[0]
    device = points.device

    if scaling_fn is None:
        assert goal_scale is not None and dealignment_fn is None
        old_R = []
        old_t = []
        old_s = []
        longest_orders = []
        for b in range(batch_size):
            point = points.padded[b]
            point = point.cpu().numpy()
            point = o3d.utility.Vector3dVector(point)
            old_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                point
            )
            old_bbox = old_bbox.get_minimal_oriented_bounding_box()
            old_s.append(old_bbox.extent)
            longest_orders.append(np.argsort(old_bbox.extent)[::-1])
            old_R.append(old_bbox.R)
            old_t.append(old_bbox.center)
        old_R = torch.tensor(np.array(old_R), device=device)
        old_t = torch.tensor(np.array(old_t), device=device)
        old_s = torch.tensor(np.array(old_s), device=device)
        old_s_inv = 1 / (old_s + 1e-6)
        longest_orders = torch.tensor(np.array(longest_orders), device=device)
        goal_scale = torch.gather(goal_scale, 1, longest_orders)
        scaling_fn = Transform3d(device=device).translate(-old_t).rotate(old_R.transpose(1,2)).scale(old_s_inv * goal_scale)
    else:
        assert goal_scale is None
        if not stay_aligned:
            assert dealignment_fn is not None
    
    points.padded = scaling_fn.transform_points(points.padded)
    if not stay_aligned:
        if dealignment_fn is None:
            dealignment_fn = Transform3d(device=device).rotate(old_R).translate(old_t)
        points.padded = dealignment_fn.transform_points(points.padded)
    points.clean(points.split_sizes)

    return points, scaling_fn, dealignment_fn