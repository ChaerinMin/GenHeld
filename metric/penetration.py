import logging

from trimesh import Trimesh
import trimesh

from loss.contact_utils import batch_mesh_contains_points

logger = logging.getLogger(__name__)


def penetration_depth(obj_mesh: Trimesh, hand_verts, obj_verts, obj_faces):
    trimesh.repair.fix_normals(obj_mesh)
    obj_triangles = obj_verts[obj_faces]
    exterior = batch_mesh_contains_points(hand_verts, obj_triangles)
    penetr_mask = ~exterior

    if penetr_mask.sum() == 0:
        max_depth = 0
    else:
        _, pene_depths, _ = trimesh.proximity.closest_point(
            obj_mesh, hand_verts[penetr_mask]
        )
        max_depth = pene_depths.max()

    return max_depth


def penetration_vox(hand_mesh: Trimesh, obj_mesh: Trimesh, pitch):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * (pitch**3)
    return volume


def penetration_volume(hand_mesh, obj_mesh, engine):
    trimesh.repair.fix_normals(obj_mesh)
    intersection = obj_mesh.intersection(hand_mesh, engine=engine)
    if isinstance(intersection, trimesh.Trimesh) and intersection.vertices.shape[0] > 0:
        # if not intersection.is_watertight:
        #     logger.error(
        #         "Hand-Object intersection should be watertight. If error persists, use intersection_vox only."
        #     )
        #     raise ValueError
        volume = intersection.volume
    else:
        volume = 0

    return volume
