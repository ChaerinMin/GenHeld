import open3d as o3d
from dataset.base_dataclass import PaddedTensor, _P3DFaces
import torch 
from rich.console import Console

console = Console()


def save_any_mesh(path, verts, faces):
    if '.' not in path:
        path += '.ply'
    
    # verts
    if isinstance(verts, PaddedTensor):
        verts = verts.padded
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if verts.ndim == 3:
        verts = verts[0]
        print("Warning: only the first mesh is saved")
    if verts.ndim != 2:
        raise ValueError(f"Check the verts dimenesion")
    
    # faces
    if hasattr(faces, "verts_idx"):
        faces = faces.verts_idx
    if isinstance(faces, PaddedTensor):
        faces = faces.padded
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if faces.ndim == 3:
        faces = faces[0]
        print("Warning: only the first mesh is saved")
    if faces.ndim != 2:
        raise ValueError(f"Check the faces dimenesion")
    
    verts = o3d.utility.Vector3dVector(verts)
    faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(verts, faces)
    o3d.io.write_triangle_mesh(path, mesh)
    console.print(f"Saved: {path}", style="bold green")
    return