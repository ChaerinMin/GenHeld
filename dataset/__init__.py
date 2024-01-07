from .datasets import Data
from collections import namedtuple

_P3DFaces = namedtuple(
    "_P3DFaces",
    ["verts_idx", "normals_idx", "textures_idx", "materials_idx"],
    defaults=(None,) * 4,
)  # Python 3.7+
