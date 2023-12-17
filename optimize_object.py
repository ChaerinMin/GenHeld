import os
import torch
import tqdm
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Transform3d
from loss import contact_loss

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

hand_obj_path = "/dev/hdd/hamer/demo_out/test2_1.obj"
object_obj_path = "/dev/hdd/YCB/031_spoon/google_16k/textured.obj"
out_obj_dir = "results"
os.makedirs(out_obj_dir, exist_ok=True)

hand_verts, hand_faces, hand_aux = load_obj(hand_obj_path)
object_verts, object_faces, object_aux = load_obj(object_obj_path)

hand_verts = hand_verts.to(device)
object_verts = object_verts.to(device)


def normalize_mesh(verts):
    verts = verts - verts.mean(dim=0, keepdim=True)
    max_norm = verts.norm(dim=1).max()
    verts = verts / max_norm
    return verts


hand_verts = normalize_mesh(hand_verts)
object_verts = normalize_mesh(object_verts)

s_params = torch.ones(1, requires_grad=True, device=device)
t_params = torch.zeros(1, 3, requires_grad=True, device=device)
R_params = torch.eye(3, requires_grad=True, device=device)
optimizer = torch.optim.SGD([s_params, t_params, R_params], lr=1.0, momentum=0.9)

Niter = 2000
plot_period = 250
loop = tqdm(range(Niter))
attraction_losses = []
repulsion_losses = []

for i in loop:
    optimizer.zero_grad()

    # transform the object mesh
    object_center = object_verts.mean(dim=0, keepdim=True)
    object_verts = object_verts - object_center
    t = Transform3d().scale(s_params).rotate(R_params).translate(t_params)
    object_verts = t.transform_points(object_verts)
    object_verts = object_verts + object_center

    # loss
    attraction_loss, repulsion_loss, contact_info, metrics = contact_loss(
        hand_verts, hand_faces, object_verts, object_faces
    )
    attraction_losses.append(attraction_loss.item())
    repulsion_losses.append(repulsion_loss.item())
    loss = attraction_loss + repulsion_loss

    loss.backward()
    optimizer.step()
    loop.set_description(
        "Attr: {:.4f} Rep: {:.4f}".format(attraction_loss.item(), repulsion_loss.item())
    )

    # combine two meshes
    if i % plot_period == 0:
        verts = torch.cat([hand_verts, object_verts], dim=0)
        faces = torch.cat([hand_faces, object_faces + hand_verts.shape[0]], dim=0)
        out_obj_path = os.path.join(out_obj_dir, "combined_{}.obj".format(i))
        save_obj(out_obj_path, verts, faces)
