import os
import torch
from tqdm import tqdm
from pytorch3d.io import load_obj, save_obj, load_ply
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles
from loss import contact_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
import numpy as np

mpl.rcParams["figure.dpi"] = 80

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

hand_obj_path = "/dev/hdd/hamer/demo_out/test2_1.obj"
# object_obj_path = "/dev/hdd/YCB/031_spoon/google_16k/textured.obj"
object_obj_path = "/dev/hdd/contactgen/exp/demo_results/toothpaste.ply"
sampled_verts_path = "/dev/hdd/contactgen/exp/demo_results/obj_verts_7.npy"
contacts_path = "/dev/hdd/contactgen/exp/demo_results/contacts_7.npy"
partition_path = "/dev/hdd/contactgen/exp/demo_results/partition_7.npy"
out_obj_dir = "results"
os.makedirs(out_obj_dir, exist_ok=True)

hand_verts, hand_faces, hand_aux = load_obj(hand_obj_path)
hand_faces = hand_faces.verts_idx
# object_verts, object_faces, object_aux = load_obj(object_obj_path)
# object_faces = object_faces.verts_idx
object_verts, object_faces = load_ply(object_obj_path)
sampled_verts = torch.from_numpy(np.load(sampled_verts_path))

hand_verts = hand_verts.to(device)
hand_faces = hand_faces.to(device)
object_verts = object_verts.to(device)
object_faces = object_faces.to(device)
sampled_verts = sampled_verts.to(device)


def normalize_mesh(verts):
    center = verts.mean(dim=0, keepdim=True)
    verts = verts - center
    max_norm = verts.norm(dim=1).max()
    verts = verts / max_norm
    return verts, center, max_norm


hand_verts, hand_center, hand_max_norm = normalize_mesh(hand_verts)
object_verts, object_center, object_max_norm = normalize_mesh(object_verts)
print(f"[hand] center: {hand_center}, max_norm: {hand_max_norm}")
print(f"[object] center: {object_center}, max_norm: {object_max_norm}")
sampled_verts = (sampled_verts - object_center) / object_max_norm

s_params = torch.ones(1, requires_grad=True, device=device)
t_params = torch.zeros(1, 3, requires_grad=True, device=device)
R_params = torch.zeros(1, 3, requires_grad=True, device=device)
optimizer = torch.optim.Adam([s_params, t_params, R_params], lr=0.1)

Niter = 2000
plot_mesh_period = 19
plot_pc_period = 1000000
loop = tqdm(range(Niter))
attraction_losses = []
repulsion_losses = []
attraction_weight = 0.1
repulsion_weight = 0.0


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


writer = wandb.init(project="optimize_object", mode="online")

for i in loop:
    optimizer.zero_grad()

    # transform the object mesh
    s_sigmoid = torch.sigmoid(s_params) * 3.0 - 1.5
    R_matrix = axis_angle_to_matrix(R_params)
    t = Transform3d(device=device).scale(s_sigmoid).rotate(R_matrix).translate(t_params)
    new_object_verts = t.transform_points(object_verts)
    new_sampled_verts = t.transform_points(sampled_verts)

    # plot point cloud
    if i % plot_pc_period == plot_pc_period - 1:
        verts = torch.cat([hand_verts, new_object_verts], dim=0)
        plot_pointcloud(verts, title=f"iter: {i}")

    # save mesh
    if i % plot_mesh_period == 0:
        verts = torch.cat([hand_verts, new_object_verts], dim=0)
        faces = torch.cat([hand_faces, object_faces + hand_verts.shape[0]], dim=0)
        out_obj_path = os.path.join(out_obj_dir, "combined_{}.obj".format(i))
        save_obj(out_obj_path, verts, faces)

    # loss
    attraction_loss, repulsion_loss, contact_info, metrics = contact_loss(
        hand_verts.unsqueeze(0),
        hand_faces,
        new_object_verts.unsqueeze(0),
        object_faces,
        sampled_verts=new_sampled_verts.unsqueeze(0),
        contacts_path=contacts_path,
        partition_path=partition_path,
    )
    attraction_losses.append(attraction_loss.item())
    repulsion_losses.append(repulsion_loss.item())
    loss = attraction_weight * attraction_loss + repulsion_weight * repulsion_loss

    loss.backward()
    optimizer.step()

    # logging
    loop.set_description(
        "Loss: {:.4f} Attr: {:.5f} Rep: {:.5f}".format(
            loss.item(), attraction_loss.item(), repulsion_loss.item()
        )
    )
    R_euler = matrix_to_euler_angles(R_matrix, "XYZ")
    writer.log(
        {
            "loss": loss,
            "attraction_loss": attraction_loss,
            "repulsion_loss": repulsion_loss,
            "scale": s_sigmoid.item(),
            "translate_x": t_params[0, 0].item(),
            "translate_y": t_params[0, 1].item(),
            "translate_z": t_params[0, 2].item(),
            "rotate_x": R_euler[0, 0].item(),
            "rotate_y": R_euler[0, 1].item(),
            "rotate_z": R_euler[0, 2].item(),
            "iter": i,
        }
    )
