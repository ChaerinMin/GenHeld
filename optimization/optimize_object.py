import os
import torch
from tqdm import tqdm
from pytorch3d.io import save_obj
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles
from loss import contact_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
import logging
from dataset import Data

mpl.rcParams["figure.dpi"] = 80
logger = logging.getLogger(__name__)


def batch_normalize_mesh(verts):
    center = verts.mean(dim=1, keepdim=True)
    verts = verts - center
    max_norm = verts.norm(dim=2).max(dim=1)[0]
    verts = verts / max_norm
    return verts, center, max_norm


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


class OptimizeObject:
    def __init__(self, opt, dataset) -> None:
        self.opt = opt
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return

    def optimize(self, device, writer):
        for data in self.dataloader:
            data = Data(**data)
            data = data.to(device)
            hand_verts = data.hand_verts
            hand_faces = data.hand_faces
            object_verts = data.object_verts
            object_faces = data.object_faces
            sampled_verts = data.sampled_verts
            contact_object = data.contacts
            partition_object = data.partitions

            batch_size = hand_verts.shape[0]

            # normalize to center
            hand_verts, hand_center, hand_max_norm = batch_normalize_mesh(hand_verts)
            object_verts, object_center, object_max_norm = batch_normalize_mesh(
                object_verts
            )
            for b in range(batch_size):
                logger.debug(
                    f"batch {b}, [hand] center: {hand_center[b]}, max_norm: {hand_max_norm[b]:.3f}"
                )
                logger.debug(
                    f"batch {b}, [object] center: {object_center[b]}, max_norm: {object_max_norm[b]:.3f}"
                )
            sampled_verts = (sampled_verts - object_center) / object_max_norm

            # parameters
            s_params = torch.ones(batch_size, requires_grad=True, device=device)
            t_params = torch.zeros(batch_size, 3, requires_grad=True, device=device)
            R_params = torch.zeros(batch_size, 3, requires_grad=True, device=device)

            attraction_losses = []
            repulsion_losses = []
            loop = tqdm(range(self.opt.Niters))
            optimizer = torch.optim.Adam([s_params, t_params, R_params], lr=0.1)

            for i in loop:
                optimizer.zero_grad()

                # transform the object mesh
                s_sigmoid = torch.sigmoid(s_params) * 3.0 - 1.5
                R_matrix = axis_angle_to_matrix(R_params)
                t = (
                    Transform3d(device=device)
                    .scale(s_sigmoid)
                    .rotate(R_matrix)
                    .translate(t_params)
                )
                new_object_verts = t.transform_points(object_verts)
                new_sampled_verts = t.transform_points(sampled_verts)

                # plot point cloud
                if i % self.opt.plot.pc_period == self.opt.plot.pc_period - 1:
                    for b in batch_size:
                        verts = torch.cat([hand_verts[b], new_object_verts[b]], dim=1)
                        plot_pointcloud(verts, title=f"iter: {i}, batch{b}")

                # save mesh
                if i % self.opt.plot.mesh_period == 0:
                    verts = torch.cat([hand_verts, new_object_verts], dim=1)
                    faces = torch.cat(
                        [hand_faces, object_faces + hand_verts.shape[1]], dim=1
                    )
                    for b in range(batch_size):
                        out_obj_path = os.path.join(
                            self.cfg.results_dir,
                            "batch_{}_combined_{}.obj".format(b, i),
                        )
                        save_obj(out_obj_path, verts[b], faces[b])

                # loss
                attraction_loss, repulsion_loss, contact_info, metrics = contact_loss(
                    hand_verts,
                    hand_faces,
                    new_object_verts,
                    object_faces,
                    contact_zones=self.opt.loss.contact_zones,
                    sampled_verts=new_sampled_verts,
                    contact_object=contact_object,
                    partition_object=partition_object,
                )
                attraction_losses.append(attraction_loss.item())
                repulsion_losses.append(repulsion_loss.item())
                loss = (
                    self.opt.loss.attraction_weight * attraction_loss
                    + self.opt.loss.repulsion_weight * repulsion_loss
                )

                loss.backward()
                optimizer.step()

                # logging
                description = "Iter: {:d}, Loss: {:.4f}, Attr: {:.5f}, Rep: {:.5f}, batch size {:d}".format(
                    i,
                    loss.item() / batch_size,
                    attraction_loss.item() / batch_size,
                    repulsion_loss.item() / batch_size,
                    batch_size,
                )
                loop.set_description(description)
                if i % self.opt.plot.log_period == 0:
                    logger.info(description)
                R_euler = matrix_to_euler_angles(R_matrix, "XYZ")
                writer.log(
                    {
                        "loss": loss.item() / batch_size,
                        "attraction_loss": attraction_loss.item() / batch_size,
                        "repulsion_loss": repulsion_loss.item() / batch_size,
                        "scale": s_sigmoid[0].item(),
                        "translate_x": t_params[0, 0].item(),
                        "translate_y": t_params[0, 1].item(),
                        "translate_z": t_params[0, 2].item(),
                        "rotate_x": R_euler[0, 0].item(),
                        "rotate_y": R_euler[0, 1].item(),
                        "rotate_z": R_euler[0, 2].item(),
                        "iter": i,
                    }
                )

        return
