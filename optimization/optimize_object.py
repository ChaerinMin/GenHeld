import os
import logging
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch3d.io import IO
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles
import matplotlib as mpl
from dataset import Data, _P3DFaces
from loss import ContactLoss
from visualization import Renderer, plot_pointcloud, merge_meshes, blend_images

mpl.rcParams["figure.dpi"] = 80
logger = logging.getLogger(__name__)


def batch_normalize_mesh(verts):
    center = verts.mean(dim=1, keepdim=True)
    verts = verts - center
    max_norm = verts.norm(dim=2).max(dim=1)[0]
    verts = verts / max_norm
    return verts, center, max_norm


class OptimizeObject:
    def __init__(self, cfg, device, writer, dataset) -> None:
        self.cfg = cfg
        self.opt = cfg.optimization
        self.device = device
        self.writer = writer
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.inpainter  = instantiate(cfg.visualization.inpaint)
        self.inpainter.to(device)
        self.contact_loss = ContactLoss(cfg, self.opt.contactloss)
        return

    def optimize(self):
        for data in self.dataloader:
            data = Data(**data)
            data = data.to(self.device)
            images = data.images
            images = images.numpy()
            intrinsics = data.intrinsics
            light = data.light
            handarm_segs = data.handarm_segs
            object_segs = data.object_segs
            hand_verts = data.hand_verts
            hand_faces = data.hand_faces
            hand_aux = data.hand_aux
            object_verts = data.object_verts
            object_faces = data.object_faces
            object_aux = data.object_aux
            sampled_verts = data.sampled_verts
            contact_object = data.contacts
            partition_object = data.partitions

            batch_size = hand_verts.shape[0]
            image_size = images.shape[1]
            if image_size != images.shape[2]:
                logger.error("Only support square image")

            # inpaint
            inpainted_images = self.inpainter(images, handarm_segs, object_segs)

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

            # nimble to mano
            hand_original_verts = hand_verts.clone()
            hand_original_faces = hand_faces
            if self.dataset.nimble:
                logger.debug("Hand model: NIMBLE")
                hand_verts, hand_faces_verts_idx = self.dataset.nimble_to_mano(
                    hand_verts
                )
                hand_faces_verts_idx = hand_faces_verts_idx.unsqueeze(0).repeat(
                    hand_verts.shape[0], 1, 1
                )
                hand_faces = _P3DFaces(verts_idx=hand_faces_verts_idx)
            else:
                logger.debug("Hand model: MANO")

            # parameters
            s_params = torch.ones(batch_size, requires_grad=True, device=self.device)
            t_params = torch.zeros(
                batch_size, 3, requires_grad=True, device=self.device
            )
            R_params = torch.zeros(
                batch_size, 3, requires_grad=True, device=self.device
            )

            attraction_losses = []
            repulsion_losses = []
            loop = tqdm(range(self.opt.Niters))
            optimizer = torch.optim.Adam([s_params, t_params, R_params], lr=self.opt.lr)

            # Pytorch3D renderer
            renderer = Renderer(self.device, image_size, intrinsics, light)

            for i in loop:
                optimizer.zero_grad()

                # transform the object verts
                s_sigmoid = torch.sigmoid(s_params) * 3.0 - 1.5
                R_matrix = axis_angle_to_matrix(R_params)
                t = (
                    Transform3d(device=self.device)
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
                    is_textured = self.dataset.nimble
                    merged_meshes = merge_meshes(
                        is_textured,
                        hand_original_verts,
                        hand_original_faces,
                        hand_aux,
                        new_object_verts.detach(),
                        object_faces,
                        object_aux,
                    )
                    p3d_io = IO()
                    for b in range(batch_size):
                        out_obj_path = os.path.join(
                            self.cfg.results_dir,
                            "batch_{}_iter_{}.obj".format(b, i),
                        )
                        p3d_io.save_mesh(
                            merged_meshes[b], out_obj_path, include_textures=is_textured
                        )
                        logger.info(f"Saved {out_obj_path}")

                # render hand
                if i % self.opt.plot.render_period == 0:
                    if i % self.opt.plot.mesh_period != 0:
                        is_textured = self.dataset.nimble
                        merged_meshes = merge_meshes(
                            is_textured,
                            hand_original_verts,
                            hand_original_faces,
                            hand_aux,
                            new_object_verts.detach(),
                            object_faces,
                            object_aux,
                        )

                    if len(merged_meshes.verts_list()) > 1:  # support only one mesh
                        logger.error("Only support one mesh for rendering")
                    
                    merged_meshes.scale_verts_(hand_max_norm[0].item())
                    merged_meshes.offset_verts_(hand_center[0, 0])
                    rendered_images = renderer.render(merged_meshes)
                    rendered_images = (rendered_images * 255).cpu().numpy().astype(np.uint8)
                    for b in range(batch_size):
                        # save rendered hand
                        out_rendered_path = os.path.join(
                            self.cfg.results_dir,
                            "batch_{}_iter_{}_rendering.png".format(b, i),
                        )
                        Image.fromarray(rendered_images[b]).save(out_rendered_path)
                        logger.info(f"Saved {out_rendered_path}")

                        # original image + rendered hand
                        blended_image = blend_images(rendered_images[b], inpainted_images[b])
                        out_blended_path = os.path.join(
                            self.cfg.results_dir,
                            "batch_{}_iter_{}_blended.png".format(b, i),
                        )
                        Image.fromarray(blended_image).save(out_blended_path)
                        logger.info(f"Saved {out_blended_path}")

                # loss
                (
                    attraction_loss,
                    repulsion_loss,
                    contact_info,
                    metrics,
                ) = self.contact_loss(
                    hand_verts,
                    hand_faces,
                    new_object_verts,
                    object_faces,
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
                self.writer.log(
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

                # save contact point cloud
                if i % self.opt.plot.contact_period == 0:
                    self.contact_loss.plot_contact(iter=i)

        return
