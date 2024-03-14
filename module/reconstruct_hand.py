from dataclasses import dataclass
import logging
import os
from typing import Any, NamedTuple
import glob
import re

import numpy as np
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from rich.console import Console
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.sampler import Sampler
import trimesh
from scipy.optimize import minimize
import pandas as pd

from dataset import HandData, _P3DFaces
from module.testtime_optimize import TestTimeOptimize
from visualization import Renderer
from utils import batch_normalize_mesh


logger = logging.getLogger(__name__)
console = Console()


class DiscriminateHand(Sampler):
    def __init__(self, opt, cfg, dataset):
        self.opt = opt
        self.cfg = cfg
        self.dataset = dataset

        # convex hull visualize path
        self.convex_hull_dir = os.path.join(self.cfg.output_dir, "hand_discrimination")
        os.makedirs(self.convex_hull_dir, exist_ok=True)

        # accepted hands path
        self.accepted_hands_path = os.path.join(
            self.cfg.output_dir, "accepted_hands.npy"
        )
        if not os.path.exists(self.accepted_hands_path):
            accepted_hands = np.zeros((len(self.dataset), 2), dtype=np.int32)
            accepted_hands[:, 0] = np.array(self.dataset.fidxs)
            np.save(self.accepted_hands_path, accepted_hands)
            logger.debug(f"Created {self.accepted_hands_path}")
        else:
            logger.debug(f"{self.accepted_hands_path} exists. Will be reused.")

        return

    @staticmethod
    def objective_function(center, convex_hull):
        sdf = trimesh.proximity.signed_distance(
            convex_hull, [center]
        )  # pip install rtree needed
        sdf *= -1  # make outside positive
        sdf += convex_hull.scale  # avoid negative loss
        return sdf

    def __iter__(self):
        for i, data in enumerate(self.dataset):
            # convel hull
            mesh = trimesh.Trimesh(
                vertices=data["hand_verts"], faces=data["hand_faces"].verts_idx
            )
            convex_hull = mesh.convex_hull

            # optimize
            initial_center = convex_hull.centroid
            result = minimize(
                DiscriminateHand.objective_function,
                initial_center,
                args=(convex_hull,),
                method=self.opt.scipy_method,
                options={"maxiter": self.opt.max_iter},
            )
            inscripted_radius = result.fun
            inscripted_radius -= convex_hull.scale
            inscripted_radius *= -1

            # save
            convex_hull_path = os.path.join(
                self.convex_hull_dir, f"{data['fidxs']:08d}_convexhull.obj"
            )
            convex_hull.export(convex_hull_path)
            sphere_path = os.path.join(
                self.convex_hull_dir, f"{data['fidxs']:08d}_sphere.obj"
            )
            inscripted_center = result.x
            sphere = trimesh.creation.icosphere(
                subdivisions=3, radius=inscripted_radius
            )
            sphere.apply_translation(inscripted_center)
            sphere.export(sphere_path)

            # accept or reject
            inscripted_ratio = sphere.volume / mesh.volume
            success = inscripted_ratio >= self.opt.accept_thresh
            if success:
                logger.debug(
                    f"Accepted {data['fidxs']}. Radius: {inscripted_radius:.3f}"
                )
                accepted_hands = np.load(self.accepted_hands_path)
                assert accepted_hands[i, 0] == data["fidxs"]
                accepted_hands[i, 1] = 1
                np.save(self.accepted_hands_path, accepted_hands)
                logger.debug(f"Updated {self.accepted_hands_path}")
                yield i
            else:
                logger.info(
                    f"Rejected {data['fidxs']}. Radius: {inscripted_radius:.3f}"
                )

        return

    def __len__(self):
        return len(self.dataset)


class ReconstructHand(LightningModule):
    def __init__(self, cfg, accelerator, device):
        super().__init__()
        self.cfg = cfg
        self.accelerator = accelerator
        self.manual_device = device
        self.hand_dataset = instantiate(
            cfg.hand_dataset, cfg=cfg, device=device, _recursive_=False
        )
        self.inpainter = instantiate(cfg.vis.inpaint, device=device, _recursive_=False)

        self.hifihr_intrinsics = torch.tensor(
            [
                [531.9495243872041, 0.0, 112.0],
                [0.0, 532.2600075028636, 112.0],
                [0.0, 0.0, 1.0],
            ]
        )  # hifihr assumption
        self.hifihr_intrinsics = self.hifihr_intrinsics.unsqueeze(0).repeat(
            self.cfg.batch_size, 1, 1
        )

        return

    def on_test_start(self):
        self.metric_results = []
        return

    def train_dataloader(self):
        hand_discriminator = instantiate(
            self.cfg.reconstruct_hand.discriminate_hand,
            cfg=self.cfg,
            dataset=self.hand_dataset,
            _recursive_=False,
        )
        self.dataloader = DataLoader(
            self.hand_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=hand_discriminator,
        )
        return self.dataloader

    def test_dataloader(self):
        # test_fidxs
        ckpts = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "*.json")))
        test_fidxs = []
        pattern = r"hand_(\d+)"
        for ckpt in ckpts:
            match = re.search(pattern, os.path.basename(ckpt))
            if match is not None:
                fidx = int(match.group(1))
                test_fidxs.append(fidx)
                logger.debug(f"Checkpoint {ckpt} is included in the test")

        self.hand_dataset.fidxs = test_fidxs
        self.dataloader = DataLoader(
            self.hand_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )
        return self.dataloader

    def configure_optimizers(self):
        self.dummy = torch.tensor([1.0], requires_grad=True)
        return torch.optim.Adam([self.dummy], lr=1.0)

    def forward(self, batch):
        data = HandData(**batch)
        fidxs = data.fidxs
        images = data.images.cpu().numpy()
        inpainted_images = data.inpainted_images
        hand_theta = data.hand_theta
        hand_verts = data.hand_verts
        hand_verts_r = data.hand_verts_r
        hand_faces = data.hand_faces
        hand_aux = data.hand_aux
        xyz = data.xyz

        batch_size = hand_verts.shape[0]
        image_size = images.shape[1]
        if image_size != images.shape[2]:
            logger.error("Only support square image")
            raise ValueError

        # inpaint
        if inpainted_images is None:
            with console.status(
                "Removing and inpainting the hand...", spinner="monkey"
            ):
                inpainted_images = self.inpainter(images, fidxs)
                inapinted_dir = os.path.dirname(
                    self.hand_dataset.cached.image.inpainted_path
                )
                os.makedirs(inapinted_dir, exist_ok=True)
                for b in range(batch_size):
                    inpainted_path = (
                        self.hand_dataset.cached.image.inpainted_path % fidxs[b]
                    )
                    Image.fromarray(inpainted_images[b]).save(inpainted_path)
                    logger.info(f"Saved {inpainted_path}")
        else:
            inpainted_images = inpainted_images.cpu().numpy()

        # normalize to center
        hand_original_verts = hand_verts.clone()
        hand_original_faces = hand_faces
        hand_verts_n, hand_center, hand_max_norm = batch_normalize_mesh(
            hand_verts
        )
        for b in range(batch_size):
            logger.debug(
                f"hand {fidxs[b]}, center: {hand_center[b]}, max_norm: {hand_max_norm[b]:.3f}"
            )

        # nimble to mano
        if self.hand_dataset.nimble:
            logger.debug("Hand model: NIMBLE")
            hand_verts_n, hand_faces_verts_idx = self.hand_dataset.nimble_to_mano(
                hand_verts_n
            )
            hand_faces_verts_idx = hand_faces_verts_idx.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            hand_faces = _P3DFaces(verts_idx=hand_faces_verts_idx)
        else:
            logger.debug("Hand model: MANO")

        # nimble to nimblearm
        if self.hand_dataset.arm:
            if self.hand_dataset.nimble:
                logger.debug("With arm.")
                (
                    hand_original_verts,
                    hand_original_faces,
                ) = self.hand_dataset.nimble_to_nimblearm(
                    xyz, hand_original_verts, hand_original_faces
                )
            else:
                logger.error("With arm, mano is not implemented. Use nimble.")
                raise NotImplementedError

        # Pytorch3D renderer
        self.hifihr_intrinsics = self.hifihr_intrinsics.to(self.device)
        renderer = Renderer(self.device, image_size, self.hifihr_intrinsics)

        # give data
        handresult = HandResult(
            batch_size=hand_verts.shape[0],
            fidxs=fidxs,
            dataset=self.hand_dataset,
            theta=hand_theta,
            verts_n=hand_verts_n,  # mano
            verts_r=hand_verts_r,
            faces=hand_faces,
            aux=hand_aux,
            max_norm=hand_max_norm,
            center=hand_center,
            original_verts=hand_original_verts,  # nimble
            original_faces=hand_original_faces,
            renderer=renderer,
            inpainted_images=inpainted_images,
        )
        return handresult

    def training_step(self, batch, batch_idx):
        handresult = self(batch)

        # optimize
        tt_optimization = TestTimeOptimize(
            self.cfg, self.device, self.accelerator, handresult
        )
        print("optimizer initialized")
        callbacks = [LearningRateMonitor(logging_interval="step")]
        loggers = [
            WandbLogger(
                project="GenHeld_TTA",
                offline=self.cfg.debug,
                save_dir=self.cfg.output_dir,
            )
        ]
        trainer = pl.Trainer(
            devices=len(self.cfg.devices),
            accelerator=self.accelerator,
            callbacks=callbacks,
            logger=loggers,
            max_epochs=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=self.cfg.output_dir,
        )
        logger.info(f"Max global steps for object optimization: {trainer.max_steps}")
        logger.info(f"Max epochs for object optimization: {trainer.max_epochs}")
        trainer.fit(tt_optimization)

        return dict(loss=torch.abs(self.dummy) + 1e10)

    def test_step(self, batch, batch_idx):
        handresult = self(batch)

        tt_optimization = TestTimeOptimize(
            self.cfg, self.manual_device, self.accelerator, handresult
        )
        tester = pl.Trainer(
            devices=self.cfg.devices[0:1],
            accelerator=self.accelerator,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=self.cfg.output_dir,
        )
        tester.test(tt_optimization)

        # metrics logging
        self.metric_results.append(tt_optimization.metric_results)

        return

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # save resume_fidx
        fidxs = batch["fidxs"]
        resume_fidx = fidxs[-1] + 1
        resume_fidx_path = os.path.join(self.cfg.output_dir, "resume_fidx.txt")
        with open(resume_fidx_path, "w") as f:
            f.write(str(resume_fidx.item()) + "\n" + str(self.hand_dataset.end_fidx))
        logger.debug(f"Saved {resume_fidx_path}")

        return

    def on_test_epoch_end(self):
        df = pd.DataFrame(self.metric_results)

        # mean and std
        mean_values = df.mean(axis=0, numeric_only=True)
        std_values = df.std(axis=0, numeric_only=True)
        mean_row = {}
        std_row = {}
        for col in df.columns:
            if col in mean_values:
                mean_row[col] = mean_values[col]
                std_row[col] = std_values[col]
            else:
                mean_row[col] = "Avg"
                std_row[col] = "Std"
        mean_row = pd.DataFrame([mean_row])
        std_row = pd.DataFrame([std_row])
        df = pd.concat([df, mean_row, std_row], ignore_index=True)

        # excel logging
        save_excel_path = os.path.join(self.cfg.output_dir, "evaluation.xlsx")
        df.to_excel(save_excel_path, index=False)
        logger.info(f"Saved metrics to {save_excel_path}")

        return


@dataclass
class HandResult:
    batch_size: int
    fidxs: Tensor
    dataset: Any
    verts_n: Tensor
    faces: NamedTuple
    aux: NamedTuple
    max_norm: Tensor
    center: Tensor
    original_verts: Tensor
    original_faces: Tensor
    renderer: Any
    inpainted_images: np.ndarray
