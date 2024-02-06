import logging
import os
from typing import Any, NamedTuple

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from rich.console import Console
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from dataset import HandData, _P3DFaces
from module import OptimizeObject
from visualization import Renderer

logger = logging.getLogger(__name__)
console = Console()


class ReconstructHand(LightningModule):
    def __init__(self, cfg, accelerator, object_optimization=None):
        self.cfg = cfg
        self.object_optimization = object_optimization
        self.accelerator = accelerator
        self.hand_dataset = instantiate(cfg.hand_dataset)
        self.dataloader = DataLoader(
            self.hand_dataset, batch_size=cfg.obtimize_object.hand_batch, shuffle=False
        )
        self.inpainter = instantiate(cfg.vis.inpaint)
        return

    def predict_dataloader(self):
        return self.dataloader

    def predict_step(self, batch):
        data = HandData(**batch)
        images = data.images.numpy()
        inpainted_images = data.inpainted_images
        handarm_segs = data.handarm_segs
        object_segs = data.object_segs
        intrinsics = data.intrinsics
        light = data.light
        hand_verts = data.hand_verts
        hand_faces = data.hand_faces
        hand_aux = data.hand_aux
        mano_pose = data.mano_pose

        batch_size = hand_verts.shape[0]
        image_size = images.shape[1]
        if image_size != images.shape[2]:
            logger.error("Only support square image")

        # inpaint
        if inpainted_images is None:
            with console.status(
                "Removing and inpainting the hand...", spinner="monkey"
            ):
                inpainted_images = self.inpainter(images, handarm_segs, object_segs)
                inpainted_path = self.hand_dataset.image.inpainted_path
                inapinted_dir = os.path.dirname(inpainted_path)
                os.makedirs(inapinted_dir, exist_ok=True)
                for b in range(batch_size):
                    Image.fromarray(inpainted_images[b]).save(inpainted_path)
                    logger.info(f"Saved {inpainted_path}")
        else:
            inpainted_images = inpainted_images.numpy()

        # normalize to center
        hand_original_verts = hand_verts.clone()
        hand_original_faces = hand_faces
        hand_verts, hand_center, hand_max_norm = OptimizeObject.batch_normalize_mesh(
            hand_verts
        )
        for b in range(batch_size):
            logger.debug(
                f"batch {b}, [hand] center: {hand_center[b]}, max_norm: {hand_max_norm[b]:.3f}"
            )

        # nimble to mano
        if self.hand_dataset.nimble:
            logger.debug("Hand model: NIMBLE")
            hand_verts, hand_faces_verts_idx = self.hand_dataset.nimble_to_mano(
                hand_verts
            )
            hand_faces_verts_idx = hand_faces_verts_idx.unsqueeze(0).repeat(
                hand_verts.shape[0], 1, 1
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
                    mano_pose, hand_original_verts, hand_original_faces
                )
            else:
                logger.error("With arm, mano is not implemented. Use nimble.")

        # Pytorch3D renderer
        renderer = Renderer(
            image_size,
            intrinsics,
            light,
            self.cfg.vis.render.use_predicted_light,
        )

        # give data
        handresult = HandResult(
            batch_size=hand_verts.shape[0],
            verts=hand_verts,
            faces=hand_faces,
            aux=hand_aux,
            max_norm=hand_max_norm,
            center=hand_center,
            original_verts=hand_original_verts,
            original_faces=hand_original_faces,
            renderer=renderer,
            inpainted_images=inpainted_images,
        )

        # optimize
        object_optimization = OptimizeObject(self.cfg, handresult)
        callbacks = [LearningRateMonitor(logging_interval="step")]
        loggers = [WandbLogger(proejct="optimize_object", offline=self.cfg.debug)]
        trainer = pl.Trainer(
            deterministic=True,
            devices=self.cfg.devices,
            accelerator=self.accelerator,
            callbacks=callbacks,
            loggers=loggers,
        )
        trainer.fit(object_optimization)

        return


class HandResult:
    batch_size: int
    verts: Tensor
    faces: NamedTuple
    aux: NamedTuple
    max_norm: Tensor
    center: Tensor
    original_verts: Tensor
    original_faces: Tensor
    renderer: Any
    inpainted_images: np.ndarray
