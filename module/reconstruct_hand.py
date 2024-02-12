from dataclasses import dataclass
import logging
import os
from typing import Any, NamedTuple
import glob 
import re

import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
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

from dataset import HandData, _P3DFaces
from module.optimize_object import OptimizeObject
from visualization import Renderer

logger = logging.getLogger(__name__)
console = Console()


class ReconstructHand(LightningModule):
    def __init__(self, cfg, accelerator, object_optimization=None):
        super().__init__()
        self.cfg = cfg
        self.object_optimization = object_optimization
        self.accelerator = accelerator
        self.hand_dataset = instantiate(cfg.hand_dataset)
        self.inpainter = instantiate(cfg.vis.inpaint)

        return
    
    def train_dataloader(self):
        self.dataloader = DataLoader(
            self.hand_dataset,
            batch_size=self.cfg.optimize_object.hand_batch,
            shuffle=False,
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
                fidx = int(match.group(1)[5:])
                test_fidxs.append(fidx)
                logger.debug(f"Checkpoint {ckpt} is included in the test")
        
        self.hand_dataset.fidxs = test_fidxs
        self.dataloader = DataLoader(
            self.hand_dataset,
            batch_size=self.cfg.optimize_object.hand_batch,
            shuffle=False,
        )
        return self.dataloader

    def configure_optimizers(self):
        self.dummy = torch.tensor([1.0], requires_grad=True)
        return torch.optim.Adam([self.dummy], lr=1.0)

    def on_train_start(self):
        # resume fidx
        resume_fidx = 0
        end_fidx = self.hand_dataset.fidxs[-1] + 1
        resume_fidx_path = os.path.join(self.cfg.resume_dir, "resume_fidx.txt")
        if os.path.exists(resume_fidx_path):
            with open(resume_fidx_path, "r") as f:
                resume_fidx, end_fidx = map(int, f.read().split("\n"))
            logger.info(f"Resuming from {resume_fidx}")
        self.resume_fidx = resume_fidx
        self.end_fidx = end_fidx
        return 
    
    def forward(self, batch):
        data = HandData(**batch)
        fidxs = data.fidxs
        images = data.images.cpu().numpy()
        inpainted_images = data.inpainted_images
        handarm_segs = data.handarm_segs
        object_segs = data.object_segs
        intrinsics = data.intrinsics
        light = data.light
        hand_verts = data.hand_verts
        hand_faces = data.hand_faces
        hand_aux = data.hand_aux
        xyz = data.xyz

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
                inapinted_dir = os.path.dirname(self.hand_dataset.image.inpainted_path)
                os.makedirs(inapinted_dir, exist_ok=True)
                for b in range(batch_size):
                    inpainted_path = self.hand_dataset.image.inpainted_path % fidxs[b]
                    Image.fromarray(inpainted_images[b]).save(inpainted_path)
                    logger.info(f"Saved {inpainted_path}")
        else:
            inpainted_images = inpainted_images.cpu().numpy()

        # normalize to center
        hand_original_verts = hand_verts.clone()
        hand_original_faces = hand_faces
        hand_verts, hand_center, hand_max_norm = OptimizeObject.batch_normalize_mesh(
            hand_verts
        )
        for b in range(batch_size):
            logger.debug(
                f"hand {fidxs[b]}, center: {hand_center[b]}, max_norm: {hand_max_norm[b]:.3f}"
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
                    xyz, hand_original_verts, hand_original_faces
                )
            else:
                logger.error("With arm, mano is not implemented. Use nimble.")

        # Pytorch3D renderer
        renderer = Renderer(
            self.device,
            image_size,
            intrinsics,
            light,
            self.cfg.vis.render.use_predicted_light,
        )

        # give data
        handresult = HandResult(
            batch_size=hand_verts.shape[0],
            fidxs=fidxs,
            dataset=self.hand_dataset,
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
        return handresult
    
    def training_step(self, batch, batch_idx):
        handresult = self(batch)
        if batch['fidxs'][-1] < self.resume_fidx or self.end_fidx <= batch['fidxs'][0]:
            return

        # optimize
        object_optimization = OptimizeObject(self.cfg, handresult)
        callbacks = [LearningRateMonitor(logging_interval="step")]
        loggers = [WandbLogger(project="optimize_object", offline=self.cfg.debug, save_dir=self.cfg.output_dir)]
        trainer = pl.Trainer(
            devices=self.cfg.devices,
            accelerator=self.accelerator,
            callbacks=callbacks,
            logger=loggers,
            max_epochs=1,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        logger.info(f"Max global steps for object optimization: {trainer.max_steps}")
        logger.info(f"Max epochs for object optimization: {trainer.max_epochs}")
        trainer.fit(object_optimization)

        return dict(loss=torch.abs(self.dummy) + 1e10)

    def test_step(self, batch, batch_idx):
        handresult = self(batch)

        object_optimization = OptimizeObject(self.cfg, handresult)
        loggers = [WandbLogger(project="test_object", offline=self.cfg.debug, save_dir=self.cfg.output_dir)]
        tester = pl.Trainer(
            devices=self.cfg.devices[0:1],
            accelerator=self.accelerator,
            logger=loggers,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        tester.test(object_optimization)

        return 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):       
        # save resume_fidx
        fidxs = batch['fidxs']
        if self.resume_fidx <= fidxs[-1] and fidxs[0] < self.end_fidx:
            self.resume_fidx =fidxs[-1] + 1
            resume_fidx_path = os.path.join(self.cfg.output_dir, "resume_fidx.txt")
            with open(resume_fidx_path, "w") as f:
                f.write(str(self.resume_fidx.item())+"\n"+str(self.end_fidx))
            logger.debug(f"Saved {resume_fidx_path}")

        return 

@dataclass
class HandResult:
    batch_size: int
    fidxs: Tensor
    dataset: Any
    verts: Tensor
    faces: NamedTuple
    aux: NamedTuple
    max_norm: Tensor
    center: Tensor
    original_verts: Tensor
    original_faces: Tensor
    renderer: Any
    inpainted_images: np.ndarray
