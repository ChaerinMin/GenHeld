import logging
import os
import random
import time
import warnings

import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from module import ReconstructHand
from configs.compare_configs import compare_cfg

torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
torch.autograd.set_detect_anomaly(True)
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
warnings.filterwarnings("ignore")
OmegaConf.register_resolver("div", lambda x, y: float(x) / float(y))

# seed
random_seed = 980828
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
pl.seed_everything(random_seed, workers=True)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg):
    logger.info(f"System timezone is {time.strftime('%Z')}")
    logger.warning(f"Number of devices: {torch.cuda.device_count()}")

    # paths
    if cfg.resume_dir:
        cfg.output_dir = cfg.resume_dir
        cfg.results_dir = os.path.join(cfg.output_dir, "results")
        if not os.path.exists(cfg.results_dir):
            logger.error(f"Resume directory {cfg.results_dir} does not exist!")
            raise FileNotFoundError
        logger.info(f"Resume from {cfg.output_dir}")
    else:
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        cfg.results_dir = os.path.join(cfg.output_dir, "results")
        os.makedirs(cfg.results_dir, exist_ok=True)
    logger.info(f"Output directory is {cfg.output_dir}")
    cfg.ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    # compare config
    if cfg.resume_dir:
        old_cfg_path = os.path.join(cfg.output_dir, ".hydra", "config.yaml")
        old_cfg = OmegaConf.load(old_cfg_path)
        logger.info(f"Comparing with {old_cfg_path}")
        compare_cfg(cfg, old_cfg)

    # accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
        device = "cuda"
    else:
        accelerator = "cpu"
        device = "cpu"
        logger.warning("CPU only, this will be slow!")

    if cfg.object_selector == "train":
        cfg.selector_ckpt_dir = os.path.join(cfg.output_dir, "selector_checkpoints")
        if not os.path.exists(cfg.selector_ckpt_dir):
            os.makedirs(cfg.selector_ckpt_dir)
        object_selection = instantiate(cfg.select_object, cfg=cfg, device=device , _recursive_=False)
        callbacks = [
            ModelCheckpoint(
                dirpath=cfg.selector_ckpt_dir, monitor="val/category_top1", mode="max", verbose=cfg.debug
            ),
            LearningRateMonitor(logging_interval="step")
        ]
        loggers = [
            WandbLogger(
                project="GenHeld_Selector_train",
                offline=cfg.debug,
                save_dir=cfg.output_dir,
            )
        ]
        selector = pl.Trainer(
            devices=len(cfg.devices),
            accelerator=accelerator,
            max_epochs=cfg.select_object.opt.train.Nepochs,
            enable_checkpointing=True,
            callbacks=callbacks,
            logger=loggers,
            enable_model_summary=True,
            default_root_dir=cfg.output_dir,
            # val_check_interval=1.0,  # one epoch
            check_val_every_n_epoch=cfg.select_object.opt.val.every_n_epoch,
        )
        if cfg.selector_ckpt:  # resume training
            logger.warning(f"Resume from {cfg.selector_ckpt}")
            selector.fit(object_selection, ckpt_path=cfg.selector_ckpt)
        else:
            selector.fit(object_selection)
    elif cfg.object_selector == "inference":
        cfg.selector_ckpt_dir = os.path.dirname(cfg.selector_ckpt)
        # main
        reconstruction = ReconstructHand(cfg, accelerator, device)
        reconstructor = pl.Trainer(
            devices=len(cfg.devices),
            accelerator=accelerator,
            max_epochs=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=cfg.output_dir,
        )
        if not cfg.is_preoptimized:
            logger.info(f"Max global steps for hands: {reconstructor.max_steps}")
            logger.info(f"Max epochs for hands: {reconstructor.max_epochs}")
            reconstructor.fit(reconstruction)

        # evaluate
        reconstructor.test(reconstruction)
    else:
        logger.error(
            f"object_selector should be either train or inference, got {cfg.object_selector}"
        )
        raise ValueError

    return


if __name__ == "__main__":
    main()
