import logging
import os
import random
import time
import warnings

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from module import ReconstructHand
from configs.compare_configs import compare_cfg

torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
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
    else:
        accelerator = "cpu"
        logger.warning("CPU only, this will be slow!")

    # main
    reconstruction = ReconstructHand(cfg, accelerator)
    reconstructor = pl.Trainer(
        devices=cfg.devices,
        accelerator=accelerator,
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=cfg.output_dir,
    )
    if not cfg.test_only:
        logger.info(f"Max global steps for hands: {reconstructor.max_steps}")
        logger.info(f"Max epochs for hands: {reconstructor.max_epochs}")
        reconstructor.fit(reconstruction)

    # evaluate
    reconstructor.test(reconstruction)

    return


if __name__ == "__main__":
    main()
