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

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.info(f"System timezone is {time.strftime('%Z')}")

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
    # paths
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Output directory is {output_dir}")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    cfg.results_dir = results_dir

    # save config
    OmegaConf.save(config=cfg, f=os.path.join(results_dir, "config.yaml"))

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
    )
    logger.info(f"Max global steps for hands: {reconstructor.max_steps}")
    logger.info(f"Max epochs for hands: {reconstructor.max_epochs}")
    reconstructor.fit(reconstruction)

    return


if __name__ == "__main__":
    main()
