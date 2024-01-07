import os
import time
import logging
import torch
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optimization.optimize_object import OptimizeObject
import numpy as np
import random

logger = logging.getLogger(__name__)

random_seed = 980828
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True  # if final test
# torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg):
    # paths
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Output directory is {output_dir}")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    cfg.results_dir = results_dir

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logger.warning("CPU only, this will be slow!")

    # loggers
    wandb_mode = "disabled" if cfg.debug else "online"
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    writer = wandb.init(project="optimize_object", mode=wandb_mode, config=wandb_config)
    logger.info(f"System timezone is {time.strftime('%Z')}")

    # main
    dataset = instantiate(cfg.dataset)
    optimizer = OptimizeObject(cfg, dataset=dataset)
    optimizer.optimize(device, writer)
    return


if __name__ == "__main__":
    main()
