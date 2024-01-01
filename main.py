import hydra
import os
import torch
import wandb
import logging
import time
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optimization.optimize_object import OptimizeObject

logger = logging.getLogger(__name__)


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
