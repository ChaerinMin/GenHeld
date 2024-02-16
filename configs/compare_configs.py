import logging 

logger = logging.getLogger(__name__)

def compare_cfg(cfg, old_cfg, parent_key=''):
    for key in cfg:
        full_key = f"{parent_key}.{key}" if parent_key else key
        if key not in old_cfg:
            logger.warning(f"New config key: {full_key}. Not found in checkpoint.")
        elif isinstance(cfg[key], dict) and isinstance(old_cfg[key], dict):
            compare_cfg(cfg[key], old_cfg[key], full_key)
        elif cfg[key] != old_cfg[key]:
            logger.warning(f"Config key {full_key} : {old_cfg[key]} (checkpoint) -> {cfg[key]} (new config)")
                
    for key in old_cfg:
        if key not in cfg:
            full_key = f"{parent_key}.{key}" if parent_key else key
            logger.warning(f"Checkpoint config key {full_key}. Not found in new config.")

    return 