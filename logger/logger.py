import os
from loguru import logger


def setup_logging(save_dir, model_name, is_training=True):

    log_file_name = str(model_name)
    if is_training:
        log_file_name = model_name + "_train_{time}.log"
    else:
        log_file_name = model_name + "_test_{time}.log"

    log_file_name = os.path.join(os.path.abspath(save_dir), log_file_name)
    logger.add(log_file_name)
