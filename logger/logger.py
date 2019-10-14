import os
from loguru import logger


def setup_logging(save_dir, model_name, timestamp, is_training=True):
    """[summary]

    Parameters
    ----------
    save_dir : [pathlib.PosixPath]
        [description]: the absolute of the directory to save log file
    model_name : [string]
        [description]
    timestamp : [string]
        [description]
    is_training : bool, optional
        [description], by default True
    """
    log_file_name = str(model_name)
    if is_training:
        log_file_name = "Train_" + \
            str(model_name)+"_" + str(timestamp) + ".log"
    else:
        log_file_name = "Test_" + \
            str(model_name) + "_" + str(timestamp)+".log"

    log_file_name = save_dir.joinpath(log_file_name)
    logger.add(str(log_file_name))
