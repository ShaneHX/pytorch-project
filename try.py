
from loguru import logger
import multiprocessing
multiprocessing.set_start_method('spawn', True)

logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
               "on this machine.".format(5, 2))
