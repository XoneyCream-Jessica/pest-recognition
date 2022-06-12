import logging
import common.file as common_file
import os

__is_init__ = False


def init():
    global __is_init__
    if __is_init__ is not True:
        __is_init__ = True
        logging.basicConfig(level=logging.INFO, filename=os.path.join(common_file.get_log_path(), 'log.txt'),
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def get_logger(name):
    init()
    logger = logging.getLogger(name)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger
