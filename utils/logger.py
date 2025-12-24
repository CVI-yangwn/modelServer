import os
import sys
import logging
import functools
from termcolor import colored
from datetime import datetime

_current_dir = os.path.abspath(os.path.dirname(__file__))

def set_logger(name="trans"):
    current_datetime = datetime.now()
    current_ymd = current_datetime.strftime("%Y-%m-%d")
    output_path = os.path.join(_current_dir, f"logs/{current_ymd}/{name}.txt")
    return create_logger(output_path, name=name)


@functools.lru_cache(maxsize=10)
def create_logger(output_path, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    # if dist_rank == 0:
    #     console_handler = logging.StreamHandler(sys.stdout)
    #     console_handler.setLevel(logging.DEBUG)
    #     console_handler.setFormatter(
    #         logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    #     logger.addHandler(console_handler)

    # create file handlers
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_handler = logging.FileHandler(output_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    logger.info("---create logger success---")
    return logger

def logger_output(logger, output, name):
    logger.debug(f"output data:  {output}")
