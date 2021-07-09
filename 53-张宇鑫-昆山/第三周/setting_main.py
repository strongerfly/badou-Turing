import os
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR,'img')


LOG_PATH = os.path.join(BASE_DIR,"runtime1.log")
logger.add(LOG_PATH, rotation="5 MB",encoding='UTF-8')
LOG = logger
