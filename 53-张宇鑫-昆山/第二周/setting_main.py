import os
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR,'src')
STATIC_DIR = os.path.join(SRC_DIR,'static')
IMAGE_DIR = os.path.join(STATIC_DIR,'img')


LOG_PATH = os.path.join(BASE_DIR,"runtime1.log")
logger.add(LOG_PATH, rotation="5 MB",encoding='UTF-8')
LOG = logger
