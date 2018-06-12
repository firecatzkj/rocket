# -*- coding:utf-8 -*-
import logging

# logging.basicConfig(level=logging.INFO, format='[%(levelname)s]:  %(asctime)s - %(name)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]:  %(asctime)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s]:  %(asctime)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='[%(levelname)s]:  %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

