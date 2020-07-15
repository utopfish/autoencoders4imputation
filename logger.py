#@Time:2019/12/24 17:42
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:logger.py
__author__ = "liuAmon"

import os, sys

import logging
from loguru import logger

# 日志文件配置

log_path = os.path.dirname((os.path.abspath(__file__))) + '/logs/'
_log = logger.add(log_path + '{time:YYYY-MM-DD-HH}h.log',
                  rotation='00:00', retention='15 days', level=logging.INFO)