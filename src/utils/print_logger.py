

import sys
import logging
import logging.handlers


class StreamToLogger(object):

    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def clear_loggers():
    logger = logging.getLogger('MyLogger')
    if (logger.hasHandlers()):
        logger.handlers.clear()


def initialize_logger(file_name):

    cfg = {
        "log_name": file_name,
        "log_max_bytes": 512 * 1024 * 1024,
        "log_backup_count": 2,
    }

    LOG_FILENAME = cfg["log_name"]
    my_logger = logging.getLogger('MyLogger')
    my_logger.setLevel(logging.DEBUG)

    handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=cfg["log_max_bytes"],
        backupCount=cfg["log_backup_count"]
    )

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)

    my_logger.addHandler(handler)

    sys.stdout = StreamToLogger(my_logger, logging.INFO)
    sys.stderr = StreamToLogger(my_logger, logging.ERROR)


def getLogger():
    logger_handler = logging.getLogger('MyLogger')
    return logger_handler

