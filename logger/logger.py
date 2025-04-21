import logging
import os
import sys
from datetime import datetime

class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ""

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.log_level, message)

    def flush(self):
        pass

def setup_logger(log_dir="logs", logger_name="training_logger"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{start_time}.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 同时将stdout和stderr重定向到logger
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger