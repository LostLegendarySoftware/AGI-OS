import logging
import os
from datetime import datetime
from pathlib import Path

class ARIELLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up file handlers with rotation
        self.setup_logger("training", "training.log")
        self.setup_logger("security", "security.log")
        self.setup_logger("performance", "performance.log")
        self.setup_logger("audit", "audit.log")
        self.setup_logger("error", "error.log")
    
    def setup_logger(self, name, filename):
        logger = logging.getLogger(f"ariel.{name}")
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_dir / filename)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        setattr(self, f"{name}_logger", logger)
    
    def log(self, level, category, message):
        logger = getattr(self, f"{category}_logger")
        logger.log(level, message)
    
    def info(self, category, message):
        self.log(logging.INFO, category, message)
    
    def warning(self, category, message):
        self.log(logging.WARNING, category, message)
    
    def error(self, category, message):
        self.log(logging.ERROR, category, message)
    
    def audit(self, action, user, details):
        self.info("audit", f"{user} | {action} | {details}")