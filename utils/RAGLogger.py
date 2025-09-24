import logging
import json
from datetime import datetime
from pathlib import Path

class RAGLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self._setup_basic_handlers()

    def _setup_basic_handlers(self):
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)

        # File handler (INFO)
        handler = logging.FileHandler('logs/app.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s ')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Console handler (DEBUG) - For development use only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter=logging.Formatter('%(levelname)s | %(name)s | %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def info(self, message:str, **data):
        if data:
            message = f"{message} | Data: {json.dumps(data)}"
        self.logger.info(message)
    
    def error(self, message:str, **data):
        if data:
            message = f"{message} | Data: {json.dumps(data)}"
        self.logger.error(message)

    def warning(self, message:str, **data):
        if data:
            message = f"{message} | Data: {json.dumps(data)}"
        self.logger.warning(message)

    def debug(self, message: str, **data):
        if data:
            message = f"{message} | Data: {json.dumps(data)}"
        self.logger.debug(message)