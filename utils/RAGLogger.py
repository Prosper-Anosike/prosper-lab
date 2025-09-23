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

        # Single file handler to start
        handler = logging.FileHandler('logs/app.log')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s ')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
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

