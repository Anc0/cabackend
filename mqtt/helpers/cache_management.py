import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheManagement:

    def __init__(self):
        self.buffer = []

    def save_record(self, timestamp: datetime, topic: str, value: float, seance_id: int):
        self.buffer.append((timestamp, topic, value, seance_id))

    def dump_buffer(self):
        tmp_buffer = self.buffer
        self.buffer = []
        return tmp_buffer
