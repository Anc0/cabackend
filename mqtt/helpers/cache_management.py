import logging
from datetime import datetime
from os.path import isfile

from django.conf import settings

logger = logging.getLogger(__name__)


class CacheManagement:

    def __init__(self):
        self.file = None

    def __del__(self):
        self.file.close()

    def save_record(self, timestamp: datetime, topic: str, value: float, seance_id: int):
        if not isfile(settings.CACHE_FILE):
            if self.file:
                self.file.close()
            self.file = open("/home/andraz/Projects/cabackend/mqtt/helpers/cache.csv", "a")
        self.file.write(",".join([timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"), topic, str(value), str(seance_id)]) + "\n")

