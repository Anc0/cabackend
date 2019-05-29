import logging
from datetime import datetime
from os import rename, remove

import pytz
from celery import shared_task
from django.conf import settings

from seances.models import Seance
from sensors.models import Sensor, SensorRecord

logger = logging.getLogger(__name__)


@shared_task
def insert_data_from_csv():
    """
    Move the cache.csv file to a tmp location, so it does not interfere with writing to it.
    Read the cache.csv file line by line, construct a list of sensor records and bulk create them.
    """
    try:
        rename(settings.CACHE_FILE, settings.TMP_CACHE_FILE)
    except FileNotFoundError:
        logger.warning("Cache file doesn't exists. Probably there is no new data.")
        return

    with open(settings.TMP_CACHE_FILE, "r") as cache:
        records = []
        line = cache.readline()
        while line:
            # Extract arguments from the csv line
            arguments = line.split(",")
            timestamp = pytz.utc.localize(datetime.strptime(arguments[0], "%Y-%m-%d %H:%M:%S.%f"))
            topic = arguments[1]
            value = float(arguments[2])
            seance_id = int(arguments[3])
            # Get the seance from seance id
            seance = Seance.objects.get(id=seance_id)
            # Get or create the sensor
            sensor, created = Sensor.objects.get_or_create(topic=topic)
            # Create and append current SensorRecord
            records.append(SensorRecord(sensor=sensor, seance=seance, value=value, timestamp=timestamp))
            line = cache.readline()
    # Save all SensorRecords at once
    SensorRecord.objects.bulk_create(records)

    remove(settings.TMP_CACHE_FILE)
