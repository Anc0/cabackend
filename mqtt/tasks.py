import logging
from datetime import datetime

from celery import shared_task

from seances.models import Seance
from sensors.models import Sensor, SensorRecord

logger = logging.getLogger(__name__)


@shared_task
def insert_data(timestamp: datetime, topic, value, seance_id):
    """
    Retrieve sensor from topic and create new sensor record with value.
    """
    try:
        logger.info("Saving sensor record...")

        seance = Seance.objects.get(id=seance_id)

        sensor, created = Sensor.objects.get_or_create(topic=topic)

        if created:
            logger.info("New sensor created ({}).".format(sensor.topic))
        else:
            logger.info("Using sensor {}.".format(sensor.topic))

        sensor_record = SensorRecord(sensor=sensor, seance=seance, value=value, timestamp=timestamp)
        sensor_record.save()

        if sensor_record.id:
            logger.info("Sensor record saved.")
        else:
            logger.error("Something went wrong when saving sensor record.")

    except Exception as e:
        logger.error(e)
