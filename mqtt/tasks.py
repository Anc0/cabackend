from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from paho.mqtt.publish import single

from seances.models import Seance
from sensors.models import Sensor, SensorRecord

logger = get_task_logger(__name__)


@shared_task
def insert_data_from_buffer(cache_records):
    """
    Create a list of sensor records from buffer and bulk insert them into the database.
    """
    print("Processing cached records...")
    records = []
    topics = {}
    for record in cache_records:
        # Extract data into separate variables for clarity
        timestamp = record[0]
        topic = record[1]
        value = record[2]
        seance_id = record[3]
        # Get the seance from seance id
        seance = Seance.objects.get(id=seance_id)
        # Get or create the sensor
        sensor, created = Sensor.objects.get_or_create(topic=topic)
        # Create and append current SensorRecord
        records.append(
            SensorRecord(sensor=sensor, seance=seance, value=value, timestamp=timestamp)
        )
        # Update topics counter
        if topic in topics:
            topics[topic] += 1
        else:
            topics[topic] = 1
    # Save all SensorRecords at once
    result = SensorRecord.objects.bulk_create(records)
    print("{} sensor records inserted.".format(len(result)))
    if len(cache_records) > 0:
        for k, v in zip(topics.keys(), topics.values()):
            print("Inserted {} records with topic {}.".format(v, k))


@shared_task
def dump_data():
    single(
        topic="cabackend/dump",
        qos=settings.MQTT_QOS,
        hostname=settings.MQTT_HOST_IP,
        port=settings.MQTT_HOST_PORT,
        client_id="periodic_dump",
        keepalive=6 * 60,
    )
