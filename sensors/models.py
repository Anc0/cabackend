from django.db import models

from seances.models import Seance


class Sensor(models.Model):
    TYPE_ACCELEROMETER = 'AC'
    TYPE_FORCE = 'FO'
    TYPE_MICROPHONE = 'MI'

    TYPES = (
        (TYPE_ACCELEROMETER, 'Accelerometer'),
        (TYPE_FORCE, 'Force sensor'),
        (TYPE_MICROPHONE, 'Microphone')
    )

    # User defined sensor name
    name = models.CharField(max_length=255)
    # Type of sensor
    type = models.CharField(max_length=2, choices=TYPES)

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)


class SensorRecord(models.Model):
    # Big integer for id
    id = models.BigAutoField(unique=True, primary_key=True)

    # Sensor and session foreign keys
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    seance = models.ForeignKey(Seance, on_delete=models.CASCADE)

    # Time of record
    timestamp = models.DateTimeField()

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
