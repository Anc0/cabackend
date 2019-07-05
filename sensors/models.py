from django.db import models

from seances.models import Seance


class Sensor(models.Model):
    TYPE_ACCELEROMETER = 'AC'
    TYPE_FORCE = 'FO'
    TYPE_MICROPHONE = 'MI'
    TYPE_PIR = 'PI'
    TYPE_HALL = 'HA'
    TYPE_UNDEFINED = 'UN'

    TYPES = (
        (TYPE_ACCELEROMETER, 'Accelerometer'),
        (TYPE_FORCE, 'Force sensor'),
        (TYPE_MICROPHONE, 'Microphone'),
        (TYPE_PIR, 'Passive IR'),
        (TYPE_HALL, 'Hall'),
        (TYPE_UNDEFINED, 'Undefined'),
    )

    # User defined sensor name
    name = models.CharField(max_length=255, default="Unnamed sensor")
    # Type of sensor
    type = models.CharField(max_length=2, choices=TYPES, default=TYPE_UNDEFINED)
    # Topic to which the sensor publishes data (unique to each sensor)
    topic = models.CharField(max_length=255, unique=True)

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "Sensor {}".format(self.topic)


class SensorRecord(models.Model):
    # Big integer for id
    id = models.BigAutoField(unique=True, primary_key=True)

    # Sensor and session foreign keys
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    seance = models.ForeignKey(Seance, on_delete=models.CASCADE)

    value = models.FloatField()

    # Time of record
    timestamp = models.DateTimeField()

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "Value: {}, from sensor {}, at {}".format(self.value, self.sensor.topic,
                                                         self.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
