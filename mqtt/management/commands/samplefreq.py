import logging
from statistics import stdev

from django.core.management.base import BaseCommand
from numpy import mean

from sensors.models import Sensor, SensorRecord

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Gives sampling frequency stats about a specific sensor.'

    def add_arguments(self, parser):
        parser.add_argument('sensor', type=str)

    def handle(self, *args, **options):
        try:
            sensor = Sensor.objects.get(topic=options["sensor"])
        except Sensor.DoesNotExist:
            print("Sensor does not exist.")

        records = SensorRecord.objects.filter(sensor=sensor).order_by('timestamp')
        deltas = []
        prev = records[0]
        for record in records[1:]:
            delta = (record.timestamp - prev.timestamp).microseconds / 1000
            print(delta)
            deltas.append(delta)
            prev = record
        print("Mean: {} ms".format(round(mean(deltas))))
        print("Standard deviation: {} ms".format(round(stdev(deltas))))

