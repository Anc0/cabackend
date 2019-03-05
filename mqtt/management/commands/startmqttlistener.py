import logging

from django.core.management.base import BaseCommand

from mqtt.helpers.mqtt_subscriber import MqttClient

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Starts the mqtt listener'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        logger.error("ASADADSFASDF")
        MqttClient().run()
