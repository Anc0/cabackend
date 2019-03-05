from datetime import datetime
import logging

import paho.mqtt.client as mqtt
import pytz

from django.conf import settings
from django.contrib.auth.models import User

from seances.models import Seance
from sensors.models import Sensor, SensorRecord

logger = logging.getLogger(__name__)


class MqttClient:

    def __init__(self, client_id=settings.MQTT_CLIENT_ID, host_ip=settings.MQTT_HOST_IP,
                 host_port=settings.MQTT_HOST_PORT, keepalive=settings.MQTT_KEEPALIVE, topic=settings.MQTT_TOPIC,
                 qos=settings.MQTT_QOS, persistent=settings.MQTT_PERSISTENT,
                 retry_first_connection=settings.MQTT_RETRY_FIRST_CONNECTION):

        self.client_id = client_id
        self.host_ip = host_ip
        self.host_port = host_port
        self.keepalive = keepalive
        self.topic = topic
        self.qos=qos
        self.persistent = persistent
        self.retry_first_connection = retry_first_connection
        self.seance = None

    def on_connect(self, mqttc, obj, flags, rc):
        logger.info("Connected with result code: {}".format(rc))

    def on_message(self, mqttc, obj, msg):
        logger.info("Message received")

        topic = msg.topic.split("/")[1]

        if topic == 'activate':
            value = str(msg.payload)
            self.initialize_seance(value)
        elif topic == 'deactivate':
            value = str(msg.payload)
            self.complete_seance()
        else:
            value = float(msg.payload)
            self.save_record(topic, value)

    def on_subscribe(self, mqttc, obj, mid, granted_qos):
        logger.info("Subscribed to {}:{} as {}.".format(self.host_ip, self.host_port, self.client_id))

    def run(self):
        mqttc = mqtt.Client(self.client_id, clean_session=self.persistent)

        mqttc.on_message = self.on_message
        mqttc.on_connect = self.on_connect
        mqttc.on_subscribe = self.on_subscribe

        mqttc.connect(self.host_ip, self.host_port, self.keepalive)
        mqttc.subscribe(self.topic, self.qos)

        mqttc.loop_forever(retry_first_connection=self.retry_first_connection)
        logger.error("Mqtt subscriber disconnected,")

    @staticmethod
    def initialize_seance(username):
        """
        Start new seance.
        """
        logger.info("Initializing seance...")
        user = User.objects.get(username=username)
        logger.info(Seance(user=user).save())
        logger.info("Seance initialized.")

    def complete_seance(self):
        """
        Finish seance.
        """
        pass

    def save_record(self, topic, value):
        """
        Retrieve sensor from topic and create new sensor record with value.
        """
        logger.info("Saving sensor record...")
        timestamp = datetime.now(tz=pytz.UTC)

        sensor, created = Sensor.objects.get_or_create(topic=topic)

        if created:
            logger.info("New sensor created.")
        else:
            logger.info("Using sensor {}.".format(sensor.name))

        result = SensorRecord(sensor=sensor, seance=self.seance, value=value, timestamp=timestamp).save()

        if result:
            logger.info("Sensor record saved.")
        else:
            logger.error("Something went wrong when saving sensor record.")
