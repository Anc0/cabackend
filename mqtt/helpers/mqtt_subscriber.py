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

    def on_subscribe(self, mqttc, obj, mid, granted_qos):
        logger.info("Subscribed to {}:{} as {}.".format(self.host_ip, self.host_port, self.client_id))

    def on_message(self, mqttc, obj, msg):
        """
        Handle received messages - there are 2 special topics that respectively start (activate) and end (deactivate) a
        seance. Otherwise if there is an active seance, save the received message with the topic name representing the
        sensor sending the data.
        """
        logger.info("########## Message received ##########")

        topic = msg.topic.split("/")[1]

        if topic == 'activate':
            user_id = str(msg.payload)
            self.initialize_seance(user_id)
        elif topic == 'deactivate':
            self.complete_seance()
        else:
            if self.seance:
                value = float(msg.payload)
                self.save_record(topic, value)
            else:
                logger.error("Recording sensor record for sensor {} outside of an active seance.".format(topic))

        logger.info("######################################")

    def initialize_seance(self, user_id):
        """
        Start new seance.
        """
        logger.info("Initializing seance...")
        try:
            # TODO: fetch user via user id, that is linked to NFC tag
            user = User.objects.get(username='cabackend')

            seance = Seance(user=user, start=datetime.now(tz=pytz.UTC))
            seance.save()

            self.seance = seance

            logger.info("Seance initialized.")
        except Exception as e:
            logger.error(e)

    def complete_seance(self):
        """
        Finish seance.
        """
        try:
            logger.info("Deactivating seance {}...".format(self.seance))

            if not self.seance:
                logger.error("No active seance.")
                return

            self.seance.end_seance()
            self.seance = None

            logger.info("Seance deactivated.")
        except Exception as e:
            logger.error(e)

    def save_record(self, topic, value):
        """
        Retrieve sensor from topic and create new sensor record with value.
        """
        try:
            logger.info("Saving sensor record...")
            timestamp = datetime.now(tz=pytz.UTC)

            sensor, created = Sensor.objects.get_or_create(topic=topic)

            if created:
                logger.info("New sensor created.")
            else:
                logger.info("Using sensor {}.".format(sensor.topic))

            sensor_record = SensorRecord(sensor=sensor, seance=self.seance, value=value, timestamp=timestamp)
            sensor_record.save()

            if sensor_record.id:
                logger.info("Sensor record saved.")
            else:
                logger.error("Something went wrong when saving sensor record.")

        except Exception as e:
            logger.error(e)

    def run(self):
        """
        Initialize the mqtt client, set callbacks, subscribe to the topic on broker and run indefinitely.
        """
        mqttc = mqtt.Client(self.client_id, clean_session=self.persistent)

        mqttc.on_message = self.on_message
        mqttc.on_connect = self.on_connect
        mqttc.on_subscribe = self.on_subscribe

        mqttc.connect(self.host_ip, self.host_port, self.keepalive)
        mqttc.subscribe(self.topic, self.qos)

        mqttc.loop_forever(retry_first_connection=self.retry_first_connection)

        logger.error("Mqtt subscriber disconnected,")
        if self.seance:
            self.complete_seance()
