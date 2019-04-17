from datetime import datetime
import logging

import paho.mqtt.client as mqtt
import pytz

from django.conf import settings

from seances.models import Seance
from mqtt.tasks import insert_data
from users.models import UserProfile


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
        self.qos = qos
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
            user_rfid = str(msg.payload)
            self.initialize_seance(user_rfid)
        elif topic == 'deactivate':
            user_rfid = str(msg.payload)
            self.complete_seance(user_rfid)
        else:
            if self.seance:
                try:
                    value = float(msg.payload)
                    # Insert the data value in the queue
                    if settings.USE_QUEUE:
                        insert_data.delay(datetime.now(tz=pytz.UTC), topic, value, self.seance.id)
                    else:
                        insert_data(datetime.now(tz=pytz.UTC), topic, value, self.seance.id)
                # Except general exceptions as we do not want to crash the mqtt listener at any point
                except Exception as e:
                    logger.error(e)
            else:
                logger.warning("Recording sensor record for sensor {} outside of an active seance.".format(topic))

        logger.info("######################################")

    def initialize_seance(self, rfid):
        """
        Start new seance.
        """
        logger.info("Initializing seance...")
        if self.seance:
            logger.error("Seance already initiated. Not starting another one.")
            return

        rfid = rfid.split("'")[1]
        try:
            user = UserProfile.objects.get(rfid=rfid).user

            if len(Seance.objects.filter(user=user, active=True)) > 0:
                logger.error("Seance already initialized. Not starting another one.")
                return

            seance = Seance(user=user, start=datetime.now(tz=pytz.UTC))
            seance.save()

            self.seance = seance

            logger.info("Seance for user {} initialized.".format(user.username))
        except UserProfile.DoesNotExist as e:
            logger.error("User with rfid {} not found.".format(rfid))
            logger.info("Seance not initialized.")
        except Exception as e:
            logger.error(e)
            logger.info("Seance not initialized.")

    def complete_seance(self, rfid):
        """
        Finish seance.
        """
        rfid = rfid.split("'")[1]
        try:
            user = UserProfile.objects.get(rfid=rfid).user
            seance = Seance.objects.get(user=user, active=True)

            logger.info("Deactivating seance {}...".format(seance))
            seance.end_seance()
            self.seance = None
            logger.info("Seance deactivated.")

        except UserProfile.DoesNotExist as e:
            logger.error("User with rfid {} does not exist.".format(rfid))
        except Seance.DoesNotExist as e:
            logger.error("No active seance for user with rfid {}.".format(rfid))
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
