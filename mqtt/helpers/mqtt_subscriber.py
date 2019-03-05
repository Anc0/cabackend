import logging

import paho.mqtt.client as mqtt

from django.conf import settings

logger = logging.getLogger(__name__)


class MqttClient:

    def __init__(self, client_id=settings.MQTT_CLIENT_ID, host_ip=settings.MQTT_HOST_IP,
                 host_port=settings.MQTT_HOST_PORT, keepalive=settings.MQTT_KEEPALIVE, topic=settings.MQTT_TOPIC,
                 qos=settings.MQTT_QOS, persistent=settings.MQTT_PERSISTENT, retry_first_connection=settings.MQTT_RETRY_FIRST_CONNECTION):

        self.client_id = client_id
        self.host_ip = host_ip
        self.host_port = host_port
        self.keepalive = keepalive
        self.topic = topic
        self.qos=qos
        self.persistent = persistent
        self.retry_first_connection = retry_first_connection

    @staticmethod
    def on_connect(mqttc, obj, flags, rc):
        logger.info("rc: " + str(rc))

    @staticmethod
    def on_message(mqttc, obj, msg):
        logger.error("Message received")
        print("Message received")
        # topic_name = str(msg._topic).split("'")[1]
        # try:
        #     topic = Topic.objects.get(name=topic_name)
        # except:
        #     topic = Topic.objects.create(name=topic_name)
        # logger.info(topic)
        # topic_record = TopicRecord(value=float(str(msg.payload).split("'")[1]), topic=topic)
        # topic_record.save()
        # topic.set_last_record(topic_record)

    @staticmethod
    def on_subscribe(mqttc, obj, mid, granted_qos):
        logger.info("Subscribed: " + str(mid) + " " + str(granted_qos))

    @staticmethod
    def on_log(mqttc, obj, level, string):
        logger.info(string)

    def run(self):
        logger.error("ASDFASDFASDFASDFADSF")
        # If you want to use a specific client id, use
        # but note that the client id must be unique on the broker. Leaving the client
        # id parameter empty will generate a random id for you.
        mqttc = mqtt.Client(self.client_id, clean_session=self.persistent)

        mqttc.on_message = self.on_message
        mqttc.on_connect = self.on_connect
        mqttc.on_subscribe = self.on_subscribe

        mqttc.connect(self.host_ip, self.host_port, self.keepalive)
        mqttc.subscribe(self.topic, self.qos)

        mqttc.loop_forever(retry_first_connection=self.retry_first_connection)
        logger.error("Mqtt subscriber disconnected,")
