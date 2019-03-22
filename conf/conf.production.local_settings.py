import os

MQTT_HOST_IP = "192.168.4.1"

DOTENV_PATH = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.env'), '.env')
