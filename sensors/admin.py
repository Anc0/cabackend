from django.contrib import admin

from sensors.models import Sensor, SensorRecord

admin.site.register(Sensor)
admin.site.register(SensorRecord)
