from django.contrib import admin

from sensors.models import Sensor, SensorRecord


class SensorRecordAdmin(admin.ModelAdmin):
    search_fields = ['sensor__topic']


admin.site.register(Sensor)
admin.site.register(SensorRecord, SensorRecordAdmin)
