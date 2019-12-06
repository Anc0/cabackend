from django.contrib import admin

from sensors.models import Sensor, SensorRecord


class SensorRecordAdmin(admin.ModelAdmin):
    list_display = ["sensor", "value", "timestamp"]
    search_fields = ["sensor__topic"]


admin.site.register(Sensor)
admin.site.register(SensorRecord, SensorRecordAdmin)
