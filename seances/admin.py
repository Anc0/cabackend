from django.contrib import admin

from seances.models import Seance


class SeanceAdmin(admin.ModelAdmin):
    list_display = ['start', 'end', 'user', 'valid']


admin.site.register(Seance, SeanceAdmin)
