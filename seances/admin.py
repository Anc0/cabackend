from django.contrib import admin

from seances.models import Seance, Experiment


class SeanceValidFilter(admin.SimpleListFilter):
    title = "Is valid"
    parameter_name = "Valid"

    def lookups(self, request, model_admin):
        return [
            ('valid', 'Valid'),
            ('not_valid', 'Not valid')
        ]

    def queryset(self, request, queryset):
        if self.value() == 'valid':
            return queryset.distinct().filter(valid=True)
        elif self.value() == 'not_valid':
            return queryset.distinct().filter(valid=False)


class SeanceAdmin(admin.ModelAdmin):
    list_display = ['start', 'end', 'user', 'valid']
    list_filter = (SeanceValidFilter, )


admin.site.register(Seance, SeanceAdmin)
admin.site.register(Experiment)
