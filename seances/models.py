import pytz
from datetime import datetime

from django.db import models
from django.contrib.auth.models import User


class Experiment(models.Model):
    # Field that determines the sequence of experiments
    sequence_number = models.IntegerField(unique=True)
    instructions = models.TextField()

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)


class Seance(models.Model):
    # Start and end time of each seance
    start = models.DateTimeField()
    end = models.DateTimeField(default=None, null=True)

    # User Foreign key
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # Flag that tells if the seance is already finished
    active = models.BooleanField(default=True)

    # This field marks if the seance can be used for further analysis
    valid = models.BooleanField(default=True)

    # If there was anything extraordinary with this session, mark it here
    notes = models.TextField(default=None, blank=True, null=True)

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, default=None, null=True, blank=True)
    try_count = models.IntegerField(default=-1)

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def end_seance(self):
        self.end = datetime.now(tz=pytz.UTC)
        self.active = False
        self.save()

    def iterate_experiment(self):
        valid_seance_count = len(Seance.objects.filter(user=self.user, valid=True))
        self.try_count = int(valid_seance_count/3) + 1
        self.experiment = Experiment.objects.get(sequence_number=(valid_seance_count + 1) % len(Experiment.objects.all()))
        self.save()

    def __str__(self):
        if self.active:
            return "Active seance started at: {} with user {}".format(self.start.strftime("%Y-%m-%d %H:%M:%S"),
                                                                       self.user.username)
        else:
            return "Completed seance started at: {} with user {}".format(self.start.strftime("%Y-%m-%d %H:%M:%S"),
                                                                          self.user.username)

