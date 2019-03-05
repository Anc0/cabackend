from django.db import models
from django.contrib.auth.models import User


class Seance(models.Model):

    start = models.DateTimeField()
    end = models.DateTimeField()

    # User Foreign key
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # Flag that tells if the seance is already finished
    active = models.BooleanField(default=True)

    # Times relating to database manipulation
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)