from datetime import datetime
from logging import getLogger

import pytz
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from seances.models import Seance

logger = getLogger(__name__)


class Command(BaseCommand):
    help = "Deletes all seances and sensor record for given user."

    def add_arguments(self, parser):
        """
        Define positional and optional arguments to manage.py call.
        """
        parser.add_argument("user", type=str)

    def handle(self, *args, **options):
        """
        Entry point of our crawler, runs the thread manager.
        """
        try:
            user = User.objects.get(username=options["user"])
        except User.DoesNotExist:
            print("User does not exist.")

        print("Purging...")
        print(Seance.objects.filter(user=user).delete())
        print("Done.")
