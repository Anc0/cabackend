# Generated by Django 2.1.7 on 2019-09-12 19:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("seances", "0005_auto_20190313_0812"),
    ]

    operations = [
        migrations.AddField(
            model_name="seance",
            name="notes",
            field=models.TextField(blank=True, default=None, null=True),
        ),
    ]
