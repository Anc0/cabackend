# Generated by Django 2.1.7 on 2019-03-10 09:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("seances", "0002_seance_active"),
    ]

    operations = [
        migrations.AlterField(
            model_name="seance", name="end", field=models.DateTimeField(default=None),
        ),
        migrations.AlterField(
            model_name="seance",
            name="start",
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
