# Generated by Django 2.1.7 on 2019-07-05 14:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sensors', '0004_auto_20190305_1428'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sensor',
            name='type',
            field=models.CharField(choices=[('AC', 'Accelerometer'), ('FO', 'Force sensor'), ('MI', 'Microphone'), ('PI', 'Passive IR'), ('HA', 'Hall'), ('UN', 'Undefined')], default='UN', max_length=2),
        ),
    ]