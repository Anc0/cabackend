# Generated by Django 2.1.7 on 2019-03-05 13:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('seances', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Sensor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('type', models.CharField(choices=[('AC', 'Accelerometer'), ('FO', 'Force sensor'), ('MI', 'Microphone')], max_length=2)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='SensorRecord',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False, unique=True)),
                ('timestamp', models.DateTimeField()),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
                ('seance', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='seances.Seance')),
                ('sensor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sensors.Sensor')),
            ],
        ),
    ]