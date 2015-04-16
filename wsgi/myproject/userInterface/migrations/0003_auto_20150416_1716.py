# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('userInterface', '0002_auto_20150416_1711'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='job',
            name='jobUser',
        ),
        migrations.AlterField(
            model_name='song',
            name='jobOwner',
            field=models.ForeignKey(to=settings.AUTH_USER_MODEL),
        ),
        migrations.DeleteModel(
            name='job',
        ),
    ]
