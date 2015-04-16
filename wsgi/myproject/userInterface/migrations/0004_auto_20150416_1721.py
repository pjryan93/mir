# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('userInterface', '0003_auto_20150416_1716'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Song',
            new_name='SongInterface',
        ),
    ]
