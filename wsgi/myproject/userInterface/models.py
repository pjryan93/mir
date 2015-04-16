from django.db import models
from django.contrib.auth.models import Permission, User

# Create your models here.

class SongInterface(models.Model):
    title = models.CharField(max_length=50)
    fileIn = models.FileField() 
    jobOwner = models.ForeignKey(User)

"""class Result(models.Model)
	values = modes.charField(max_length=700)
	resultOwner = modles.ForeignKey(User)"""
