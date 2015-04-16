# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django import forms
from django.forms.formsets import BaseFormSet, formset_factory
from django.contrib.auth.models import Permission, User

from bootstrap3.tests import TestForm


class LoginForm(forms.Form):
    emailAddress = forms.EmailField(label = 'Email')
    password = forms.CharField(widget=forms.PasswordInput())
class UserForm(forms.Form):
    class Meta:
        model = User
        fields = ('email','password')
class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)
class ContactForm(TestForm):
    pass

class FilesForm(forms.Form):
    file = forms.FileField(required=False, widget=forms.ClearableFileInput)


class ArticleForm(forms.Form):
    title = forms.CharField()
    pub_date = forms.DateField()

    def clean(self):
        cleaned_data = super(ArticleForm, self).clean()
        raise forms.ValidationError("This error was added to show the non field errors styling.")
        return cleaned_data