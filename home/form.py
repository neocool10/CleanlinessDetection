from django import forms
from django.db.models import fields
from .models import Plus
from .models import Contact


# Create your forms here.
class ImageForm(forms.ModelForm):

    class Meta:
        model = Plus
        fields = ('image')
class VideoForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ('video')