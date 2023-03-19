from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now

class Contact(models.Model):
    sno= models.AutoField(primary_key=True)
    video = models.FileField(upload_to='videos/', null=True)

    timeStamp=models.DateTimeField(auto_now_add=True, blank=True)

    # def __str__(self):
    #     return "Message from " + self.name + ' - ' + self.email
    
    
    
class Plus(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to='images/', null=True)
    
    slug=models.CharField(max_length=130)
    timeStamp=models.DateTimeField(auto_now_add=True,blank=True)


    # def __str__(self):
    #     return self.title + " by " + self.name
