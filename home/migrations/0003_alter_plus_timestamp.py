# Generated by Django 3.2 on 2021-07-30 10:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_plus'),
    ]

    operations = [
        migrations.AlterField(
            model_name='plus',
            name='timeStamp',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]