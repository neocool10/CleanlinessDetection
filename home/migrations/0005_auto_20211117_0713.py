# Generated by Django 3.2.9 on 2021-11-17 07:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_auto_20211117_0555'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='contact',
            name='content',
        ),
        migrations.RemoveField(
            model_name='contact',
            name='email',
        ),
        migrations.RemoveField(
            model_name='contact',
            name='name',
        ),
        migrations.RemoveField(
            model_name='contact',
            name='phone',
        ),
        migrations.RemoveField(
            model_name='plus',
            name='file',
        ),
        migrations.AddField(
            model_name='contact',
            name='video',
            field=models.FileField(null=True, upload_to='videos/'),
        ),
    ]
