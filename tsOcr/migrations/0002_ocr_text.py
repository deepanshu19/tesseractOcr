# Generated by Django 4.1.1 on 2022-10-05 03:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tsOcr', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='ocr',
            name='text',
            field=models.TextField(blank=True),
        ),
    ]
