# Generated by Django 4.1.1 on 2022-10-05 03:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('tsOcr', '0002_ocr_text'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='ocr',
            name='text',
        ),
    ]
