# Generated by Django 4.1.1 on 2022-10-05 06:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tsOcr', '0003_remove_ocr_text'),
    ]

    operations = [
        migrations.AddField(
            model_name='ocr',
            name='name',
            field=models.CharField(default='name', max_length=50),
            preserve_default=False,
        ),
    ]