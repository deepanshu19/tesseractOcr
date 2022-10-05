from dataclasses import fields
from rest_framework import serializers
from .models import Ocr


class ocrSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ocr
        fields = "__all__"


class ocrImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ocr
        fields = ['image']
