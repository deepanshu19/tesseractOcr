from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'imageupload', views.OcrToText)

urlpatterns = [
    path('', views.index, name='index'),
    # path('ocr/', views.OcrToText.as_view(), name='ocrToText'),
    path('imageupload/', include(router.urls)),
    path('getData/', views.getData, name="getData"),
    path('getImage/<str:name1>', views.getImage, name="getImage"),
]
