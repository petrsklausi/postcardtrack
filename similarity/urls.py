from django.urls import path
from .views import classify_image, upload_image

urlpatterns = [
    path('classify/', classify_image, name='classify_image'),
    path('upload/', upload_image, name='upload_image'),  # URL for the upload form
]