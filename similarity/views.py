# similarity/views.py

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pickle
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image
import io
import os

# Load filenames and feature_list once and store them in global variables
filenames = pickle.load(open('similarity/data/postkarten.pickle', 'rb'))
feature_list = pickle.load(open('similarity/data/postkarten-resnet.pickle', 'rb'))

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')

def upload_image(request):
    return render(request, 'upload.html')  # Render the upload form

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        
        # Open the uploaded image using PIL
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))  # Resize to match model input size
        image_array = img_to_array(img)
        expanded_image_array = np.expand_dims(image_array, axis=0)
        processed_image = preprocess_input(expanded_image_array)

        features = model.predict(processed_image)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)

        distances, indices = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list).kneighbors([normalized_features])
        similar_image_paths = [filenames[indices[0][i]] for i in range(len(indices[0]))]

        # Assuming your images are stored in a directory accessible via a URL
        image_urls = [os.path.join(settings.MEDIA_URL, 'dataset', os.path.basename(path)) for path in similar_image_paths]

        return JsonResponse({'similar_images': image_urls})

    return JsonResponse({'error': 'Invalid request'}, status=400)
