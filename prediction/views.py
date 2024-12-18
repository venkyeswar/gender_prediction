import tensorflow as tf
import numpy as np
from django.shortcuts import render
from .forms import ImageForm
from .models import Image
from tensorflow.keras.preprocessing import image as keras_image

import os
from keras.models import load_model
from django.conf import settings

# Load the model once when the module is imported
model_path = os.path.join(settings.BASE_DIR, 'gender_prediction.h5')
model = load_model(model_path)
# Absolute path to the model file


def predict_gender(img_path):
    # Load and preprocess the image
    img = keras_image.load_img(img_path, target_size=(224, 224))  # Adjust size as needed
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if required by your model
    prediction = model.predict(img_array)
    return 'Female' if prediction[0][0] > 0.5 else 'Male'

def upload_image(request):
    gender = None  # Default to None, which means no prediction yet
    uploaded_image_url = None  # To hold the uploaded image's URL
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()

            # Predict gender
            img_path = image.image.path
            gender = predict_gender(img_path)
            uploaded_image_url = image.image.url  # URL of the uploaded image

    else:
        form = ImageForm()

    return render(request, 'prediction/upload.html', {'form': form, 'gender': gender, 'uploaded_image_url': uploaded_image_url})
