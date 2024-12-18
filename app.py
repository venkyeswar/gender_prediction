import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Folder Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the VGG16 model
model = load_model('model/gender_prediction.h5')

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_gender(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match VGG16 input
    img_array = img_to_array(img) / 255.0              # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
    prediction = model.predict(img_array)
    print(prediction)
    prediction = np.argmax(prediction)
    labels=["Male","Female"]
    prediction=labels[prediction]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is in the request
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If no file is selected or the file is not allowed
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the file
            
            # Make a prediction
            prediction = predict_gender(file_path)
            return render_template('index.html', prediction=prediction, uploaded_image=file_path)

    return render_template('index.html', prediction=None, uploaded_image=None)
            

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

