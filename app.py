from flask import Flask, request, render_template, redirect, url_for, Response
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from db import db_init, db
from models import Img

app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Directory to store uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# File size limit for uploaded images (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Initialize the database
db_init(app)

# Load the pre-trained VGG16 model
MODEL_PATH = 'vgg16_model/gender_prediction.h5'
model = load_model(MODEL_PATH)

# Preprocessing function
def preprocess_image(filepath):
    """Preprocess the image for VGG16 model."""
    img = load_img(filepath, target_size=(224, 224))  # Resize to match VGG16 input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Predict function
def predict_gender(filepath):
    """Predict gender using the pre-trained model."""
    processed_img = preprocess_image(filepath)
    prediction = model.predict(processed_img)
    prediction = np.argmax(prediction)
    labels=["Male","Female"]
    prediction = labels[prediction]
    return prediction
   

# Route for image upload and prediction
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Retrieve the uploaded image
        pic = request.files.get('pic')
        if not pic:
            return 'No image uploaded!', 400

        # Secure the filename
        filename = secure_filename(pic.filename)
        mimetype = pic.mimetype

        # Validate filename and mimetype
        if not filename or not mimetype:
            return 'Invalid upload!', 400

        # Save the file to the server
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pic.save(filepath)

        # Run the model for prediction
        prediction = predict_gender(filepath)

        # Save metadata and prediction in the database
        img = Img(filename=filename, mimetype=mimetype, filepath=filepath, prediction=prediction)
        db.session.add(img)
        db.session.commit()

        return redirect(url_for('view_result', img_id=img.id))

    return render_template('index.html')

# Route to view prediction result
@app.route('/result/<int:img_id>')
def view_result(img_id):
    img = Img.query.filter_by(id=img_id).first()
    if not img:
        return 'Image not found!', 404

    return render_template('index.html', image=img)

# Route to serve the image file
@app.route('/static/uploads/<filename>')
def serve_image(filename):
    return Response(open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb').read(), mimetype='image/jpeg')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
