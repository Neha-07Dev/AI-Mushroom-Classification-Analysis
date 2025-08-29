# importing libraries
from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define model path and initialize model
model_path = "Mushroom Classification Model.h5"
model = None

# Load our saved model
def load_mushroom_model():
    global model
    model = load_model(model_path)

# Load the model before app starts
load_mushroom_model()

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Routes
@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/Mushroom-classification-predict")
def predict_page():
    return render_template("input.html")


@app.route("/Mushroom-classification-predict", methods=["POST"])
def predict():
    try:
        imageFile = request.files["image_file"]
        if imageFile.filename == '':
            return render_template("input.html", prediction="No file selected.")

        imagePath = os.path.join(UPLOAD_FOLDER, imageFile.filename)
        imageFile.save(imagePath)

        # Preprocess the image
        inputImage = image.load_img(imagePath, target_size=(224, 224))
        inputImage = image.img_to_array(inputImage)
        inputImage = np.expand_dims(inputImage, axis=0)
        inputImage = inputImage / 255.0

        # Predict the class
        prediction = model.predict(inputImage)
        predicted_class_index = np.argmax(prediction)
        class_names = ['Boletus', 'Lactarius', 'Russula']
        predicted_class = class_names[predicted_class_index]

        return render_template("input.html", prediction=predicted_class)

    except Exception as e:
        return render_template("input.html", prediction=f"Error: {str(e)}")

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
