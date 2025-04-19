import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("Could not read the image.")
        return "Error: Image not readable"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = np.array(img) / 255.0  # Normalize if model was trained with normalized images
    input_img = np.expand_dims(img, axis=0)

    # Updated prediction logic
    prediction = model.predict(input_img)

    # For binary classification (1 output neuron with sigmoid activation)
    result = int((prediction > 0.5).astype("int32")[0][0])
    
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
