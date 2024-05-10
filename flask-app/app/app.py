from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
from sklearn import svm
from LPQ import *
from base64 import b64encode
import io
from PIL import Image

# Load your trained SVM model
model = joblib.load(R"D:\Downloads\College\Neural Networks\Project\archive\Michael\flask-app\app\models\SVM.pkl")

app = Flask(__name__)

classes = ['IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']

def preprocess_image(image, desired_size=(256, 256)):
    img = cv2. cvtColor(image, cv2. COLOR_BGR2GRAY)

    # Apply median filter to remove salt and pepper noise
    denoised_img = cv2.medianBlur(img, 3)

    # Threshold the image using Otsu's method
    _, thresh_img = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Check the color of the background at all four corners
    corners = [thresh_img[0, 0], thresh_img[0, -1], thresh_img[-1, 0], thresh_img[-1, -1]]
    white_corners = np.sum(np.array(corners) == 255)

    # If the majority of the corners are white, invert the image to make the background black
    if white_corners > 2:
        thresh_img = cv2.bitwise_not(thresh_img)

    # Resize the image to the desired size
    resized_img = cv2.resize(thresh_img, desired_size, interpolation=cv2.INTER_AREA)

    return resized_img

@app.route('/', methods=['GET'])
def upload_file():
    return render_template('upload.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Read the image file
        image_stream = file.read()
        # Convert the image file to a numpy array
        npimg = np.frombuffer(image_stream, np.uint8)
        # Convert numpy array to image
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Preprocess the image here
        processed_image = preprocess_image(image)
        lpq_features = lpq(processed_image).reshape(1, -1)

        predicted_class = model.predict(lpq_features)[0]

        # Make a prediction
        prediction = classes[predicted_class - 1]

        # Convert the image to base64
        img = Image.fromarray(image.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = b64encode(rawBytes.getvalue()).decode('ascii')
        mime = "image/jpeg"
        uri = "data:%s;base64,%s" % (mime, img_base64)

        # Pass the base64 string and prediction to the template
        return render_template('result.html', image_data=uri, label=prediction)

if __name__ == '__main__':
    app.run(debug=True)
