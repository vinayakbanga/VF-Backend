from flask import Flask, render_template, request, redirect, url_for, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from flask_cors import CORS
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import cloudinary.uploader


load_dotenv()


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the CNN model
model = load_model('my_cnn_model.h5')

          
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)
# Preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return 'Flask server is running!'


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    img_path = os.path.join(app.root_path, file.filename)
    file.save(img_path)

    img = preprocess_image(img_path)
    result = model.predict(img)
    prediction = "True" if result[0][0] > 0.5 else "False"
    print(result[0][0])

    return jsonify({'result': prediction})

@app.route('/swap', methods=['POST'])
def swap():
    if 'file1' not in request.files or 'file2' not in request.files:
        return "No file part"

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Check if the files are empty
    if file1.filename == '' or file2.filename == '':
        return "No selected files"

    # Check if the files have allowed extensions
    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        # Save the files to the upload folder
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(file1_path)
        file2.save(file2_path)

        # Process the images and swap faces
        res = process_images(file1_path, file2_path)

        # Save the swapped image locally
        swapped_filename = 'swapped_' + filename1
        swapped_file_path = os.path.join(app.config['UPLOAD_FOLDER'], swapped_filename)
        cv2.imwrite(swapped_file_path, res)

        # Upload swapped image to Cloudinary
        cloudinary_upload_result = cloudinary.uploader.upload(swapped_file_path, folder="swapped_images")
        
        # Return the URL of the swapped image uploaded to Cloudinary
        swapped_image_url = cloudinary_upload_result['secure_url']
        return jsonify({'swapped_image_url': swapped_image_url})

    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg'})

def process_images(img1_path, img2_path):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if image dimensions exceed 640x640 pixels
    # if img1.shape[0] > 640 or img1.shape[1] > 640:
    #     img1 = cv2.resize(img1, (640, 640))
    # if img2.shape[0] > 640 or img2.shape[1] > 640:
    #     img2 = cv2.resize(img2, (640, 640))

    faces = app.get(img2)
    source_face = faces[0]

    res = img1.copy()

    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    return res

if __name__ == '__main__':
    app.run(debug=True)
