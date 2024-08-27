from flask import Flask, request, render_template, redirect, url_for, send_from_directory, abort
import os
import werkzeug
from werkzeug.utils import secure_filename
import webbrowser
from flask import Flask
import threading
import zipfile

from flask import send_file, after_this_request

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import math

import glob
import scipy.signal
from scipy.signal import savgol_filter
from itertools import product
import shutil
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import functional as TF



dir = 'media' +'/'
input_dir = dir + 'uploaded_videos'
uploaded_file_dir = '/' + input_dir + '/'
output_dir = dir + 'image_output'

def load_model():
    model_path = 'models/model.pth'  # Ensure this is the correct path to your model file
    try:
        model = torch.load(model_path, map_location='cpu')  # Load on CPU for compatibility
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return None

model = load_model()
if model is None:
    raise RuntimeError("Model could not be loaded.")

app = Flask(__name__)
app.secret_key = '12345'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'uploaded_videos')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MASK_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'image_output')
if not os.path.exists(app.config['MASK_FOLDER']):
    os.makedirs(app.config['MASK_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

@app.route('/')
def index():
    return render_template('index.html')

def save_image(image, filename, directory):
    import cv2
    import os
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, filename), image)

def process_video(video_file, model):
    group_id = os.path.splitext(os.path.basename(video_file))[0]
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = cap.read()
    i = 0

    while success and i < 5:
        if success:
            # Preprocess the frame as required by the model
            processed_frame = cv2.resize(frame[608-480:608+480, 912-512:912+512], (512, 480))
            # Convert frame to tensor
            frame_tensor = TF.to_tensor(processed_frame).unsqueeze(0)  # Add batch dimension
            
            # Model inference
            with torch.no_grad():
                output = model(frame_tensor)  # Adjust if your model requires different preprocessing
            
            # Optionally, convert output to an appropriate format to save or further process
            output_image = output_to_image(output)  # Define this function based on your output handling needs
            
            # Save the output image or further process it
            save_image(output_image, f"{group_id}_{i}.png", output_dir)

        success, frame = cap.read()
        i += 1

    cap.release()

def output_to_image(output):
    output_image = output.squeeze().cpu().numpy()  # Convert tensor to NumPy array and remove batch dimension

    if output_image.ndim == 2:  # If it's a single-channel image
        output_image = (output_image * 255).astype(np.uint8)

    if output_image.ndim == 3 and output_image.shape[0] == 3:  # If it's an RGB image
        output_image = output_image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        output_image = (output_image * 255).astype(np.uint8)
    
    return output_image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            group_id = os.path.splitext(filename)[0]  # Use this as video_id
            process_video(video_path, model)
            return render_template('upload_success.html', video_id=group_id)
    return render_template('index.html')

@app.route('/upload/<filename>')
def uploaded_file(filename):
    # Ensure this page does something visible or returns a response
    return render_template('upload_success.html', filename=filename)
    

@app.route('/downloads/<filename>')
def download_file(filename):
    directory = app.config['UPLOAD_FOLDER']
    if not os.path.isfile(os.path.join(directory, filename)):
        abort(404)  # Return a 404 not found error if the file does not exist
    return send_from_directory(directory, filename, as_attachment=True)


import zipfile
import os
from flask import send_file, after_this_request, flash

@app.route('/download_masks/<video_id>')
def download_masks(video_id):
    zip_filename = f"{video_id}_masks.zip"
    zip_path = os.path.join(app.config['MASK_FOLDER'], zip_filename)

    # Create a ZIP file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        added_files = 0
        for root, dirs, files in os.walk(app.config['MASK_FOLDER']):
            for file in files:
                if file.startswith(video_id) and file.endswith('.png'):
                    zipf.write(os.path.join(root, file), file)
                    added_files += 1
                    print(f"Added {file} to the ZIP archive.")

    if added_files == 0:
        print("No files were added to the ZIP archive.")
        flash("No files were found to add to the ZIP archive.", "error")
        return "No files were found to add to the ZIP archive.", 404

    @after_this_request
    def remove_file(response):
        try:
            os.remove(zip_path)  # This ensures the ZIP file is deleted after download
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    return send_file(zip_path, as_attachment=True)

def open_browser():
    """Open the web browser."""
    webbrowser.open_new('http://localhost:5007')
1
if __name__ == '__main__':
    # Use threading to prevent blocking the server start
    threading.Timer(1.25, open_browser).start()
    app.run(host='0.0.0.0', port=5007)