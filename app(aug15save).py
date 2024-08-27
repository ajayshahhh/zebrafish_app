from flask import Flask, request, render_template, redirect, url_for, send_from_directory, abort
import os
import werkzeug
from werkzeug.utils import secure_filename
import webbrowser
from flask import Flask
import threading

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import math
import os
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
    model_path = './models/model.pth'  # Ensure this is the correct path to your model file
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'uploaded_videos')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

@app.route('/')
def index():
    return render_template('index.html')

def save_image(image, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(os.path.join(directory, filename), image)

def process_video(video_file):
    group_id = os.path.splitext(os.path.basename(video_file))[0]

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = cap.read()
    i = 0
    #frame = cv2.resize(frame, (512, 480), interpolation=cv2.INTER_AREA)
    #save_image(frame, group_id + "_" + str(i) + '.png', output_dir)

    while success and i < 50:
        if success:
            #if (i % 8 == 0):
            frame = cv2.resize(frame[608-480:608+480, 912-512:912+512], (512, 480), interpolation=cv2.INTER_AREA)
            save_image(frame, group_id + "_" + str(i) + '.png', output_dir)
        success, frame = cap.read()
        i += 1
    cap.release()
 


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
            try:
                # Call the process_video function from applyVideo.py
                process_video(video_path)
            except Exception as e:
                print(f"Error processing video: {e}")
                abort(500)  # Internal Server Error
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

@app.route(uploaded_file_dir + '<filename>')
def uploaded_file(filename):
    # Using a template
    return render_template('upload_success.html', filename=filename)
    

@app.route('/downloads/<filename>')
def download_file(filename):
    directory = app.config['UPLOAD_FOLDER']
    if not os.path.isfile(os.path.join(directory, filename)):
        abort(404)  # Return a 404 not found error if the file does not exist
    return send_from_directory(directory, filename, as_attachment=True)

def open_browser():
    """Open the web browser."""
    webbrowser.open_new('http://localhost:5007')
1
if __name__ == '__main__':
    # Use threading to prevent blocking the server start
    threading.Timer(1.25, open_browser).start()
    app.run(host='0.0.0.0', port=5007)