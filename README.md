# Zebrafish Image Processing Flask App

This repository contains a Flask application for processing zebrafish video files, extracting masks, annotating them, and generating CSV reports. The application utilizes a pre-trained deep learning model for segmentation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Endpoints](#endpoints)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)

## Installation

### 1. Prerequisites

- Python 3.8 or later
- Virtual environment tool (optional but recommended)

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/zebrafish_app.git
cd zebrafish_app
```

### 3. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### 4. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 5. Download the Model
Make sure to place the pre-trained model (model.pth) inside the models directory:

## Usage

### 1. Running the Flask App
Activate your virtual environment (if not already activated) and run the Flask application:
```bash
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
python app.py
```
The application will automatically open in your default web browser at http://localhost:5007.

### 2. Uploading and Processing Videos
1.	Open the web interface by navigating to http://localhost:5007.
2.	Upload a .mp4 video file using the provided form.
3.	Select a timeframe for processing the video.
4.	Click the process button to begin segmentation.
5.	After processing, download the results, which will include masked images, annotated images, and a CSV report.

## Directory Structure
Below is the directory structure for the application:
  zebrafish_app/
│
├── app.py                 # Main Flask application file
├── requirements.txt       # Required Python packages
├── media/
│   ├── uploaded_videos/   # Directory to store uploaded videos
│   └── image_output/      # Directory to store output images and CSV files
├── models/
│   └── model.pth          # Pre-trained segmentation model
├── templates/
│   ├── index.html         # Main page template
│   ├── select_timeframe.html  # Template for selecting time frame
│   └── processing_success.html # Success message template
└── static/
    ├── styles.css         # CSS for styling the app
    └── scripts.js         # JavaScript for handling UI interactions

## Notes
- Ensure that the model.pth file is placed correctly inside the models directory before running the application.
- The application uses the PyTorch library to load and run the deep learning model. Make sure that your environment has access to a compatible version of PyTorch.

## Troubleshooting
- Model Loading Error: If the app cannot load the model, check the path to model.pth and ensure it is correctly placed in the models directory.
- Missing Output Files: Make sure the directories media/uploaded_videos and media/image_output are correctly set up.
- Port Conflicts: If port 5007 is already in use, either free it up or modify the app to use a different port by changing the app.run() line in app.py.
