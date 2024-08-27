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
