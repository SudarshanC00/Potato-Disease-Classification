# Potato Disease Classification

## Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) 
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white) 
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) 


## Overview
This project is a machine learning application designed to classify potato leaf images into three categories: **Early Blight**, **Late Blight**, and **Healthy**. Using a convolutional neural network (CNN), the system accurately identifies and categorizes potato diseases based on leaf images. The project includes a web interface for uploading images and receiving classification results, powered by FastAPI.

## Features
- **Disease Classification**: Classifies potato leaf images into Early Blight, Late Blight, or Healthy.
- **Web Interface**: Provides an HTML interface for image uploads and displays the predicted class and confidence score.
- **API Endpoint**: Built with FastAPI, allowing image uploads for real-time predictions.

## Directory Structure
- **potato-disease/**
  - **api/**
    - `main.py`: FastAPI application with an endpoint for predicting the disease class of an uploaded image.
    - `requirements.txt`: List of dependencies required to run the project.
  - **saved_models/**
    - `version_1_model`: Contains the first version of the trained model.
    - `version_2_model`: Contains the second version of the trained model (optional for updates).
  - **training/**
    - `Potato_Disease_Detection.ipynb`: Jupyter Notebook containing code to train the CNN model on potato leaf images.
  - **index.html**: HTML file providing a simple frontend interface for uploading images and displaying prediction results.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/Potato-Disease-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Potato-Disease-Classification/potato-disease/api
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure your model is saved in the `saved_models/version_1_model` directory or update the path in `main.py` if necessary.

## Usage
1. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
2. **Access the Web Interface**:
   - Open `index.html` in a browser to use the web interface for image uploads.
   - Upload an image of a potato leaf, and click **Upload** to get the disease classification and confidence score.

3. **Using the API**:
   - Send a `POST` request to `http://localhost:8000/predict` with an image file to receive the classification.

## Model Details
- **Model Architecture**: A convolutional neural network (CNN) was used to classify potato leaf images.
- **Classes**: The model predicts one of the following classes:
  - Early Blight
  - Late Blight
  - Healthy
- **Training Details**: The model was trained using images of potato leaves with different diseases, achieving high accuracy in distinguishing between classes.

## Dependencies
- **FastAPI**: For serving the API endpoint.
- **Uvicorn**: ASGI server for running the FastAPI application.
- **TensorFlow**: For loading and utilizing the trained CNN model.
- **NumPy** and **Pillow**: For image preprocessing.

## License
This project is licensed under the MIT License.
