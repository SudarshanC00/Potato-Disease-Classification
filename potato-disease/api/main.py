from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../saved_models/version_1_model")

MODEL = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(256,256,3)),  # Replace with the correct input shape for your model
        tf.keras.layers.TFSMLayer("../saved_models/version_1_model", call_endpoint='serving_default')
    ])

# endpoint = "https://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,0)
    prediction = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction['output_0'])]
    confidence = (round(np.max(prediction['output_0'])))*100
    return JSONResponse(content={"message": f"Predicted Class: {predicted_class}, Confidence: {confidence}%"})
    # return {
    #     'class' : predicted_class,
    #     'confidence': confidence
    # }



if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)



