from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import tensorflow as tf
import os
from fastapi import UploadFile, File
from PIL import Image
import numpy as np
import io


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "1")

print("Loading model from:", MODEL_PATH)

MODEL = tf.saved_model.load(MODEL_PATH)

CLASS_NAMES = [
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy"
]


@app.get("/")
def home():
    return {
        "message": "Welcome to Kloris Potato Disease Detection API!"
    }


@app.get("/ping")
def ping():
    return {
        "message": "Hello I am alive!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Resize to match model input
    image = image.resize((256, 256))

    # Convert to numpy array
    img_array = np.array(image)

    # Create batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    infer = MODEL.signatures["serve"]
    prediction = infer(tf.constant(img_batch, dtype=tf.float32))

    prediction = list(prediction.values())[0].numpy()

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2)
    }