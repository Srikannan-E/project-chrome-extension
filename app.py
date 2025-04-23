import os
import zipfile
import gdown
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from apscheduler.schedulers.background import BackgroundScheduler
from asyncio import to_thread
from contextlib import asynccontextmanager
import threading
import asyncio

# Define persistent volume path and model/feedback file locations
VOLUME_PATH = "/model-store"
MODEL_DIR = os.path.join(VOLUME_PATH, "sentiment_model")
MODEL_ZIP = os.path.join(VOLUME_PATH, "sentiment_model.zip")
FEEDBACK_FILE = os.path.join(VOLUME_PATH, "user_feedback.json")

# Google Drive file id for downloading the model zip
GDRIVE_FILE_ID = "1MWjTdCFHeYE1OVrpm7EEBDE19Jdtf9uj"

# Download and extract model if not present on the persistent volume
async def download_and_extract_model():
    # Ensure the volume directory exists
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    if not os.path.exists(MODEL_DIR):
        print("📥 Model not found in volume. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        # Download the model zip file into the persistent volume
        gdown.download(url, MODEL_ZIP, quiet=False)
        
        # Extract the zip file contents into the volume path
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(VOLUME_PATH)
        
        print("✅ Model extracted to volume.")
    else:
        print("✅ Model already exists in volume.")

# Load model from the persistent volume (only once)
async def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Global variable to hold the pipeline instance
sentiment_pipeline = None

# Label map for sentiment categories
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Lifespan event to load the model during startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_pipeline
    await download_and_extract_model()
    sentiment_pipeline = await load_model()
    yield  # Place for any shutdown cleanup if needed

# Initialize FastAPI app with lifespan events
app = FastAPI(lifespan=lifespan)

# CORS Middleware (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic Request Models
class PredictRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text: str
    sentiment: str

@app.get("/")
async def home():
    return {"message": "Sentiment API with auto-retraining is live!"}

@app.post("/predict")
async def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"error": "No text provided."}

    if sentiment_pipeline is None:
        return {"error": "Model is still loading. Please try again later."}

    # Offload the CPU-bound sentiment analysis to a separate thread
    result = await to_thread(sentiment_pipeline, text)
    sentiment_label = result[0]["label"]
    sentiment = label_map.get(sentiment_label, "Unknown")
    confidence = round(result[0]["score"] * 100, 2)
    return {
        "sentiment": sentiment,
        "confidence": confidence
    }

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    text = req.text.strip()
    sentiment = req.sentiment.strip()
    if not text or sentiment not in ["Positive", "Negative", "Neutral"]:
        return {"error": "Invalid feedback."}
    
    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        # Asynchronously read the feedback file
        feedback_data = await to_thread(read_feedback_file)
    
    feedback_data.append({"text": text, "sentiment": sentiment})
    await to_thread(write_feedback_file, feedback_data)
    
    return {"message": "Feedback saved!"}

# Helper functions to read and write the feedback file asynchronously
async def read_feedback_file():
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

async def write_feedback_file(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Scheduled retraining job function
def scheduled_retrain():
    print("🔁 Checking for feedback to retrain...")
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback found.")
        return

    async def retrain_model():
        import subprocess
        try:
            subprocess.run(["python", "retrain_model.py"], check=True)
            global sentiment_pipeline
            sentiment_pipeline = await load_model()  # Reload model after retraining
            print("✅ Model reloaded after retraining.")
        except Exception as e:
            print(f"❌ Retrain error: {e}")

    # Run the retraining asynchronously in a separate thread
    retrain_thread = threading.Thread(target=lambda: asyncio.run(retrain_model()))
    retrain_thread.start()

# Schedule retraining job every 60 minutes using APScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_retrain, 'interval', minutes=60)
scheduler.start()

# Run the app via Uvicorn if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
