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

MODEL_DIR = "sentiment_model"
MODEL_ZIP = "sentiment_model.zip"
FEEDBACK_FILE = "user_feedback.json"
GDRIVE_FILE_ID = "1fcmCfWgcPLGQshqp_vOfL9D9wsaoxj_w"

# Download and extract model if not present
async def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print("📥 Model not found. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP, quiet=False)
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Model extracted.")
    else:
        print("✅ Model already exists locally.")

# Load model once
async def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

sentiment_pipeline = None

# Label map for sentiment categories
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_pipeline
    await download_and_extract_model()
    sentiment_pipeline = await load_model()
    yield  # You can add shutdown cleanup here later if needed

# FastAPI App
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request Models
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
        return {"error": "No text provided"}
    # Perform the prediction asynchronously
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
        return {"error": "Invalid feedback"}

    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        # Perform file I/O asynchronously
        feedback_data = await to_thread(read_feedback_file)

    feedback_data.append({"text": text, "sentiment": sentiment})
    await to_thread(write_feedback_file, feedback_data)

    return {"message": "Feedback saved!"}

async def read_feedback_file():
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

async def write_feedback_file(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Retrain Job
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
            sentiment_pipeline = await load_model()  # Re-load model asynchronously
            print("✅ Model reloaded after retraining.")
        except Exception as e:
            print(f"❌ Retrain error: {e}")

    # Use a thread to retrain asynchronously
    retrain_thread = threading.Thread(target=lambda: asyncio.run(retrain_model()))
    retrain_thread.start()

# Schedule retraining every hour
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_retrain, 'interval', minutes=60)
scheduler.start()

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

