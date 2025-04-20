import os
import zipfile
import gdown
import json
import pandas as pd
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from apscheduler.schedulers.background import BackgroundScheduler
from flask_cors import CORS

# 🔧 Config
MODEL_DIR = "sentiment_model"
MODEL_ZIP = "sentiment_model.zip"
FEEDBACK_FILE = "user_feedback.json"
GDRIVE_FILE_ID = "1fcmCfWgcPLGQshqp_vOfL9D9wsaoxj_w"  # Your Drive file

# 🔁 Download model if not present
if not os.path.exists(MODEL_DIR):
    print("📥 Model not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_ZIP, quiet=False)

    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ Model extracted.")
else:
    print("✅ Model already exists locally.")

# ✅ Load model once globally for fast response
print("🔁 Loading model into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("✅ Model loaded.")

# 🚀 Flask setup
app = Flask(__name__)
CORS(app)

# 🧠 Label map
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

@app.route("/")
def home():
    return "Sentiment API with auto-retraining is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_pipeline(text)[0]
    return jsonify({
        "sentiment": label_map.get(result["label"], "Unknown"),
        "confidence": round(result["score"] * 100, 2)
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    text = data.get("text", "").strip()
    sentiment = data.get("sentiment", "").strip()

    if not text or sentiment not in ["Positive", "Negative", "Neutral"]:
        return jsonify({"error": "Invalid feedback"}), 400

    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            feedback_data = json.load(f)

    feedback_data.append({"text": text, "sentiment": sentiment})

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=4)

    return jsonify({"message": "Feedback saved!"})

# 🔁 Retraining scheduler
def scheduled_retrain():
    print("🔁 Checking for feedback to retrain...")
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback found.")
        return

    try:
        import subprocess
        subprocess.run(["python", "retrain_model.py"], check=True)

        # ⚡ Reload the updated model
        global sentiment_pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print("✅ Model reloaded after retraining.")
    except Exception as e:
        print(f"❌ Retrain error: {e}")

# 🔂 Schedule retraining every 60 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_retrain, 'interval', minutes=60)
scheduler.start()

# ✅ Run app (Railway-compatible)
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
