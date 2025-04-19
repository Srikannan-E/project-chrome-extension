# retrain_model.py
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load feedback data
with open("user_feedback.json") as f:
    feedback_data = json.load(f)

if len(feedback_data) < 10:
    print("⚠️ Not enough feedback to retrain.")
    exit()

label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df = pd.DataFrame(feedback_data)
df["label"] = df["sentiment"].map(label_map)
df = df[["text", "label"]]

dataset = Dataset.from_pandas(df)

# Load model
model_dir = "sentiment_model"
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_dir if os.path.exists(model_dir) else model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_dir if os.path.exists(model_dir) else model_name)

# Preprocess
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

# Training
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_steps=10,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("✅ Model retrained and saved!")
