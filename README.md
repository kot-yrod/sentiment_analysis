# 🐦 Tweet Sentiment Classifier (DistilBERT)

A lightweight sentiment analysis model that classifies tweets into *positive* or *negative* categories using a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) transformer.

[🤗 View on Hugging Face](https://huggingface.co/KotYrod/tweet-sentiment-distilbert)

---

## 🧠 Model Overview

- *Base model*: distilbert-base-uncased
- *Task*: Tweet sentiment classification (2 classes: Positive 😊 / Negative 😠)
- *Dataset*: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- *Training device*: Local GPU

---

## 🛠️ Training Details

- *Data Preprocessing*:
  - Removed URLs, mentions, hashtags, and extra spaces
  - Deduplicated and dropped empty tweets
- *Train/Val/Test Split*: 60% / 20% / 20%
- *Loss function*: CrossEntropyLoss
- *Optimizer*: Adam
- *Learning rate*: 2e-6
- *Batch size*: 32
- *Epochs*: 7
- *Evaluation metrics*: Accuracy, F1, Precision, Recall, ROC-AUC

---

## 📊 Evaluation Results

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.8070 |
| Precision   | 0.7880 |
| Recall      | 0.8400 |
| F1 Score    | 0.8132 |
| ROC AUC     | 0.8910 |

---

## 🚀 Example Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("KotYrod/tweet-sentiment-distilbert").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict_sentiment(text):
    model.eval()
    text = re.sub(r"http\S+|www\S+|\@\w+|\#|\s+", ' ', text).strip()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return ("Positive 😊" if pred == 1 else "Negative 😠", probs[0][pred].item())

if _name_ == "_main_":
    label, conf = predict_sentiment("wow im so glad")
    print(f"Prediction: {label} (Confidence: {conf:.2f})")