import torch
import os
os.environ["HF_HOME"] = "D:/pytorch_project/huggingface_cache"
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    label, conf = predict_sentiment("wow im so glad")
    print(f"Prediction: {label} (Confidence: {conf:.2f})")