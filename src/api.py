from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import os

# تعريف هيكل البيانات المستقبلة
class SentimentRequest(BaseModel):
    text: str

# إعداد التطبيق
app = FastAPI(title="Sentiment Analysis API")

# مسار المودل (تأكد أن المودل مدرب وموجود هنا)
MODEL_PATH = "./models/distilbert_finetuned"

# تحميل المودل والتوكينايزر عند تشغيل التطبيق
try:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model.eval() # وضع المودل في حالة التقييم
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you ran 'python main.py' first to train the model.")

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Use /predict to classify text."}

@app.post("/predict")
def predict(request: SentimentRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    # معالجة النص
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # التوقع
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

    # تحويل الرقم إلى نص
    label = "Positive" if predicted_class_id == 1 else "Negative"
    confidence = torch.softmax(logits, dim=1).max().item()

    return {
        "text": request.text,
        "sentiment": label,
        "confidence": f"{confidence:.4f}"
    }