from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np
from scipy.special import softmax

MODEL_NAME = "Muskan1304/xlmr-language-detector-onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME, file_name="model_quantized.onnx")

app = FastAPI(title="Language Detection API")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    language: str
    confidence: float
    top_5: list

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text field is empty.")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    probs = softmax(logits)

    top_id = int(np.argmax(probs))
    predicted_language = model.config.id2label[top_id]

    top5_ids = np.argsort(probs)[::-1][:5]
    top_5 = [{"language": model.config.id2label[int(i)], "confidence": round(float(probs[i]), 4)} for i in top5_ids]

    return PredictResponse(language=predicted_language, confidence=round(float(probs[top_id]), 4), top_5=top_5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_index():
    return FileResponse("index.html")
