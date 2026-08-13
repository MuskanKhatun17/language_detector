from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Muskan1304/xlmr-language-detector"
MAX_LENGTH = 128

app = FastAPI(title="Language Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

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

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    top_prob, top_id = torch.max(probs, dim=-1)
    predicted_language = model.config.id2label[int(top_id)]

    top5_ids = torch.topk(probs, k=min(5, probs.shape[0])).indices.tolist()
    top_5 = [{"language": model.config.id2label[i], "confidence": round(float(probs[i]), 4)} for i in top5_ids]

    return PredictResponse(language=predicted_language, confidence=round(float(top_prob), 4), top_5=top_5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_index():
    return FileResponse("index.html")
