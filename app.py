import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient

MODEL_REPO = "Muskan1304/xlmr-language-detector"   # your original PyTorch model repo, not the ONNX one
HF_TOKEN = os.environ.get("HF_TOKEN")               # set this in Render's Environment tab

client = InferenceClient(model=MODEL_REPO, token=HF_TOKEN)

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

    try:
        results = client.text_classification(text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model service error: {e}")

    # results is a list of {"label": ..., "score": ...}, sorted by score already
    top = results[0]
    top_5 = [{"language": r["label"], "confidence": round(r["score"], 4)} for r in results[:5]]

    return PredictResponse(language=top["label"], confidence=round(top["score"], 4), top_5=top_5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_index():
    return FileResponse("index.html")
