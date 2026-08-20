import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort
from scipy.special import softmax

MODEL_REPO = "Muskan1304/xlmr-language-detector-onnx"
MAX_LENGTH = 128

# Download just the 3 files we actually need, straight from the Hub
model_path = hf_hub_download(MODEL_REPO, "model_quantized.onnx")
tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json")
config_path = hf_hub_download(MODEL_REPO, "config.json")

tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(max_length=MAX_LENGTH)
tokenizer.enable_padding(length=MAX_LENGTH)

with open(config_path) as f:
    config = json.load(f)
id2label = {int(k): v for k, v in config["id2label"].items()}

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_names = {i.name for i in session.get_inputs()}

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

    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

    logits = session.run(None, ort_inputs)[0][0]
    probs = softmax(logits)

    top_id = int(np.argmax(probs))
    predicted_language = id2label[top_id]

    top5_ids = np.argsort(probs)[::-1][:5]
    top_5 = [{"language": id2label[int(i)], "confidence": round(float(probs[i]), 4)} for i in top5_ids]

    return PredictResponse(language=predicted_language, confidence=round(float(probs[top_id]), 4), top_5=top_5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_index():
    return FileResponse("index.html")
