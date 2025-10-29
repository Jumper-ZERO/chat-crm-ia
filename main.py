from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pysentimiento import create_analyzer

app = FastAPI(title="Analyzer Sentiment API", version="1.0.0")
analyzer = create_analyzer(task="sentiment", lang="es")

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/analyze_message")
def analyze_message(input: TextInput):
    try:
        result = analyzer.predict(input.text)
        return {
            "text": input.text,
            "label": result.output,
            "probabilities": result.probas
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
