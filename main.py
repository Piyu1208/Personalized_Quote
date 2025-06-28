from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import requests
from fastapi.middleware.cors import CORSMiddleware




# ----------------------------
# Initialize App
# ----------------------------
app = FastAPI(title="Personalized Quote Generator")

origins = [
    "http://localhost:3000",      # Local frontend
    "http://127.0.0.1:3000",
    "https://your-frontend.web.app",  # Deployed frontend (change as needed)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Can also be ["*"] to allow all origins (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods
    allow_headers=["*"],    # Allow all headers
)

@app.get("/")
def root():
    return {"message": "🚀 API is live and ready!"}

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
try:
    vectorizer = load("sentiment_vectorizer.joblib")
    model = load("sentiment_classifier.joblib")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or vectorizer: {e}")

# ----------------------------
# Request Body Schema
# ----------------------------
class UserInput(BaseModel):
    text: str

# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
def generate_quote(user_input: UserInput):
    text = user_input.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    try:
        # Predict sentiment
        X = vectorizer.transform([text])
        sentiment = model.predict(X)[0].capitalize()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    try:
        # Fetch quote from ZenQuotes
        response = requests.get("https://zenquotes.io/api/random", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and data:
                quote = data[0]["q"]
                author = data[0]["a"]
            else:
                raise HTTPException(status_code=502, detail="Unexpected quote format.")
        else:
            raise HTTPException(status_code=502, detail="Could not fetch quote.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Quote fetch failed: {e}")

    return {
        "sentiment": sentiment,
        "quote": {
            "text": quote,
            "author": author
        }
    }
