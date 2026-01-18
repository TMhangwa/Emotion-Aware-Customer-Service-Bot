import os
import uvicorn
import os
from fastapi import FastAPI
from huggingface_hub import InferenceClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Connect to the Inference API (No local model download required!)
client = InferenceClient(
    model="j-hartmann/emotion-english-distilroberta-base",
    token=os.getenv("HF_API_TOKEN")  # Add this in Railway Variables
)


@app.post("/chat")
async def chat(message: str):
    # This call happens over the network, keeping your image tiny
    results = client.text_classification(message)
    top_emotion = results[0]['label']

    return {"emotion": top_emotion, "reply": f"I detect you are feeling {top_emotion}."}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific domain
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # Railway provides the port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    # Use 0.0.0.0 to allow external traffic to reach the container
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

