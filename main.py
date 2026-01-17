import os
import requests
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

app = FastAPI()
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Hugging Face Inference API for Emotion
HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}


def query_emotion(text):
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    return response.json()[0][0]  # Returns {'label': 'anger', 'score': 0.9}


@app.post("/chat")
async def chat(message: str):
    # 1. Get Emotion
    emotion_data = query_emotion(message)
    label = emotion_data['label']

    # 2. Generate Empathetic Response
    sys_msg = SystemMessage(content=f"The user is feeling {label}. Respond empathetically.")
    user_msg = HumanMessage(content=message)

    ai_response = llm.invoke([sys_msg, user_msg])

    return {
        "reply": ai_response.content,
        "emotion_detected": label
    }
if __name__ == "__main__":
    # Railway provides the port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    # Use 0.0.0.0 to allow external traffic to reach the container
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)