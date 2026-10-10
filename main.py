
from fastapi import FastAPI

app = FastAPI(title="ArtMentorAI")


@app.get("/")
def home():
    return {
        "message": "Welcome to ArtMentorAI!",
        "status": "Your AI art tutor backend is running."
    }
