from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="TalentRanker AI")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "TalentRanker AI - Working!", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "simple"}

@app.post("/rank")
async def rank(resume: str = None, jobs: list = None):
    if not resume or not jobs:
        return {"error": "Resume and jobs required"}
    
    # Simple mock ranking
    return {
        "match_score": 0.85,
        "explanation": "Mock analysis - working!",
        "status": "success"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
