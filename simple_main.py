from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import re
from typing import List, Dict, Any

app = FastAPI(title="TalentRanker AI")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def simple_text_similarity(resume: str, job: str) -> float:
    """Simple text similarity without ML dependencies"""
    # Convert to lowercase and split into words
    resume_words = set(re.findall(r'\w+', resume.lower()))
    job_words = set(re.findall(r'\w+', job.lower()))
    
    # Calculate Jaccard similarity
    intersection = len(resume_words & job_words)
    union = len(resume_words | job_words)
    
    if union == 0:
        return 0.0
    
    return intersection / union

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "TalentRanker AI - Working!", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "simple-text-similarity"}

@app.post("/rank")
async def rank(resume: str = None, jobs: List[str] = None):
    if not resume or not jobs:
        return {"error": "Resume and jobs required"}
    
    # Calculate similarity scores
    scores = []
    for job in jobs:
        similarity = simple_text_similarity(resume, job)
        scores.append(similarity)
    
    # Get the first job's score for match_score
    match_score = scores[0] if scores else 0.0
    
    # Create explanation based on score
    if match_score > 0.7:
        explanation = f"Strong match with {match_score:.1%} similarity. Resume contains many keywords from the job description."
    elif match_score > 0.4:
        explanation = f"Moderate match with {match_score:.1%} similarity. Some relevant skills found but could be improved."
    else:
        explanation = f"Weak match with {match_score:.1%} similarity. Consider updating resume with more relevant keywords."
    
    return {
        "status": "success",
        "match_score": match_score,
        "explanation": explanation,
        "ranked_jobs": [
            {
                "rank": i + 1,
                "job": jobs[i][:200] + "..." if len(jobs[i]) > 200 else jobs[i],
                "similarity_score": round(scores[i], 4),
                "match_quality": "High" if scores[i] > 0.7 else "Medium" if scores[i] > 0.4 else "Low"
            }
            for i in range(len(jobs))
        ],
        "engine": "simple-text-similarity",
        "processed_jobs": len(jobs)
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
