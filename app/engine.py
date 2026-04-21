# --- LINE 1: NO IMPORTS ABOVE THIS ---
import sys
import huggingface_hub
from huggingface_hub import hf_hub_download

# Inject the missing function before any other library can complain
huggingface_hub.cached_download = hf_hub_download
sys.modules["huggingface_hub.cached_download"] = hf_hub_download

# NOW it is safe to import FastAPI
from fastapi import FastAPI
app = FastAPI()

# Move this import here to ensure that patch is fully registered
from sentence_transformers import SentenceTransformer

@app.on_event("startup")
def load_model():
    # Loading the model here ensures that patch is fully registered
    global model
    model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/health")
def health():
    return {"status": "online", "engine": "patched"}

@app.post("/rank")
async def rank_jobs(resume: str, jobs: list[str]):
    """Rank jobs using sentence-transformers with patch"""
    try:
        # Get embeddings
        resume_embedding = model.encode(resume)
        job_embeddings = model.encode(jobs)
        
        # Compute similarities
        from sentence_transformers import util
        similarities = util.cos_sim(resume_embedding, job_embeddings)
        
        # Create ranked results
        ranked_jobs = sorted(zip(jobs, similarities.tolist()), key=lambda x: x[1], reverse=True)
        
        return {
            "status": "success",
            "resume": resume,
            "ranked_jobs": [
                {"job": job, "similarity_score": float(score)} 
                for job, score in ranked_jobs
            ],
            "engine": "sentence-transformers-patched"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"message": "Job Ranking API is running", "engine": "sentence-transformers-patched"}
