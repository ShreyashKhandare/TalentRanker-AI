import logging
from fastapi import FastAPI
import numpy as np
from typing import List

# Use sklearn's TF-IDF for text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("MINIMAL FIX: Using sklearn TF-IDF only")

class SimpleEmbedder:
    """TF-IDF based text similarity"""
    def __init__(self, model_name: str = None):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False
        
    def encode(self, texts: List[str]):
        """Encode texts using TF-IDF"""
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        return self.vectorizer.transform(texts).toarray()
        
    def compute_similarity(self, text1: str, text2: str):
        """Compute cosine similarity between two texts"""
        embeddings = self.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Initialize embedder
model = SimpleEmbedder()
print("MINIMAL FIX: TF-IDF embedder ready")

app = FastAPI(title="Job Ranking Engine")

@app.get("/health")
def health():
    return {"status": "online", "model_loaded": model is not None, "engine": "sklearn-tfidf"}

@app.post("/rank")
async def rank_jobs(resume: str, jobs: list[str]):
    """Rank jobs using TF-IDF similarity"""
    try:
        scores = []
        for job in jobs:
            similarity = model.compute_similarity(resume, job)
            scores.append(float(similarity))
        
        ranked_jobs = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
        
        return {
            "status": "success",
            "resume": resume,
            "ranked_jobs": [
                {"job": job, "similarity_score": score} 
                for job, score in ranked_jobs
            ],
            "engine": "sklearn-tfidf"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"message": "Job Ranking API is running", "engine": "sklearn-tfidf"}
