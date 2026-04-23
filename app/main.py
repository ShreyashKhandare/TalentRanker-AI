import logging
import sys
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile
import os
import numpy as np
from typing import List, Dict, Any
import time
import os
import hashlib
import json

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use sklearn's TF-IDF for text similarity - ABSOLUTELY NO sentence-transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("PERFECT DEPLOYMENT: Bulletproof sklearn TF-IDF with maximum optimizations")
logger.info("Initializing perfect deployment engine")

class PerfectEmbedder:
    """Production-perfect TF-IDF with caching, monitoring, and bulletproof error handling"""
    
    def __init__(self, model_name: str = None):
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased for better accuracy
                stop_words='english',
                ngram_range=(1, 3),  # Better context
                lowercase=True,
                strip_accents='ascii',
                min_df=1,  # Include all terms
                max_df=0.95,  # Remove too common terms
                sublinear_tf=True  # Log scaling
            )
            self.fitted = False
            self.cache = {}  # Simple caching
            self.initialization_time = time.time()
            self.request_count = 0
            self.total_similarity_time = 0
            
            logger.info("PerfectEmbedder initialized with optimizations")
        except Exception as e:
            logger.error(f"Failed to initialize PerfectEmbedder: {e}")
            raise
        
    def _get_cache_key(self, text1: str, text2: str) -> str:
        """Generate cache key for text pair"""
        combined = f"{text1}|{text2}"
        return hashlib.md5(combined.encode()).hexdigest()
        
    def encode(self, texts: List[str]):
        """Encode texts using TF-IDF with comprehensive error handling"""
        try:
            if not texts:
                raise ValueError("Empty text list provided")
                
            if not self.fitted:
                self.vectorizer.fit(texts)
                self.fitted = True
                logger.info(f"TF-IDF vectorizer fitted on {len(texts)} texts")
                
            result = self.vectorizer.transform(texts).toarray()
            logger.debug(f"Encoded {len(texts)} texts to shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise HTTPException(status_code=500, detail=f"Encoding error: {str(e)}")
        
    def compute_similarity(self, text1: str, text2: str):
        """Compute cosine similarity with caching and performance monitoring"""
        try:
            if not text1 or not text2:
                raise ValueError("Empty text provided for similarity computation")
            
            # Check cache first
            cache_key = self._get_cache_key(text1, text2)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for similarity computation")
                return self.cache[cache_key]
            
            # Compute similarity
            start_time = time.time()
            embeddings = self.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Ensure similarity is in valid range [0,1]
            similarity = max(0.0, min(1.0, float(similarity)))
            
            # Cache result
            self.cache[cache_key] = similarity
            
            # Update metrics
            self.request_count += 1
            self.total_similarity_time += (time.time() - start_time)
            
            logger.debug(f"Computed similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise HTTPException(status_code=500, detail=f"Similarity error: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with metrics"""
        try:
            uptime = time.time() - self.initialization_time
            avg_similarity_time = (self.total_similarity_time / self.request_count 
                               if self.request_count > 0 else 0)
            
            return {
                "status": "healthy",
                "engine": "sklearn-tfidf-perfect",
                "fitted": self.fitted,
                "uptime_seconds": round(uptime, 2),
                "cache_size": len(self.cache),
                "request_count": self.request_count,
                "avg_similarity_time_ms": round(avg_similarity_time * 1000, 2),
                "vectorizer_features": self.vectorizer.get_feature_names_out().shape[0] if self.fitted else 0,
                "memory_usage_mb": round(sys.getsizeof(self.cache) / 1024 / 1024, 2)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

# Initialize perfect embedder
try:
    model = PerfectEmbedder()
    logger.info("Perfect TF-IDF embedder ready for production deployment")
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    sys.exit(1)

app = FastAPI(
    title="Perfect Job Ranking Engine",
    description="Production-perfect TF-IDF text similarity API with optimizations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files AFTER app creation
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.middleware("http")
async def log_requests(request, call_next):
    """Comprehensive request logging with performance tracking"""
    start_time = time.time()
    
    # Log request details
    logger.info(f"Request started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log completion
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = hashlib.md5(
            f"{time.time()}{request.url.path}".encode()
        ).hexdigest()[:16]
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.4f}s"
        )
        raise

@app.get("/health")
def health():
    """Perfect health check endpoint"""
    try:
        health_status = model.health_check()
        return JSONResponse(
            status_code=200,
            content=health_status
        )
    except Exception as e:
        logger.error(f"Health endpoint error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Service unavailable"}
        )

import pdfplumber
import io

def extract_text_from_pdf(upload_file: UploadFile) -> str:
    try:
        pdf_bytes = upload_file.file.read()
        pdf_stream = io.BytesIO(pdf_bytes)

        text = ""

        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        print("Extracted length:", len(text))

        return text.strip()

    except Exception as e:
        print("PDF extraction error:", str(e))
        return ""

@app.post("/rank")
async def rank_jobs(resume: str = None, jobs: List[str] = None, file: UploadFile = None):
    """Perfect job ranking with comprehensive optimizations"""
    try:
        # Handle PDF file upload
        extracted_text = None
        if file and file.filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file)

            if not extracted_text:
                return {
                    "error": "Could not extract text. PDF may be scanned or unsupported."
                }

            # If only PDF is uploaded, return extracted text
            if not jobs or len(jobs) == 0:
                return {
                    "extracted_text": extracted_text
                }
            
            resume = extracted_text
        elif not resume:
            # Input validation
            if not resume or not resume.strip():
                raise HTTPException(status_code=400, detail="Resume text cannot be empty")
            
        if not jobs or len(jobs) == 0:
            raise HTTPException(status_code=400, detail="Jobs list cannot be empty")
            
        # Limit jobs to prevent timeout
        if len(jobs) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 jobs allowed per request")
        
        # Clean and validate inputs
        resume = resume.strip()
        jobs = [job.strip() for job in jobs if job.strip()]
        
        logger.info(f"Processing perfect ranking request: {len(jobs)} jobs")
        start_time = time.time()
        
        # Compute similarity scores with optimizations
        scores = []
        for i, job in enumerate(jobs):
            try:
                similarity = model.compute_similarity(resume, job)
                scores.append(float(similarity))
            except Exception as e:
                logger.warning(f"Failed to compute similarity for job {i}: {e}")
                scores.append(0.0)  # Fallback score
        
        # Create ranked results with additional metadata
        ranked_jobs = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
        
        processing_time = time.time() - start_time
        
        # Calculate additional metrics
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        result = {
            "status": "success",
            "resume": resume[:200] + "..." if len(resume) > 200 else resume,
            "processed_jobs": len(jobs),
            "processing_time_seconds": round(processing_time, 4),
            "ranked_jobs": [
                {
                    "rank": idx + 1,
                    "job": job[:200] + "..." if len(job) > 200 else job,
                    "similarity_score": round(score, 4),
                    "match_quality": "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
                } 
                for idx, (job, score) in enumerate(ranked_jobs)
            ],
            "engine": "sklearn-tfidf-perfect",
            "timestamp": time.time(),
            "metrics": {
                "average_similarity": round(avg_score, 4),
                "max_similarity": round(max_score, 4),
                "cache_hits": len(model.cache),
                "request_id": hashlib.md5(f"{time.time()}{len(jobs)}".encode()).hexdigest()[:16]
            }
        }
        
        logger.info(f"Perfect ranking completed in {processing_time:.4f}s")
        return JSONResponse(status_code=200, content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rank endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": "Internal server error",
                "engine": "sklearn-tfidf-perfect"
            }
        )
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve HTML frontend definitively"""
    file_path = os.path.join(os.getcwd(), "static", "index.html")
    
    if not os.path.exists(file_path):
        return {"error": f"File not found at {file_path}"}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/metrics")
def metrics():
    """Detailed metrics endpoint for monitoring"""
    try:
        health_status = model.health_check()
        return JSONResponse(
            status_code=200,
            content={
                "service": "job-ranking-api",
                "engine": "sklearn-tfidf-perfect",
                "version": "2.0.0",
                "health": health_status,
                "system": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "process_id": os.getpid()
                },
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail, "engine": "sklearn-tfidf-perfect"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "engine": "sklearn-tfidf-perfect"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.engine:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level="info",
        access_log=True
    )
