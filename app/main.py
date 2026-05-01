import logging
import sys
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict, Any
import time
import hashlib
import json

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modular components
from recommender import JobRanker
from utils import extract_text_from_upload_file, validate_pdf_file, sanitize_text

print("PERFECT DEPLOYMENT: Bulletproof sklearn TF-IDF with maximum optimizations")
logger.info("Initializing perfect deployment engine")

# Initialize job ranker
try:
    ranker = JobRanker()
    logger.info("JobRanker initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize JobRanker: {e}")
    sys.exit(1)


app = FastAPI(
    title="Perfect Job Ranking Engine",
    description="Production-perfect TF-IDF text similarity API with optimizations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

@app.get("/test")
def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {"status": "ok", "message": "API is working"}

@app.get("/health")
async def health_check():
    try:
        health_status = ranker.health_check()
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


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from PDF file"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be a PDF"}
            )
        
        extracted_text = extract_text_from_upload_file(file)
        
        if not extracted_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text. PDF may be scanned, empty, or corrupted."}
            )
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
        return JSONResponse(
            status_code=200,
            content={"extracted_text": extracted_text}
        )
        
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": f"PDF extraction failed: {str(e)}"}
        )

@app.get("/rank")
async def rank_jobs_get():
    """GET endpoint for /rank to prevent Method Not Allowed errors"""
    return {"message": "POST endpoint available", "methods": ["POST"]}

@app.post("/rank")
async def rank_jobs(resume: str = None, jobs: List[str] = None, file: UploadFile = None):
    """Perfect job ranking with comprehensive optimizations"""
    print(f"DEBUG: Received payload - resume: {resume}, jobs: {jobs}, file: {file}")
    try:
        # Handle PDF file upload
        extracted_text = None
        if file and file.filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_upload_file(file)
            
            if not extracted_text:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not extract text. PDF may be scanned or unsupported."}
                )
            
            # If only PDF is uploaded, return extracted text
            if not jobs or len(jobs) == 0:
                return {
                    "extracted_text": extracted_text
                }
            
            resume = extracted_text
        elif not resume:
            # Input validation for text-only requests
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
        
        # Use JobRanker for processing
        result = ranker.rank_resume_against_jobs(resume, jobs)
        
        logger.info(f"Ranking completed in {result.get('processing_time_seconds', 0):.4f}s")
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
async def serve_ui():
    """Serve HTML frontend definitively"""
    try:
        file_path = os.path.join(os.getcwd(), "static", "index.html")
        
        if not os.path.exists(file_path):
            logger.error(f"Static file not found: {file_path}")
            return HTMLResponse(content="<h1>Service unavailable</h1>", status_code=503)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        logger.info("Served frontend HTML successfully")
        return HTMLResponse(content=content)
        
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return HTMLResponse(content="<h1>Service unavailable</h1>", status_code=503)

@app.get("/metrics")
def metrics():
    """Detailed metrics endpoint for monitoring"""
    try:
        health_status = ranker.health_check()
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
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level="info"
    )
