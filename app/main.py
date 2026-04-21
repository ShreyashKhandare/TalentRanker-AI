import sys
import logging

# 1. Fix the HuggingFace Import Error prominently
try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download
    setattr(huggingface_hub, 'cached_download', hf_hub_download)
except ImportError:
    logging.warning("huggingface_hub not found, skipping shim.")

"""FastAPI service for job-match inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from xgboost import XGBRanker
import PyPDF2
import pdfplumber

from .core.features import SKILL_WEIGHTS, create_feature_matrix
from .core.recommender import get_verdict
from .core.utils import _extract_text_skills



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_advanced.json")  # Use Advanced model
FEATURE_COLUMNS = ["skill_overlap", "weighted_score", "semantic_similarity"]
RANKER_MODEL: XGBRanker | None = None
EMBEDDING_MODEL: Any | None = None

# Extendable skill vocabulary for missing-skill extraction.
KNOWN_SKILLS = sorted(
    {
        *SKILL_WEIGHTS.keys(),
        "docker",
        "pytorch",
        "tensorflow",
        "statistics",
        "nlp",
        "aws",
        "kubernetes",
        "spark",
        "java",
        "c++",
        "react",
    }
)
STOP_WORDS = {
    "and",
    "the",
    "with",
    "for",
    "to",
    "of",
    "in",
    "on",
    "a",
    "an",
    "is",
    "are",
}


class InferenceRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="User resume or profile text")
    job_description: str = Field(..., min_length=1, description="Target job description text")


class InferenceResponse(BaseModel):
    match_score: float = Field(..., ge=0, le=1, description="Predicted match quality score (0-1 scale, sigmoid normalized)")
    verdict: str = Field(..., description="Recommendation: Apply/Maybe/Not Recommended (with uncertainty prefix)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1 scale)")
    feature_contributions: dict[str, float] = Field(..., description="Relative contribution of each feature")
    explanation: str = Field(..., description="Human-readable explanation of the match")
    missing_skills: list[str] = Field(default_factory=list, description="Skills missing from resume")
    skill_overlap: float = Field(..., ge=0, le=1, description="Raw skill overlap score")
    weighted_score: float = Field(..., ge=0, le=1, description="Weighted skill match score")
    semantic_similarity: float = Field(..., ge=0, le=1, description="Semantic similarity score")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once on startup."""
    global RANKER_MODEL, EMBEDDING_MODEL
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model load failure: expected file at '{MODEL_PATH}'")
            raise FileNotFoundError("Model file not found. Please run train_model.py first.")

        ranker = XGBRanker()
        ranker.load_model(MODEL_PATH)
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Model feature importances: {ranker.feature_importances_}")

        RANKER_MODEL = ranker
        EMBEDDING_MODEL = st_model

        app.state.ranker_model = ranker
        app.state.embedding_model = st_model
    except FileNotFoundError as exc:
        print(f"{exc} (looked in: {MODEL_PATH})")
        raise FileNotFoundError("Model file not found. Please run train_model.py first.")
    except Exception as exc:
        print(f"Failed to initialize models: {exc}")
        raise
    yield

app = FastAPI(
    title="Job Match API",
    version="1.0.0",
    description="Predicts resume-to-job match quality with ranker + semantic similarity.",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend index.html."""
    from fastapi.responses import HTMLResponse
    with open("/app/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health(request: Request) -> dict:
    """Health check endpoint for deployment platforms."""
    ranker_loaded = getattr(request.app.state, "ranker_model", None) is not None
    embedder_loaded = getattr(request.app.state, "embedding_model", None) is not None
    if ranker_loaded and embedder_loaded:
        return {"status": "healthy", "models": {"ranker": True, "embeddings": True}}
    return {"status": "unhealthy", "models": {"ranker": False, "embeddings": False}}


def get_ranker_model(request: Request) -> XGBRanker:
    """Get ranker model from global variable or app state."""
    model = RANKER_MODEL if RANKER_MODEL is not None else getattr(request.app.state, "ranker_model", None)
    if model is None:
        raise RuntimeError("Ranker model is not loaded.")
    return model


def get_embedding_model(request: Request) -> Any:
    """Get embedding model from global variable or app state."""
    model = EMBEDDING_MODEL if EMBEDDING_MODEL is not None else getattr(request.app.state, "embedding_model", None)
    if model is None:
        raise RuntimeError("Embedding model is not loaded.")
    return model


def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text from uploaded PDF file using multiple methods for better accuracy."""
    try:
        # Read PDF content
        pdf_content = pdf_file.file.read()
        
        # Try pdfplumber first (better for modern PDFs)
        try:
            with pdfplumber.open(pdf_file.file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text.strip()
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        try:
            pdf_file.file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file.file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
        
        return ""
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


@app.post("/extract-resume-text")
async def extract_resume_text(file: UploadFile = File(...)):
    """Extract text from uploaded PDF resume file."""
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported"}
        )
    
    try:
        extracted_text = extract_text_from_pdf(file)
        if not extracted_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from PDF. Please ensure the PDF contains readable text."}
            )
        
        return {
            "success": True,
            "extracted_text": extracted_text,
            "filename": file.filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process PDF: {str(e)}"}
        )




def _extract_text_skills(text: str) -> set[str]:
    normalized = (text or "").strip().lower()
    tokens = [token for token in re.findall(r"[a-z0-9\+\#]+", normalized) if token not in STOP_WORDS]
    filtered_text = " ".join(tokens)
    return {skill for skill in KNOWN_SKILLS if skill in filtered_text}


def _compute_feature_contributions(
    feature_values: dict[str, float],
) -> dict[str, float]:
    """Compute normalized feature contributions that sum to 1 for better interpretability."""
    try:
        skill_overlap = feature_values.get("skill_overlap", 0)
        weighted_score = feature_values.get("weighted_score", 0)
        semantic_similarity = feature_values.get("semantic_similarity", 0)
        
        total = skill_overlap + weighted_score + semantic_similarity
        
        if total > 0:
            contributions = {
                "skill_overlap": round(skill_overlap / total, 2),
                "weighted_score": round(weighted_score / total, 2),
                "semantic_similarity": round(semantic_similarity / total, 2)
            }
        else:
            # Handle division by zero case
            contributions = {
                "skill_overlap": 0.0,
                "weighted_score": 0.0,
                "semantic_similarity": 0.0
            }
        
        return contributions
    except Exception as e:
        return {"error": f"Failed to compute contributions: {str(e)}"}


def _generate_human_readable_explanation(
    contributions: dict[str, float],
    skill_overlap: float,
    weighted_score: float,
    semantic_similarity: float,
) -> str:
    """Generate human-readable explanation based on feature contributions."""
    if "error" in contributions:
        return "Unable to generate explanation due to processing error."
    
    # Find dominant feature
    max_contribution = max(contributions.values()) if contributions else 0
    
    # All features are low
    if max_contribution < 0.3:
        return "Limited alignment with job requirements."
    
    # Determine dominant feature
    if contributions.get("semantic_similarity", 0) >= 0.5:
        return "Your profile aligns conceptually with the role, though exact skill matches are limited."
    elif contributions.get("weighted_score", 0) >= 0.5:
        return "Strong alignment with required skills and their importance."
    elif contributions.get("skill_overlap", 0) >= 0.5:
        return "Good match in required skills, though depth may vary."
    else:
        # Mixed contributions - provide balanced explanation
        primary = max(contributions.items(), key=lambda x: x[1])[0]
        
        if primary == "semantic_similarity":
            return "Conceptual alignment with role, complemented by some skill overlap."
        elif primary == "weighted_score":
            return "Strong skill alignment with good relevance to job requirements."
        else:  # skill_overlap
            return "Good skill match with solid foundation for the role."


def _verdict(score: float) -> str:
    if score > 0.75:
        return "Apply"
    if score > 0.5:
        return "Maybe"
    return "Not Recommended"


@app.post(
    "/match-job",
    response_model=InferenceResponse,
    summary="Score a resume against one job description",
    description="Generates semantic + skill-based features, predicts match score, and returns a verdict with missing skills.",
)
def match_job(
    payload: InferenceRequest,
    model: XGBRanker = Depends(get_ranker_model),
    embedding_model: Any = Depends(get_embedding_model),
) -> InferenceResponse:
    """Production-ready job matching endpoint with feature contribution analysis."""
    resume_text = payload.resume_text.strip().lower()
    job_description = payload.job_description.strip().lower()
    resume_skills = _extract_text_skills(resume_text)
    job_skills = _extract_text_skills(job_description)
    missing_skills = sorted(job_skills - resume_skills)

    row = pd.DataFrame(
        [
            {
                "user_skills": list(resume_skills),
                "job_skills": list(job_skills),
                "user_exp": 0,
                "job_req_exp": 0,
                "user_description": resume_text,
                "job_description": job_description,
            }
        ]
    )

    feature_df = create_feature_matrix(
        row,
        user_col="user_skills",
        job_col="job_skills",
        user_exp_col="user_exp",
        job_req_exp_col="job_req_exp",
        user_desc_col="user_description",
        job_desc_col="job_description",
    )
    
    # Use Advanced model for prediction
    raw_score = float(model.predict(feature_df[FEATURE_COLUMNS])[0])
    
    # Apply sigmoid normalization to convert to [0, 1] range
    score = 1 / (1 + np.exp(-raw_score))

    skill_overlap = float(feature_df.loc[0, "skill_overlap"])
    weighted_score = float(feature_df.loc[0, "weighted_score"])
    semantic_similarity = float(feature_df.loc[0, "semantic_similarity"])
    
    # Boost semantic similarity influence for cases where keyword matching fails but semantic meaning exists
    semantic_similarity = semantic_similarity * 3

    # Compute feature contributions using exact importance weights
    feature_values = {
        "skill_overlap": skill_overlap,
        "weighted_score": weighted_score,
        "semantic_similarity": semantic_similarity,
    }
    feature_contributions = _compute_feature_contributions(feature_values)
    
    # Calculate confidence score
    confidence_score = abs(score - 0.5) * 2
    confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp between 0 and 1
    
    # Generate human-readable explanation based on feature contributions
    explanation = _generate_human_readable_explanation(feature_contributions, skill_overlap, weighted_score, semantic_similarity)

    # Handle edge case: no skill match exists - use semantic similarity as fallback
    if skill_overlap == 0 and weighted_score == 0:
        score = semantic_similarity
        # Recalculate confidence based on semantic fallback
        confidence_score = abs(score - 0.5) * 2
        confidence_score = max(0.0, min(1.0, confidence_score))
        # Update verdict with enhanced explanation
        verdict, _ = get_verdict(score, confidence_score, feature_contributions)
        explanation = f"No direct skill matches found. {explanation}"
    else:
        # Use smart verdict from recommender for normal cases
        verdict, _ = get_verdict(score, confidence_score, feature_contributions)

    return InferenceResponse(
        match_score=round(score, 2),
        verdict=verdict,
        confidence=round(confidence_score, 3),
        feature_contributions=feature_contributions,
        explanation=explanation,
        missing_skills=missing_skills,
        skill_overlap=skill_overlap,
        weighted_score=weighted_score,
        semantic_similarity=semantic_similarity,
    )


def get_ranker_model(request: Request) -> XGBRanker:
    model = RANKER_MODEL if RANKER_MODEL is not None else getattr(request.app.state, "ranker_model", None)
    if model is None:
        raise RuntimeError("Ranker model is not loaded.")
    return model


def get_embedding_model(request: Request) -> Any:
    model = EMBEDDING_MODEL if EMBEDDING_MODEL is not None else getattr(request.app.state, "embedding_model", None)
    if model is None:
        raise RuntimeError("Embedding model is not loaded.")
    return model


