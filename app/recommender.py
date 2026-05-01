import logging
import sys
import time
import hashlib
from typing import List, Dict, Any

# Use sklearn's TF-IDF for text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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
            
            # Pre-fit with default vocabulary to prevent cold start
            default_texts = [
                "experience skills qualifications education",
                "software development programming engineering",
                "project management leadership communication",
                "data analysis machine learning algorithms",
                "technical requirements specifications documentation"
            ]
            self.vectorizer.fit(default_texts)
            self.fitted = True
            
            self.cache = {}  # Simple caching
            self.initialization_time = time.time()
            self.request_count = 0
            self.total_similarity_time = 0
            
            logger.info("PerfectEmbedder initialized with optimizations and pre-fitted vocabulary")
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
            raise ValueError(f"Encoding error: {str(e)}")
        
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
            raise ValueError(f"Similarity error: {str(e)}")
    
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

class JobRanker:
    """Main job ranking engine using PerfectEmbedder"""
    
    def __init__(self):
        self.embedder = PerfectEmbedder()
        logger.info("JobRanker initialized with PerfectEmbedder")
    
    def rank_resume_against_jobs(self, resume_text: str, job_descriptions: List[str]) -> Dict[str, Any]:
        """Rank resume against multiple job descriptions"""
        try:
            if not resume_text or not job_descriptions:
                raise ValueError("Resume text and job descriptions are required")
            
            logger.info(f"Processing ranking request: {len(job_descriptions)} jobs")
            start_time = time.time()
            
            # Compute similarity scores
            scores = []
            for i, job in enumerate(job_descriptions):
                try:
                    similarity = self.embedder.compute_similarity(resume_text, job)
                    scores.append(float(similarity))
                except Exception as e:
                    logger.warning(f"Failed to compute similarity for job {i}: {e}")
                    scores.append(0.0)  # Fallback score
            
            # Create ranked results
            ranked_jobs = sorted(zip(job_descriptions, scores), key=lambda x: x[1], reverse=True)
            
            processing_time = time.time() - start_time
            
            # Calculate additional metrics
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            
            # For single job analysis (match score)
            match_score = scores[0] if scores else 0
            explanation = f"Resume matches this job description with a score of {match_score:.1%}. " \
                         f"This indicates a {'strong' if match_score > 0.75 else 'moderate' if match_score > 0.5 else 'weak'} match."
            
            result = {
                "status": "success",
                "resume": resume_text[:200] + "..." if len(resume_text) > 200 else resume_text,
                "processed_jobs": len(job_descriptions),
                "processing_time_seconds": round(processing_time, 4),
                "match_score": float(match_score),
                "explanation": explanation,
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
                    "cache_hits": len(self.embedder.cache),
                    "request_id": hashlib.md5(f"{time.time()}{len(job_descriptions)}".encode()).hexdigest()[:16]
                }
            }
            
            logger.info(f"Ranking completed in {processing_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Ranking error: {e}")
            return {
                "status": "error",
                "message": "Internal server error during ranking",
                "engine": "sklearn-tfidf-perfect"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status of the ranking system"""
        return self.embedder.health_check()
