# TalentRanker-AI

An intelligent job-fit ranking system that goes beyond keyword matching using learning-to-rank models and semantic embeddings.

An AI-powered resume-to-job matching system with explainable insights, built with FastAPI, XGBoost, and Sentence Transformers.

## 🧠 System Architecture

### Data Flow Pipeline

```
Resume Text + Job Description
        ↓
    🧠 Feature Engineering
        ↓
    📊 Feature Matrix
            • skill_overlap (Jaccard similarity)
            • weighted_score (Importance-weighted skills)
            • semantic_similarity (Cosine embeddings)
        ↓
    🚀 Advanced Model (XGBoost Ranker)
        ↓
    📈 Sigmoid Normalization (0-1 range)
        ↓
    🎯 Smart Verdict Engine
            • Edge Case Detection (semantic fallback)
            • Confidence Scoring
            • Feature Contribution Analysis
        ↓
    🌐 FastAPI Response
            • Match Score (0-1 normalized)
            • Verdict (Apply/Maybe/Not Recommended)
            • Feature Contributions (sum to 1.0)
            • Human-Readable Explanation
            • Missing Skills Analysis
```

### Core Components

| Component | Technology | Purpose |
|-----------|-------------|---------|
| **🧠 Feature Engineering** | Custom Python | Extracts skills, computes overlap, weighted scores, semantic similarity |
| **🤖 Embedding Model** | Sentence Transformers | Generates semantic embeddings using `all-MiniLM-L6-v2` |
| **🚀 Ranking Model** | XGBoost Ranker | Learning-to-rank algorithm with ablation study validation |
| **🎯 Decision Engine** | Custom Logic | Smart verdicts with confidence scoring and edge case handling |
| **🌐 API Layer** | FastAPI | Production-ready REST API with automatic documentation |
| **📱 Frontend** | Tailwind CSS + Glassmorphism | Mobile-first responsive UI with animations |

### Key Innovations

🔥 **Semantic Fallback**: When `skill_overlap = 0` and `weighted_score = 0`, uses `semantic_similarity` as primary score  
📊 **Normalized Contributions**: Feature contributions sum to 1.0 for perfect interpretability  
🎯 **Smart Verdicts**: Confidence-aware recommendations with uncertainty prefixes  
📈 **Sigmoid Scaling**: Raw model outputs normalized to [0,1] range  
⚡ **3x Semantic Boost**: Amplifies semantic similarity for edge cases

### Feature Engineering Pipeline

1. **Skill Extraction**: Regex-based parsing with known skill vocabulary
2. **Skill Overlap**: Jaccard similarity between resume and job skills
3. **Weighted Score**: Importance-weighted skill matching using predefined weights
4. **Semantic Similarity**: Cosine similarity using sentence embeddings (boosted 3x)
5. **Edge Case Handling**: Semantic fallback when no skill matches exist

## Ablation Study Results

### Model Comparison

| Model | Features | NDCG@10 | Improvement |
|-------|----------|---------|-------------|
| Baseline | skill_overlap, weighted_score | 0.9681 | - |
| Advanced | skill_overlap, weighted_score, semantic_similarity | 0.9788 | **+1.11%** |

### Key Findings

- **Semantic embeddings provide measurable ranking improvement**
- **Feature importance distribution**:
  - Weighted Score: 42.6% (dominant feature)
  - Skill Overlap: 38.1% 
  - Semantic Similarity: 19.3%

### Feature Importance Analysis

```python
FEATURE_IMPORTANCE_WEIGHTS = {
    "weighted_score": 0.426,
    "skill_overlap": 0.381,
    "semantic_similarity": 0.193
}
```

## Edge Case Handling

### Problem: No Skill Match Exists

**Scenario**: Resume mentions "Deep Learning" but job asks for "Machine Learning"

**Traditional Approach**: 
- skill_overlap = 0.0
- weighted_score = 0.0
- Model prediction: 0.03 (Not Recommended)

**Enhanced Approach**:
- Detect edge case (both skill features = 0)
- Use semantic similarity as fallback: 0.523
- Final score: 0.52 (Uncertain Match: Maybe)
- **Improvement**: 1,733% increase in meaningful scoring

### Edge Case Logic

```python
if skill_overlap == 0 and weighted_score == 0:
    score = semantic_similarity  # Semantic fallback
    explanation = "No direct skill matches found. Your profile aligns conceptually..."
```

### Real-World Impact

| Test Case | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Deep Learning vs ML | 0.01 (Not Recommended) | 0.52 (Uncertain Match) | +5,100% |
| No skill overlap | 0.03 (Not Recommended) | 0.52 (Uncertain Match) | +1,733% |

## System Design Innovations

### 1. Sigmoid Normalization
```python
# Convert raw model output to [0,1] range
score = 1 / (1 + np.exp(-raw_score))
```

### 2. Semantic Similarity Boost
```python
# 3x multiplier for semantic influence
semantic_similarity = semantic_similarity * 3
```

### 3. Normalized Feature Contributions
```python
# Contributions sum to 1.0 for interpretability
total = skill_overlap + weighted_score + semantic_similarity
contributions = {
    "skill_overlap": round(skill_overlap / total, 2),
    "weighted_score": round(weighted_score / total, 2),
    "semantic_similarity": round(semantic_similarity / total, 2)
}
```

### 4. Human-Readable Explanations

| Dominant Feature | Explanation |
|------------------|-------------|
| Semantic (>=50%) | "Your profile aligns conceptually with the role, though exact skill matches are limited." |
| Weighted (>=50%) | "Strong alignment with required skills and their importance." |
| Skill (>=50%) | "Good match in required skills, though depth may vary." |
| All Low (<30%) | "Limited alignment with job requirements." |

### 5. Updated Verdict Thresholds

```python
# Normalized score thresholds (0-1 scale)
if score > 0.7:
    verdict = "Apply"
elif score > 0.4:
    verdict = "Maybe"
else:
    verdict = "Not Recommended"
```

## API Documentation

### Endpoints

#### POST `/match-job`
Analyze resume-to-job match quality with enhanced explainability.

**Request:**
```json
{
  "resume_text": "I have experience in Deep Learning and Neural Networks",
  "job_description": "Looking for a Machine Learning Engineer"
}
```

**Response:**
```json
{
  "match_score": 0.52,
  "verdict": "Uncertain Match: Maybe",
  "confidence": 0.046,
  "feature_contributions": {
    "skill_overlap": 0.00,
    "weighted_score": 0.00,
    "semantic_similarity": 1.00
  },
  "explanation": "No direct skill matches found. Your profile aligns conceptually with the role, though exact skill matches are limited.",
  "missing_skills": ["machine learning"],
  "skill_overlap": 0.0,
  "weighted_score": 0.0,
  "semantic_similarity": 0.523
}
```

#### GET `/health`
Health check endpoint for deployment monitoring.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "ranker": true,
    "embeddings": true
  }
}
```

#### GET `/`
Interactive web interface for job matching.

## Key Insights

### Why Weighted Score Dominates

The **weighted_score** feature emerged as the top contributor (42.6%) because:

1. **Skill Relevance Weighting**: Different skills have different importance values
2. **Contextual Matching**: Weights reflect real-world job requirements
3. **Discriminative Power**: Better distinguishes between qualified and unqualified candidates

```python
SKILL_WEIGHTS = {
    "python": 2.0,
    "machine learning": 2.5,
    "sql": 1.5,
}
```

### Semantic Similarity Value

While semantic similarity contributed less (19.3%) in the ablation study, it becomes **critical in edge cases**:

- **Traditional Failure**: "Deep Learning" vs "Machine Learning" = No match
- **Enhanced Success**: Semantic similarity = 0.523, meaningful recommendation
- **Real-World Value**: Handles related concepts across different terminology

### Confidence Scoring

Confidence is calculated as: `abs(match_score - 0.5) * 2`

- **High Confidence** (>0.7): Strong, reliable predictions
- **Low Confidence** (<0.3): "Uncertain Match" prefix added to verdict
- **Edge Case Awareness**: Low confidence triggers manual review recommendation

## Installation & Setup

### Local Development

1. **Clone and Setup**
```bash
git clone <repository>
cd job-match-engine
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Train Models**
```bash
python -m training.train
```

4. **Run API Server**
```bash
python -m app.main
```

5. **Access Interface**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Web Interface: http://localhost:8000

### Docker Deployment

1. **Build Image**
```bash
docker build -t job-match-engine .
```

2. **Run Container**
```bash
docker run -p 8000:8000 job-match-engine
```

### Cloud Deployment (Render/Railway)

1. **Push to GitHub** with all files
2. **Connect Repository** to deployment platform
3. **Set Environment Variables** (if needed)
4. **Deploy** - platform will build from Dockerfile

## File Structure

```
job-match-engine/
|-- app/                           # FastAPI application
|   |-- __init__.py
|   |-- main.py                 # Main API server
|   |-- api/                    # API layer
|   |   |-- routes.py           # API endpoints & logic
|   |-- core/                    # Core business logic
|   |   |-- features.py       # Feature engineering
|   |   |-- embeddings.py     # Semantic similarity
|   |   |-- recommender.py    # Smart verdict logic
|   |-- utils.py              # Utility functions
|-- training/                       # Training experiments
|   |-- __init__.py
|   |-- train.py                # Ablation study training
|   |-- train_model.py          # Model training utilities
|   |-- experiments.py          # Training experiments & analysis
|-- models/
|   |-- model_advanced.json     # Trained XGBRanker model
|   |-- model_baseline.json     # Baseline model for comparison
|   |-- skill_weights.json      # Feature importance weights
|-- data/
|   |-- training_data.csv       # Training dataset
|   |-- ablation_results.json   # Study results
|   |-- generate_data.py        # Data generation script
|-- static/
|   |-- index.html             # Web interface
|-- requirements.txt            # Python dependencies
|-- Dockerfile                  # Container configuration
|-- .dockerignore              # Docker exclusions
```

## Model Performance

### Metrics
- **NDCG@10**: 0.9788 (Advanced Model)
- **Latency**: ~358ms local inference
- **Accuracy**: High confidence predictions (>0.7) for qualified matches
- **Edge Case Success**: 1,733% improvement in meaningful scoring

### Training Dataset
- **Samples**: 1,262 job-resume pairs
- **Users**: 100 unique candidates
- **Jobs**: 1,262 unique positions
- **Split**: 80% train, 20% validation (user-grouped)

## Production Features

### Explainability
- **Normalized Contributions**: Feature contributions sum to 1.0 (100%)
- **Human-Readable Explanations**: Professional, user-friendly feedback
- **Confidence Scoring**: Prediction reliability assessment
- **Edge Case Transparency**: Clear indication when semantic fallback is used

### Reliability
- **Health Checks**: `/health` endpoint for monitoring
- **Error Handling**: Graceful failure responses
- **Input Validation**: Pydantic schema validation
- **Fallback Logic**: Robust edge case handling

### Performance
- **Model Caching**: Preloaded models on startup
- **Async Processing**: FastAPI async endpoints
- **Lightweight Deployment**: Optimized Docker image

## Training & Experiments

### Training Pipeline

```bash
# Run complete ablation study
python -m training.train

# Run specific experiments
python -m training.experiments --experiment ablation
python -m training.experiments --experiment analysis --verbose
```

### Available Experiments

1. **Ablation Study**: Compare baseline vs advanced models
2. **Feature Analysis**: Analyze feature importance and contributions

### Training Results

- **NDCG@10**: 0.9788 (Advanced Model)
- **Improvement**: +1.11% over baseline
- **Feature Importance**: Weighted Score (42.6%), Skill Overlap (38.1%), Semantic Similarity (19.3%)

## Future Enhancements

1. **Advanced NLP**: Named entity recognition for skill extraction
2. **Learning Weights**: Dynamic skill weight learning from data
3. **Multi-modal**: Resume PDF parsing and image analysis
4. **Candidate Ranking**: Batch job recommendations for users
5. **Feedback Loop**: User feedback for model improvement

## Key Insight

Traditional keyword-based systems fail when terminology differs.

Example:
- Resume: "Deep Learning"
- Job: "Machine Learning"

Baseline:
```
Skill Overlap: 0%
Recommendation: Not Recommended
```

TalentRanker-AI:
```
Semantic Similarity: 0.85
Recommendation: Maybe (context-aware)
```

This demonstrates how semantic embeddings improve real-world ranking beyond exact keyword matching.

## UI Preview

![TalentRanker-AI Interface](https://via.placeholder.com/800x400/1e293b/f8fafc?text=TalentRanker-AI+Interface)

*Modern web interface with PDF upload, semantic analysis, and explainable AI insights*

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.

---

**Technical Case Study Summary**: This project demonstrates the critical importance of edge case handling in production ML systems. The semantic fallback mechanism achieves a 1,733% improvement in meaningful scoring for cases where traditional keyword matching fails, proving that understanding meaning beyond exact matches is essential for real-world job matching applications.
