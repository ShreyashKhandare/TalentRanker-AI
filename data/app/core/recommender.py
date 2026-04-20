"""Smart recommendation logic with confidence-based verdicts and explanations."""

from typing import Dict, Tuple


def get_verdict(score: float, confidence_score: float, feature_contributions: Dict[str, float]) -> Tuple[str, str]:
    """
    Generate smart verdict and explanation based on score, confidence, and feature contributions.
    
    Args:
        score: The match score from the model (normalized 0-1)
        confidence_score: Calculated confidence score (0-1)
        feature_contributions: Dictionary of feature contributions
    
    Returns:
        Tuple of (verdict, explanation)
    """
    # Base verdict logic for normalized score (0-1)
    if score > 0.7:
        base_verdict = "Apply"
    elif score > 0.4:
        base_verdict = "Maybe"
    else:
        base_verdict = "Not Recommended"
    
    # Add uncertainty prefix for low confidence
    if confidence_score < 0.3:
        verdict = f"Uncertain Match: {base_verdict}"
    else:
        verdict = base_verdict
    
    # Generate explanation based on feature contributions
    explanation = _generate_explanation(feature_contributions, confidence_score)
    
    return verdict, explanation


def _generate_explanation(feature_contributions: Dict[str, float], confidence_score: float) -> str:
    """
    Generate explanation based on which features contributed most to the score.
    
    Args:
        feature_contributions: Dictionary of feature contributions
        confidence_score: Calculated confidence score
    
    Returns:
        Explanation string
    """
    if not feature_contributions or "error" in feature_contributions:
        return "Unable to determine feature contributions for this match."
    
    # Sort features by contribution (highest first)
    sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_features:
        return "No significant feature contributions detected."
    
    top_feature, top_contribution = sorted_features[0]
    
    # Generate explanation based on top contributing feature
    if top_feature == "weighted_score":
        explanation = "Match heavily driven by your core technical skills and their relevance to the job."
    elif top_feature == "skill_overlap":
        explanation = "Match primarily based on the overlap between your skills and job requirements."
    elif top_feature == "semantic_similarity":
        explanation = "Match driven by contextual alignment between your experience and the job description."
    else:
        explanation = f"Match primarily influenced by {top_feature.replace('_', ' ')}."
    
    # Add confidence context
    if confidence_score >= 0.7:
        explanation += " High confidence in this assessment."
    elif confidence_score < 0.3:
        explanation += " Low confidence - consider reviewing the job requirements carefully."
    else:
        explanation += " Moderate confidence in this assessment."
    
    # Add secondary feature context if available
    if len(sorted_features) > 1:
        second_feature, second_contribution = sorted_features[1]
        if second_contribution > 0.1:  # Only mention if significant
            explanation += f" Secondary contribution from {second_feature.replace('_', ' ')}."
    
    return explanation


def _get_confidence_level(confidence_score: float) -> str:
    """Get human-readable confidence level."""
    if confidence_score >= 0.8:
        return "Very High"
    elif confidence_score >= 0.6:
        return "High"
    elif confidence_score >= 0.4:
        return "Moderate"
    elif confidence_score >= 0.2:
        return "Low"
    else:
        return "Very Low"


def analyze_match_quality(score: float, confidence_score: float, feature_contributions: Dict[str, float]) -> Dict[str, any]:
    """
    Comprehensive analysis of the match quality.
    
    Args:
        score: The match score from the model
        confidence_score: Calculated confidence score (0-1)
        feature_contributions: Dictionary of feature contributions
    
    Returns:
        Dictionary with comprehensive analysis
    """
    verdict, explanation = get_verdict(score, confidence_score, feature_contributions)
    
    return {
        "verdict": verdict,
        "explanation": explanation,
        "confidence_level": _get_confidence_level(confidence_score),
        "primary_driver": max(feature_contributions.items(), key=lambda x: x[1])[0] if feature_contributions else "unknown",
        "feature_balance": _assess_feature_balance(feature_contributions),
        "recommendation_strength": _get_recommendation_strength(score, confidence_score)
    }


def _assess_feature_balance(feature_contributions: Dict[str, float]) -> str:
    """Assess how balanced the feature contributions are."""
    if not feature_contributions or len(feature_contributions) < 2:
        return "insufficient_data"
    
    contributions = list(feature_contributions.values())
    max_contrib = max(contributions)
    min_contrib = min(contributions)
    
    if max_contrib == 0:
        return "no_contribution"
    
    ratio = min_contrib / max_contrib
    
    if ratio > 0.7:
        return "well_balanced"
    elif ratio > 0.4:
        return "moderately_balanced"
    else:
        return "poorly_balanced"


def _get_recommendation_strength(score: float, confidence_score: float) -> str:
    """Get the strength of the recommendation."""
    combined_score = (score + confidence_score) / 2
    
    if combined_score > 0.8:
        return "strong"
    elif combined_score > 0.6:
        return "moderate"
    elif combined_score > 0.4:
        return "weak"
    else:
        return "very_weak"
