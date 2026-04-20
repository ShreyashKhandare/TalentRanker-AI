"""Shared utility helpers for preprocessing inputs."""

from __future__ import annotations


def _extract_text_skills(text: str) -> set[str]:
    """Extract known skills from text using regex."""
    if not isinstance(text, str) or not text.strip():
        return set()
    
    # Normalize text and extract skill mentions
    text_lower = text.lower()
    found_skills = set()
    
    # Known skills vocabulary
    KNOWN_SKILLS = {
        "python", "machine learning", "sql", "docker", "pytorch", 
        "tensorflow", "statistics", "nlp", "aws", "kubernetes",
        "spark", "java", "c++", "react"
    }
    
    for skill in KNOWN_SKILLS:
        # Check for whole skill name first
        if skill in text_lower:
            found_skills.add(skill)
        else:
            # Check for individual words in multi-word skills
            skill_words = skill.split()
            if all(word in text_lower for word in skill_words):
                found_skills.add(skill)
    
    return found_skills
