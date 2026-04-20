"""Feature engineering utilities for a job recommendation system."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

from .embeddings import compute_similarity


# Predefined skill weights (can be expanded over time).
SKILL_WEIGHTS: dict[str, float] = {
    "python": 2.0,
    "machine learning": 2.5,
    "sql": 1.5,
}


def _normalize_skills(skills: Iterable[str] | str | None) -> set[str]:
    """Normalize incoming skills into a lowercase, trimmed set."""
    if skills is None:
        return set()

    if isinstance(skills, str):
        return set(parse_skill_string(skills))

    return {
        str(skill).strip().lower()
        for skill in skills
        if skill is not None and str(skill).strip()
    }


def _compute_skill_overlap(user_set: set[str], job_set: set[str]) -> float:
    """Compute |user ∩ job| / |job| with safe zero-division handling."""
    if not job_set:
        return 0.0
    return len(user_set & job_set) / len(job_set)


def _compute_weighted_score(
    user_set: set[str],
    job_set: set[str],
    skill_weights: Mapping[str, float],
) -> float:
    """Compute weighted match over required job skills.

    Formula:
        sum(weights for matched job skills) / sum(weights for all job skills)
    If a skill is missing from the weight map, it defaults to 1.0.
    """
    if not job_set:
        return 0.0

    matched = user_set & job_set
    total_weight = sum(skill_weights.get(skill, 1.0) for skill in job_set)
    if total_weight == 0:
        return 0.0

    matched_weight = sum(skill_weights.get(skill, 1.0) for skill in matched)
    return matched_weight / total_weight


def _compute_experience_gap(user_exp: int, job_req_exp: int) -> float:
    """Score experience fit from 0.0 to 1.0."""
    if job_req_exp <= 0:
        return 1.0
    if user_exp >= job_req_exp:
        return 1.0
    return max(0.0, 1.0 - ((job_req_exp - user_exp) / job_req_exp))


def compute_semantic_similarity(user_desc: str, job_desc: str) -> float:
    """Return cosine similarity between user and job descriptions."""
    return compute_similarity(user_desc, job_desc)


def compute_individual_features(
    user_skills: Sequence[str] | None,
    job_skills: Sequence[str] | None,
    user_exp: int = 0,
    job_req_exp: int = 0,
    user_description: str = "",
    job_description: str = "",
    skill_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Compute features for one user-job pair.

    Includes skill overlap, weighted skill match, experience gap, and semantic score.
    """
    weights = skill_weights or SKILL_WEIGHTS
    user_set = _normalize_skills(user_skills)
    job_set = _normalize_skills(job_skills)

    semantic_similarity = compute_semantic_similarity(user_description, job_description)
    return {
        "skill_overlap": _compute_skill_overlap(user_set, job_set),
        "weighted_score": _compute_weighted_score(user_set, job_set, weights),
        "experience_gap": _compute_experience_gap(int(user_exp), int(job_req_exp)),
        "semantic_similarity": semantic_similarity,
        # Backward compatibility for earlier naming.
        "semantic_score": semantic_similarity,
    }


def create_feature_matrix(
    data: pd.DataFrame,
    user_col: str = "user_skills",
    job_col: str = "job_skills",
    user_exp_col: str = "user_exp",
    job_req_exp_col: str = "job_req_exp",
    user_desc_col: str = "user_description",
    job_desc_col: str = "job_description",
    skill_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute features for each row in a dataset.

    Refactored to use DataFrame.apply(axis=1) instead of iterrows.
    """
    weights = skill_weights or SKILL_WEIGHTS

    required_cols = [user_col, job_col]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    def _row_features(row: pd.Series) -> dict[str, float]:
        return compute_individual_features(
            user_skills=row.get(user_col),
            job_skills=row.get(job_col),
            user_exp=int(row.get(user_exp_col, 0) or 0),
            job_req_exp=int(row.get(job_req_exp_col, 0) or 0),
            user_description=str(row.get(user_desc_col, row.get("user_desc", "")) or ""),
            job_description=str(row.get(job_desc_col, row.get("job_desc", "")) or ""),
            skill_weights=weights,
        )

    feature_series = data.apply(_row_features, axis=1)
    return pd.DataFrame(
        feature_series.tolist(),
        columns=[
            "skill_overlap",
            "weighted_score",
            "experience_gap",
            "semantic_similarity",
            "semantic_score",
        ],
    )


# Backward-compatible wrappers for earlier API names.
def compute_features(
    user_skills: Sequence[str] | None,
    job_skills: Sequence[str] | None,
    skill_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    return compute_individual_features(
        user_skills=user_skills,
        job_skills=job_skills,
        user_exp=0,
        job_req_exp=0,
        user_description="",
        job_description="",
        skill_weights=skill_weights,
    )


def compute_batch_features(
    data: pd.DataFrame,
    user_col: str = "user_skills",
    job_col: str = "job_skills",
    skill_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    return create_feature_matrix(
        data=data,
        user_col=user_col,
        job_col=job_col,
        skill_weights=skill_weights,
    )
