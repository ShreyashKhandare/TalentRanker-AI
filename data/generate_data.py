"""Generate synthetic training data for ranking."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("training_data.csv")

SKILL_POOL = [
    "python",
    "machine learning",
    "sql",
    "docker",
    "pytorch",
    "tensorflow",
    "statistics",
    "data visualization",
    "aws",
    "nlp",
]


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_synthetic_data(
    n_users: int = 100,
    min_jobs_per_user: int = 10,
    max_jobs_per_user: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic ranking data with per-user job groups."""
    random.seed(seed)
    np.random.seed(seed)

    rows: list[dict[str, float | int | str]] = []
    job_counter = 1

    for user_id in range(1, n_users + 1):
        num_jobs = random.randint(min_jobs_per_user, max_jobs_per_user)
        user_exp = random.randint(0, 8)

        for _ in range(num_jobs):
            overlap = float(np.random.beta(2.0, 2.0))
            weighted = float(_bounded(overlap + np.random.normal(0, 0.12), 0.0, 1.0))
            semantic_similarity = float(_bounded(0.7 * overlap + np.random.normal(0, 0.15), 0.0, 1.0))
            job_req_exp = random.randint(0, 8)

            # Relevance combines core signals + small exp compatibility + noise.
            exp_fit = 1.0 if user_exp >= job_req_exp else (1.0 - ((job_req_exp - user_exp) / max(job_req_exp, 1)))
            latent = (
                0.35 * overlap
                + 0.30 * weighted
                + 0.25 * semantic_similarity
                + 0.10 * exp_fit
                + np.random.normal(0, 0.08)
            )
            latent = _bounded(latent, 0.0, 1.0)
            relevance = int(round(latent * 5))

            rows.append(
                {
                    "user_id": f"u{user_id:03d}",
                    "job_id": f"j{job_counter:05d}",
                    "skill_overlap": overlap,
                    "weighted_score": weighted,
                    "semantic_similarity": semantic_similarity,
                    "relevance_score": relevance,
                }
            )
            job_counter += 1

    return pd.DataFrame(rows)


def main() -> None:
    df = generate_synthetic_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
