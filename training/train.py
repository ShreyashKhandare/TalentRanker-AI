"""Standalone training entrypoint with guaranteed artifact outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .features import SKILL_WEIGHTS
from .train_model import (
    BASELINE_FEATURE_COLUMNS,
    V2_FEATURE_COLUMNS,
    _print_feature_importance,
    _train_for_features,
    _validate_training_frame,
    _split_by_user_groups,
)

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
BASELINE_MODEL_PATH = BASE_DIR / "models" / "model_baseline.json"
ADVANCED_MODEL_PATH = BASE_DIR / "models" / "model_advanced.json"
SKILL_WEIGHTS_PATH = BASE_DIR / "models" / "skill_weights.json"
ABLATION_RESULTS_PATH = BASE_DIR / "data" / "ablation_results.json"


def _build_dummy_dataset() -> pd.DataFrame:
    """Build a small fallback dataset so pipeline can always run."""
    rows = []
    for uid in range(1, 11):
        user_id = f"u{uid:03d}"
        for jid, (so, ws, ss, rel) in enumerate(
            [
                (0.9, 0.95, 0.90, 5),
                (0.7, 0.75, 0.70, 4),
                (0.4, 0.45, 0.40, 2),
                (0.2, 0.25, 0.20, 1),
            ],
            start=1,
        ):
            rows.append(
                {
                    "user_id": user_id,
                    "job_id": f"j{uid:03d}_{jid:02d}",
                    "skill_overlap": so,
                    "weighted_score": ws,
                    "semantic_similarity": ss,
                    "relevance_score": rel,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Perform Ablation Study comparing Baseline vs Advanced models."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Using dataset: {DATA_PATH} ({len(df)} rows)")
    else:
        df = _build_dummy_dataset()
        print("training_data.csv not found. Using generated dummy dataset.")

    # Validate and prepare data
    _validate_training_frame(df)
    df_sorted = df.sort_values(["user_id", "job_id"]).reset_index(drop=True)
    train_df, val_df = _split_by_user_groups(df_sorted)
    train_df = train_df.sort_values(["user_id", "job_id"]).reset_index(drop=True)
    val_df = val_df.sort_values(["user_id", "job_id"]).reset_index(drop=True)

    print("Training Baseline model (skill_overlap, weighted_score)...")
    baseline_model, baseline_ndcg = _train_for_features(train_df, val_df, BASELINE_FEATURE_COLUMNS)
    baseline_model.save_model(BASELINE_MODEL_PATH.as_posix())
    _print_feature_importance(baseline_model, BASELINE_FEATURE_COLUMNS)

    print("\nTraining Advanced model (skill_overlap, weighted_score, semantic_similarity)...")
    advanced_model, advanced_ndcg = _train_for_features(train_df, val_df, V2_FEATURE_COLUMNS)
    advanced_model.save_model(ADVANCED_MODEL_PATH.as_posix())
    _print_feature_importance(advanced_model, V2_FEATURE_COLUMNS)

    # Calculate improvement metrics
    improvement = advanced_ndcg - baseline_ndcg
    improvement_percentage = ((improvement / baseline_ndcg) * 100) if baseline_ndcg > 0 else 0

    # Save ablation study results
    ablation_results = {
        "study_type": "Ablation Study - Baseline vs Advanced",
        "baseline_features": BASELINE_FEATURE_COLUMNS,
        "advanced_features": V2_FEATURE_COLUMNS,
        "baseline_ndcg": round(baseline_ndcg, 4),
        "advanced_ndcg": round(advanced_ndcg, 4),
        "improvement": round(improvement, 4),
        "improvement_percentage": round(improvement_percentage, 2),
        "baseline_feature_importances": {
            feature: round(float(importance), 6)
            for feature, importance in zip(BASELINE_FEATURE_COLUMNS, baseline_model.feature_importances_)
        },
        "advanced_feature_importances": {
            feature: round(float(importance), 6)
            for feature, importance in zip(V2_FEATURE_COLUMNS, advanced_model.feature_importances_)
        },
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "unique_users": df["user_id"].nunique() if "user_id" in df.columns else "unknown",
            "unique_jobs": df["job_id"].nunique() if "job_id" in df.columns else "unknown"
        },
        "models_saved": {
            "baseline_model": str(BASELINE_MODEL_PATH.resolve()),
            "advanced_model": str(ADVANCED_MODEL_PATH.resolve())
        }
    }

    with ABLATION_RESULTS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(ablation_results, fp, indent=2)

    with SKILL_WEIGHTS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(SKILL_WEIGHTS, fp, indent=2)

    # Print clean comparison table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"NDCG Baseline:    {baseline_ndcg:.4f}")
    print(f"NDCG Advanced:    {advanced_ndcg:.4f}")
    print(f"Delta Improvement: {improvement:+.4f} ({improvement_percentage:+.2f}%)")
    print("="*60)
    print(f"Saved Baseline model to: {BASELINE_MODEL_PATH.resolve()}")
    print(f"Saved Advanced model to: {ADVANCED_MODEL_PATH.resolve()}")
    print(f"Saved skill weights to: {SKILL_WEIGHTS_PATH.resolve()}")
    print(f"Saved ablation results to: {ABLATION_RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
