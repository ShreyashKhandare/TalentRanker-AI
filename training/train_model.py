"""Train baseline and semantic-enhanced XGBRanker models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker
from .features import SKILL_WEIGHTS

BASELINE_FEATURE_COLUMNS = ["skill_overlap", "weighted_score"]
V2_FEATURE_COLUMNS = ["skill_overlap", "weighted_score", "semantic_similarity"]
TARGET_COLUMN = "relevance_score"
GROUP_COLUMN = "user_id"
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH_V2 = BASE_DIR / "models" / "ranker_model_v2.json"
SKILL_WEIGHTS_PATH = BASE_DIR / "models" / "skill_weights.json"


def _validate_training_frame(df: pd.DataFrame) -> None:
    required = {GROUP_COLUMN, "job_id", *V2_FEATURE_COLUMNS, TARGET_COLUMN}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _split_by_user_groups(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data so each user's jobs stay in one split only."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COLUMN]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _mean_group_ndcg(df_eval: pd.DataFrame, preds: np.ndarray, k: int = 10) -> float:
    """Compute mean NDCG@k over user groups."""
    scored = df_eval[[GROUP_COLUMN, TARGET_COLUMN]].copy()
    scored["pred"] = preds

    scores: list[float] = []
    for _, group in scored.groupby(GROUP_COLUMN, sort=False):
        # Need at least 2 rows for meaningful ranking metrics.
        if len(group) < 2:
            continue
        y_true = group[TARGET_COLUMN].to_numpy().reshape(1, -1)
        y_pred = group["pred"].to_numpy().reshape(1, -1)
        scores.append(float(ndcg_score(y_true, y_pred, k=min(k, len(group)))))

    return float(np.mean(scores)) if scores else 0.0


def _build_ranker() -> XGBRanker:
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        # LambdaRank-specific configuration.
        lambdarank_num_pair_per_sample=8,
    )


def _train_for_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[XGBRanker, float]:
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    group_train = train_df.groupby(GROUP_COLUMN).size().to_list()

    x_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN]
    group_val = val_df.groupby(GROUP_COLUMN).size().to_list()

    model = _build_ranker()
    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_val, y_val)],
        eval_group=[group_val],
        verbose=False,
    )

    val_preds = model.predict(x_val)
    val_ndcg = _mean_group_ndcg(val_df, val_preds, k=10)
    return model, val_ndcg


def _print_feature_importance(model: XGBRanker, feature_columns: list[str]) -> None:
    importances = model.feature_importances_
    ranked = sorted(zip(feature_columns, importances), key=lambda pair: pair[1], reverse=True)
    print("Feature importances (highest to lowest):")
    for name, value in ranked:
        print(f"  - {name}: {value:.6f}")


def train_ranker(df: pd.DataFrame) -> tuple[XGBRanker, float, float]:
    """Train baseline and v2 rankers, return v2 model and NDCGs."""
    _validate_training_frame(df)

    # Sort by user to keep contiguous sessions in fit() order.
    df_sorted = df.sort_values([GROUP_COLUMN, "job_id"]).reset_index(drop=True)
    train_df, val_df = _split_by_user_groups(df_sorted)
    train_df = train_df.sort_values([GROUP_COLUMN, "job_id"]).reset_index(drop=True)
    val_df = val_df.sort_values([GROUP_COLUMN, "job_id"]).reset_index(drop=True)

    _, baseline_ndcg = _train_for_features(train_df, val_df, BASELINE_FEATURE_COLUMNS)
    model_v2, ndcg_v2 = _train_for_features(train_df, val_df, V2_FEATURE_COLUMNS)
    model_v2.save_model(MODEL_PATH_V2.as_posix())
    return model_v2, baseline_ndcg, ndcg_v2


def main() -> None:
    data_path = Path("training_data.csv")
    if not data_path.exists():
        raise FileNotFoundError("training_data.csv not found. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    model_v2, baseline_ndcg, ndcg_v2 = train_ranker(df)
    _print_feature_importance(model_v2, V2_FEATURE_COLUMNS)
    improvement = ndcg_v2 - baseline_ndcg
    with SKILL_WEIGHTS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(SKILL_WEIGHTS, fp, indent=2)
    print(f"Baseline NDCG@10 (without embeddings): {baseline_ndcg:.4f}")
    print(f"V2 NDCG@10 (with semantic_similarity): {ndcg_v2:.4f}")
    print(f"NDCG improvement delta: {improvement:+.4f}")
    print(f"Saved upgraded model to: {MODEL_PATH_V2}")
    print(f"Saved skill weights to: {SKILL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
