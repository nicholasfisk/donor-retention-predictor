"""
shap_utils.py
-------------
Utility to build a SHAP explainer for the trained pipeline.
Loads 200 random rows from the augmented donor file
and passes them through the pipeline’s pre-processor so
SHAP gets the same feature space the model sees.
"""

from pathlib import Path
import pandas as pd
import shap


BACKGROUND_CSV = Path("data/processed/transfusion_augmented.csv")


def shap_explainer(trained_pipe, n_background: int = 200):
    """
    Returns a TreeExplainer (works for both GBM and logistic),
    fitted on a small, pre-processed background sample.
    """
    if not BACKGROUND_CSV.exists():
        raise FileNotFoundError(
            f"Background file {BACKGROUND_CSV} missing. "
            "Run data_generation.py first or commit the CSV."
        )

    # 1. sample rows & drop the target column
    df_bg = (
        pd.read_csv(BACKGROUND_CSV)
        .drop(columns=["donated_again"])
        .sample(n=n_background, random_state=42)
        .reset_index(drop=True)
    )

    # 2. pass through the pipeline’s pre-processor
    X_bg_proc = trained_pipe.named_steps["prep"].transform(df_bg)

    # 3. Build SHAP explainer on the *fitted model* + background
    explainer = shap.Explainer(
        trained_pipe.named_steps["clf"],
        X_bg_proc,
        algorithm="auto",
    )
    return explainer
