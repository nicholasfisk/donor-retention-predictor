"""
shap_utils.py
-------------
Build a SHAP explainer with real, readable feature names
--------------------------------------------------------
"""

from pathlib import Path
import pandas as pd
import shap


BACKGROUND_CSV = Path("data/processed/transfusion_augmented.csv")


def _preprocess_with_feature_names(pipe, df):
    """Return (X_proc_dataframe, feature_names) after pipeline transform."""
    pre = pipe.named_steps["prep"]
    X_proc = pre.transform(df)
    names = pre.get_feature_names_out()         # all numeric + one-hot cols
    X_df   = pd.DataFrame(X_proc, columns=names)
    return X_df, names


def shap_explainer(trained_pipe, n_background: int = 200):
    """Return (explainer, feature_names) ready for plotting."""
    # ---------- Get / build background sample ---------------------------
    if BACKGROUND_CSV.exists():
        df_bg = (
            pd.read_csv(BACKGROUND_CSV)
            .drop(columns=["donated_again"])
            .sample(n=n_background, random_state=42)
            .reset_index(drop=True)
        )
    else:
        # fall back: regenerate on the fly
        from src.data_generation import make_synthetic_features
        raw = pd.read_csv(
            "data/raw/transfusion.data",
            header=None,
            names=[
                "recency_months",
                "frequency",
                "monetary_cc",
                "time_months",
                "donated_again",
            ],
        )
        df_bg = (
            make_synthetic_features(raw)
            .drop(columns=["donated_again"])
            .sample(n=n_background, random_state=42)
            .reset_index(drop=True)
        )

    # ---------- Pre-process and build explainer -------------------------
    X_bg_proc_df, feat_names = _preprocess_with_feature_names(trained_pipe, df_bg)

    # Strip num__/cat__ prefixes
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feat_names]

    explainer = shap.Explainer(
        trained_pipe.named_steps["clf"],
        X_bg_proc_df,
        feature_names=clean_names,   #  ← passes cleaned names
    )
    return explainer, clean_names   #  ← returns cleaned names