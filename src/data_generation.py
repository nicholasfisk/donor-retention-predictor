"""
data_generation.py
------------------
Takes the raw UCI Blood-Transfusion file (`transfusion.data`) and
adds five synthetic engagement features, then writes
`transfusion_augmented.csv`.

Run:
    python src/data_generation.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with extra engagement columns attached."""
    df = df.copy()

    # 1. email_open_rate: 0–1, loosely correlated with donation frequency
    df["email_open_rate"] = np.clip(
        np.random.normal(0.30 + 0.015 * df["frequency"], 0.08), 0.0, 1.0
    )

    # 2. events_attended: Poisson, λ rises with frequency
    lam = 0.20 + 0.03 * df["frequency"]
    df["events_attended"] = np.random.poisson(lam).astype(int)

    # 3. sms_opt_in flag: probability rises slightly with frequency
    sms_p = np.clip(0.35 + 0.01 * df["frequency"], 0.0, 1.0)
    df["sms_opt_in"] = np.random.binomial(n=1, p=sms_p)

    # 4. last_campaign_days: days since last fundraising appeal
    campaign_offset = np.random.randint(5, 180, size=len(df))
    df["last_campaign_days"] = np.maximum(0, campaign_offset - df["recency_months"])

    # 5. newsletter_segment: quartiles of lifetime donation volume
    df["newsletter_segment"] = pd.qcut(
        df["monetary_cc"],
        4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
    )

    return df


def main(args):
    # Raw UCI file has no header
    base_cols = [
        "recency_months",
        "frequency",
        "monetary_cc",
        "time_months",
        "donated_again",
    ]
    df_raw = pd.read_csv(args.input, header=None, names=base_cols)
    
    # --- ensure numeric types ---------------------------------------------
    numeric_cols = ["recency_months", "frequency", "monetary_cc", "time_months"]
    df_raw[numeric_cols] = df_raw[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    df_raw = df_raw.dropna(subset=numeric_cols)

    df_aug = make_synthetic_features(df_raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(args.output, index=False)
    print(f"Augmented dataset written to {args.output}  (rows: {len(df_aug)})")


if __name__ == "__main__":
    default_in = Path("data/raw/transfusion.data")
    default_out = Path("data/processed/transfusion_augmented.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=default_in,  type=Path)
    parser.add_argument("--output", default=default_out, type=Path)
    args = parser.parse_args()

    main(args)

