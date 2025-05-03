from src.data_generation import make_synthetic_features
import pandas as pd

def test_feature_shapes():
    df = pd.read_csv("data/raw/transfusion.data", header=None,
                     names=["recency_months","frequency","monetary_cc",
                            "time_months","donated_again"])
    out = make_synthetic_features(df)
    assert out.shape[1] == 10     # 5 original + 5 synthetic
    assert out["email_open_rate"].between(0,1).all()