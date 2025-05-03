import shap

def shap_explainer(trained_pipe):
    # pull out the fitted GradientBoostingClassifier
    model = trained_pipe.named_steps["clf"]
    pre   = trained_pipe.named_steps["prep"]

    # shap needs X as NumPy after preprocessing
    explainer = shap.Explainer(
        model.predict_proba,
        shap.sample(pre.transformer_list[0][2], 200)  # background ≈200 rows
    )
    return explainer
