import pytest
import numpy as np
from src.forecast.baseline_model import LogisticBaseline, RandomForestBaseline
from src.forecast.sequence_model import LSTMForecaster, build_model
from src.forecast.explain import compute_shap
from src.forecast.evaluate import evaluate


def test_baseline_and_sequence_models(tmp_path):
    X = np.random.randn(20, 10, 4)
    y = np.random.randint(0, 2, size=20)

    # 1. Logistic
    clf1 = LogisticBaseline()
    clf1.fit(X, y)
    probs1 = clf1.predict_proba(X)
    assert len(probs1) == 20

    # Save & load
    model_file = tmp_path / "model.pkl"
    clf1.save(model_file)
    loaded1 = LogisticBaseline.load(model_file)
    assert len(loaded1.predict_proba(X)) == 20

    # 2. RandomForest
    clf2 = RandomForestBaseline(n_estimators=10)
    clf2.fit(X, y)
    probs2 = clf2.predict_proba(X)
    assert len(probs2) == 20

    # 3. LSTM via factory
    lstm = build_model("lstm", hidden_dim=16, epochs=1)
    lstm.fit(X, y)
    probs3 = lstm.predict_proba(X)
    assert len(probs3) == 20

    # SHAP
    shap_df = compute_shap(clf2, X, feature_names=["f1", "f2", "f3", "f4"])
    assert not shap_df.empty

    # Evaluation
    metrics = evaluate(y, probs2)
    assert "tpr" in metrics
    assert "far" in metrics
    assert "roc_auc" in metrics
