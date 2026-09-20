import numpy as np
import pandas as pd

from mvp_model import load_models, predict_account


def test_mvp_inference_matches_final_case():
    transactions = pd.read_csv("mvp_assets/account_transactions.csv")
    models = load_models("artifacts")
    account = transactions[transactions["nameDest"] == "C977281429"]
    prediction = predict_account(account, *models)

    assert prediction.is_alert
    # La serialización CSV de la muestra produce diferencias de redondeo pequeñas.
    assert np.isclose(prediction.probability, 0.99976844, atol=2e-4)
    assert np.isclose(prediction.anomaly_score, 1.4455553, atol=1e-3)
    assert np.isclose(prediction.transactions["attention"].sum(), 1.0, atol=1e-6)
    assert len(prediction.transactions) <= 24


def test_all_demo_accounts_produce_valid_outputs():
    transactions = pd.read_csv("mvp_assets/account_transactions.csv")
    models = load_models("artifacts")
    for _, account in transactions.groupby("nameDest"):
        prediction = predict_account(account, *models)
        assert 0 <= prediction.probability <= 1
        assert prediction.anomaly_score >= 0
        assert np.isclose(prediction.transactions["attention"].sum(), 1.0, atol=1e-5)
