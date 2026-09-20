"""Genera una muestra pequeña y reproducible de cuentas para el MVP."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "paysim1" / "PS_20174392719_1491204439457_log.csv"
OUTPUT_DIR = ROOT / "mvp_assets"
CASE_FILE = ROOT / "artifacts" / "interpretability_cases.csv"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"No se encontró PaySim en {SOURCE}")

    columns = ["step", "type", "amount", "nameDest", "isFraud"]
    frame = pd.read_csv(SOURCE, usecols=columns)
    frame = frame[frame["nameDest"].str.startswith("C")].copy()
    frame.sort_values(["nameDest", "step"], kind="mergesort", inplace=True)

    recent = frame.groupby("nameDest", sort=False).tail(24)
    summary = recent.groupby("nameDest", sort=False).agg(
        window_fraud=("isFraud", "max"), transactions=("step", "size")
    )

    case_accounts = pd.read_csv(CASE_FILE)["Cuenta"].tolist()
    positive_pool = summary[
        (summary["window_fraud"] == 1) & (~summary.index.isin(case_accounts))
    ]
    normal_pool = summary[
        (summary["window_fraud"] == 0)
        & (summary["transactions"] >= 4)
        & (~summary.index.isin(case_accounts))
    ]

    extra_positive = positive_pool.sample(n=8, random_state=42).index.tolist()
    extra_normal = normal_pool.sample(n=12, random_state=42).index.tolist()
    selected_accounts = case_accounts + extra_positive + extra_normal
    sample = frame[frame["nameDest"].isin(selected_accounts)].copy()

    OUTPUT_DIR.mkdir(exist_ok=True)
    sample.to_csv(OUTPUT_DIR / "account_transactions.csv", index=False)
    profiles = (
        sample.groupby("nameDest", sort=False)
        .agg(
            total_transactions=("step", "size"),
            reference_label=("isFraud", "max"),
        )
        .reset_index()
    )
    profiles["study_case"] = profiles["nameDest"].isin(case_accounts)
    profiles.to_csv(OUTPUT_DIR / "account_profiles.csv", index=False)
    print(
        f"Muestra creada: {len(profiles)} cuentas y {len(sample)} transacciones "
        f"en {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
