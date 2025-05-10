"""
pdp_house.py

Entrena un RandomForestRegressor sobre kc_house_data.csv y genera
PDP 1D para bedrooms, bathrooms, sqft_living y floors.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error

from data_loader import load_house

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def _train(df: pd.DataFrame):
    features = ["bedrooms", "bathrooms", "sqft_living", "floors"]
    X = df[features]
    y = df["price"]
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, X, y, features


def run() -> None:
    df = load_house()
    model, X, y, features = _train(df)

    # Métricas
    r2 = r2_score(y, model.predict(X))
    rmse = mean_squared_error(y, model.predict(X), squared=False)
    print(f"R² house = {r2:.3f}  |  RMSE house = ${rmse:,.0f}")

    disp = PartialDependenceDisplay.from_estimator(model, X, features)
    for ax, f in zip(disp.axes_.ravel(), features):
        ax.set_xlabel(f)
        ax.set_ylabel("Predicción price")
    disp.figure_.suptitle("House prices – PDP 1D")
    disp.figure_.tight_layout()
    out = FIG_DIR / "house_pdp_1d.png"
    disp.figure_.savefig(out, dpi=300)
    print(f"✔ Saved {out}")


if __name__ == "__main__":
    run()