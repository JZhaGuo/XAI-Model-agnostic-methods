"""
pdp_bike.py

Entrena un RandomForestRegressor sobre el dataset day.csv y genera:
1) PDP 1D de days_since_start, temp, hum, windspeed
2) PDP 2D de (temp, hum) con dispersión de densidad
Las figuras se guardan en repo_root/figures/
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error

from data_loader import load_bike

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def _train(df: pd.DataFrame):
    y = df["cnt"]
    X = df[["days_since_start", "temp", "hum", "windspeed"]]
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, X, y


def _pdp_1d(model, X):
    features = ["days_since_start", "temp", "hum", "windspeed"]
    disp = PartialDependenceDisplay.from_estimator(model, X, features)
    for ax, f in zip(disp.axes_.ravel(), features):
        ax.set_xlabel(f)
        ax.set_ylabel("Predicción cnt")
    disp.figure_.suptitle("Bike sharing – PDP 1D")
    disp.figure_.tight_layout()
    out = FIG_DIR / "bike_pdp_1d.png"
    disp.figure_.savefig(out, dpi=300)
    print(f"✔ Saved {out}")


def _pdp_2d(model, X, df):
    disp = PartialDependenceDisplay.from_estimator(
        model, X, [("temp", "hum")], grid_resolution=50
    )
    ax = disp.axes_[0, 0]
    sample = df.sample(n=min(5000, len(df)), random_state=RANDOM_STATE)
    ax.scatter(sample["temp"], sample["hum"], s=5, alpha=0.1, linewidths=0)
    ax.set_title("Bike sharing – PDP 2D temp vs hum")
    out = FIG_DIR / "bike_pdp2d_temp_hum.png"
    disp.figure_.savefig(out, dpi=300)
    print(f"✔ Saved {out}")


def run() -> None:
    df = load_bike()
    model, X, y = _train(df)

    # Métricas
    r2 = r2_score(y, model.predict(X))
    rmse = mean_squared_error(y, model.predict(X), squared=False)
    print(f"R² bike = {r2:.3f}  |  RMSE bike = {rmse:.1f} alquileres")

    _pdp_1d(model, X)
    _pdp_2d(model, X, df)


if __name__ == "__main__":
    run()