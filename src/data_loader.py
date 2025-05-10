"""
data_loader.py

Funciones auxiliares para cargar y preprocesar los datasets empleados
en los análisis de Partial Dependence (PDP).

Coloca los CSV en:  repo_root/data/
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_bike(csv_path: str | None = None) -> pd.DataFrame:
    """Carga day.csv y añade la columna days_since_start."""
    path = Path(csv_path) if csv_path else DATA_DIR / "day.csv"
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["days_since_start"] = (df["dteday"] - df["dteday"].min()).dt.days
    return df


def load_house(csv_path: str | None = None) -> pd.DataFrame:
    """Carga kc_house_data.csv."""
    path = Path(csv_path) if csv_path else DATA_DIR / "kc_house_data.csv"
    return pd.read_csv(path)