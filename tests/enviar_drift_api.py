import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8008"
df = pd.read_csv("data/data_drift_dataset.csv").head(10)

for _, row in df.iterrows():
    payload = {
        "antiguedad": int(row["antiguedad"]),
        "cargo_mensual": float(row["cargo_mensual"]),
        "reclamos": int(row["reclamos"])
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print(resp.status_code, resp.json().get("alertas_datos"))