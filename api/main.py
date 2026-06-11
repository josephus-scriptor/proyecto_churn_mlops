# from pathlib import Path

# import joblib
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# BASE_DIR = Path(__file__).resolve().parents[1]
# MODEL_FILE = BASE_DIR / "models" / "modelo_churn.pkl"

# app = FastAPI(title="Servicio ML-Ops - Churn-API-Thenier")

# class Cliente(BaseModel):
#     edad: int
#     antiguedad_meses: int
#     saldo_promedio: float
#     reclamos: int
#     usa_app: int

# def cargar_modelo():
#     """
#     Carga el modelo entrenado si existe.
#     """
#     if not MODEL_FILE.exists():
#         return None

#     return joblib.load(MODEL_FILE)

# @app.get("/")
# def inicio():
#     return {
#         "mensaje": "Servicio ML-Ops activo",
#         "estado": "ok",
#         "autor": "Joseph Thenier Oyola"
#     }

# @app.get("/health")
# def health():
#     return {
#         "estado": "ok",
#         "modelo_disponible": MODEL_FILE.exists()
#     }

# @app.post("/predict")
# def predict(cliente: Cliente):
#     modelo = cargar_modelo()

#     if modelo is None:
#         raise HTTPException(
#             status_code=503,
#             detail="El modelo aún no está disponible. Primero se debe entrenar el modelo."
#         )

#     datos = pd.DataFrame([cliente.model_dump()])

#     prediccion = int(modelo.predict(datos)[0])

#     probabilidad = None
#     if hasattr(modelo, "predict_proba"):
#         probabilidad = float(modelo.predict_proba(datos)[0][1])

#     return {
#         "churn_predicho": prediccion,
#         "probabilidad_churn": probabilidad
#     }

"""
API de predicción de churn con FastAPI.

La API carga un modelo serializado, valida los datos de entrada
y devuelve una predicción junto con su probabilidad.
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "Joseph Thenier Oyola"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)

class ClienteEntrada(BaseModel):
    antiguedad: int = Field(
        ...,
        ge=0,
        le=120,
        description="Antigüedad del cliente expresada en meses",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Cargo mensual del cliente",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=50,
        description="Cantidad de reclamos recientes",
        examples=[3],
    )

class PrediccionSalida(BaseModel):
    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str

app = FastAPI(
    title="API de predicción de churn",
    description="Servicio académico ML-Ops para estimar riesgo de abandono.",
    version="1.0.0",
)

@app.get("/")
def inicio() -> dict[str, str]:
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": "Joseph Thenier Oyola",
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
    }

@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        probabilidad = float(modelo.predict_proba(X)[0][1])
        etiqueta = "alto_riesgo" if probabilidad >= 0.50 else "bajo_riesgo"

        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc
