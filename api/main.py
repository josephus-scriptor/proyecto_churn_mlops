"""
API de predicción de churn con FastAPI.

La API carga un modelo serializado, valida los datos de entrada
y devuelve una predicción junto con su probabilidad.
Expone endpoints de observabilidad: /info.
"""

import time
import platform
from pathlib import Path
from datetime import datetime
from collections import deque

from typing import Dict, Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "Joseph Thenier Oyola"
SERVICE_NAME = "churn-prediction-service"
ENVIRONMENT = "development"  # Cambiar a production/staging según despliegue
API_VERSION = "1.0.0"

# Intento de obtener el commit SHA desde Git (opcional)
def get_git_commit_sha() -> str:
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        return "unknown"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unknown"
    with open(head_path, "r") as f:
        head_content = f.read().strip()
    if head_content.startswith("ref: "):
        ref_path = git_dir / head_content[5:]
        if ref_path.exists():
            with open(ref_path, "r") as f:
                return f.read().strip()[:7]
    return head_content[:7]

GIT_COMMIT_SHA = get_git_commit_sha()

# Timestamp de inicio del servicio
START_TIME = datetime.now()
START_TIMESTAMP = START_TIME.isoformat()

# Dependencias (simplificado - solo verifica que el modelo existe)
def check_dependencies_status() -> Dict[str, str]:
    return {
        "model_storage": "ok" if MODEL_PATH.exists() else "failed",
        "database": "not_configured",  # En producción se conectaría a una DB real
        "feature_store": "not_applicable"
    }

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)
DEPLOYMENT_TIME = datetime.now().isoformat()

# ---------- Clases Pydantic ----------
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

# ---------- FastAPI app ----------
app = FastAPI(
    title="API de predicción de churn",
    description="Servicio académico ML-Ops para estimar riesgo de abandono.",
    version=API_VERSION,
)

# ---------- Endpoints básicos ----------
@app.get("/")
def inicio() -> dict[str, str]:
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": AUTOR,
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
    }

# ---------- Nuevo endpoint de información ----------
@app.get("/info")
def info() -> dict[str, Any]:
    """Información del sistema y metadata del despliegue."""
    uptime_seconds = (datetime.now() - START_TIME).total_seconds()
    return {
        "author": AUTOR,
        "service_name": SERVICE_NAME,
        "environment": ENVIRONMENT,
        "api_version": API_VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
        "current_server_time": datetime.now().isoformat(),
        "git_commit_sha": GIT_COMMIT_SHA,
        "dependencies": check_dependencies_status(),
    }

# ---------- Endpoint de predicción ----------
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
