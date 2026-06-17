"""
Generación de data drift y concept drift para el modelo de churn.

Propósito:
- Producir conjuntos de datos sintéticos que simulan cambios en la población
  (data drift) o en la relación entrada-salida (concept drift).
- Guardar los datasets generados en data/ para su posterior evaluación.

Basado en la función generar_datos_sinteticos de entrenar_modelo.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Rutas relativas al proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_FILE = DATA_DIR / "reference_training_data.csv"  # generado por entrenar_modelo.py
DATA_DRIFT_FILE = DATA_DIR / "data_drift_dataset.csv"
CONCEPT_DRIFT_FILE = DATA_DIR / "concept_drift_dataset.csv"

def generar_data_drift(n_registros: int = 800, semilla: int = 123) -> pd.DataFrame:
    """
    Genera un dataset con data drift:
    - Antigüedad: desplazada a valores más altos (media +10 meses)
    - Cargo mensual: media aumentada en 30%
    - Reclamos: mayor frecuencia de valores altos (distribución sesgada)
    
    La función de riesgo (probabilidad de churn) se mantiene igual que en el
    entrenamiento original.
    """
    rng = np.random.default_rng(seed=semilla)
    
    # Data drift: parámetros modificados
    # Antigüedad: originalmente Uniform(1,73) -> ahora Uniform(20, 100)
    antiguedad = rng.integers(20, 101, size=n_registros)
    # Cargo mensual: original Uniform(20,150) -> ahora Uniform(50, 250)
    cargo_mensual = rng.uniform(50, 250, size=n_registros)
    # Reclamos: original Poisson? original era Integer(0,7) -> ahora aumentamos rango y frecuencia
    # Usamos distribución binomial negativa para más valores altos
    reclamos = rng.negative_binomial(2, 0.5, size=n_registros)
    reclamos = np.clip(reclamos, 0, 20)  # límite superior 20
    
    # Misma regla sintética de churn que en entrenar_modelo.py
    puntaje_riesgo = (
        -0.045 * antiguedad
        + 0.025 * cargo_mensual
        + 0.65 * reclamos
        - 1.8
    )
    probabilidad = 1 / (1 + np.exp(-puntaje_riesgo))
    churn = rng.binomial(1, probabilidad)
    
    df = pd.DataFrame({
        "antiguedad": antiguedad,
        "cargo_mensual": cargo_mensual,
        "reclamos": reclamos,
        "churn": churn
    })
    return df

def generar_concept_drift(n_registros: int = 800, semilla: int = 456) -> pd.DataFrame:
    """
    Genera un dataset con concept drift:
    - La distribución de las características se mantiene similar a la original.
    - La relación se invierte: ahora los reclamos tienen un peso negativo
      (clientes que reclaman mucho son más leales) y la antigüedad pierde influencia.
    """
    rng = np.random.default_rng(seed=semilla)
    
    # Características con distribución similar al entrenamiento original
    antiguedad = rng.integers(1, 73, size=n_registros)
    cargo_mensual = rng.uniform(20, 150, size=n_registros)
    reclamos = rng.integers(0, 8, size=n_registros)
    
    # Concept drift: cambian los coeficientes
    # Antes: riesgo = -0.045*antiguedad + 0.025*cargo + 0.65*reclamos - 1.8
    # Ahora: los reclamos reducen el riesgo, la antigüedad ya no protege tanto,
    #        el cargo mensual tiene más peso.
    puntaje_riesgo = (
        -0.010 * antiguedad          # casi no afecta
        + 0.060 * cargo_mensual      # peso duplicado
        - 0.40 * reclamos            # ¡los reclamos ahora protegen!
        - 1.5
    )
    probabilidad = 1 / (1 + np.exp(-puntaje_riesgo))
    churn = rng.binomial(1, probabilidad)
    
    df = pd.DataFrame({
        "antiguedad": antiguedad,
        "cargo_mensual": cargo_mensual,
        "reclamos": reclamos,
        "churn": churn
    })
    return df

def guardar_datasets():
    """Genera ambos tipos de deriva y los guarda en archivos CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    
    if not REFERENCE_FILE.exists():
        print(f"ADVERTENCIA: No se encontró {REFERENCE_FILE}. Ejecute primero 'src/entrenar_modelo.py'.")
    else:
        print(f"Referencia cargada desde {REFERENCE_FILE}")
    
    df_data_drift = generar_data_drift()
    df_data_drift.to_csv(DATA_DRIFT_FILE, index=False)
    print(f"Data drift generado: {DATA_DRIFT_FILE}")
    
    df_concept_drift = generar_concept_drift()
    df_concept_drift.to_csv(CONCEPT_DRIFT_FILE, index=False)
    print(f"Concept drift generado: {CONCEPT_DRIFT_FILE}")

if __name__ == "__main__":
    guardar_datasets()