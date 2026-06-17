"""
Detección de data drift y concept drift mediante PSI y pruebas KS.

PSI: Mide cambios en la distribución de cada característica.
KS test: Compara distribuciones continuas y devuelve p-valor.
Además, se evalúa la caída de rendimiento (accuracy) como indicador de concept drift.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCE_FILE = DATA_DIR / "reference_training_data.csv"
MODEL_PATH = MODELS_DIR / "modelo_churn_v1.joblib"

def calcular_psi(esperadas, actuales, buckets=10):
    """
    Calcula el Population Stability Index (PSI) para una variable.
    Esperadas y actuales son arrays o Series.
    """
    # Crear bins basados en los percentiles de la distribución esperada
    esperadas = np.array(esperadas)
    actuales = np.array(actuales)
    
    # Eliminar nulos
    esperadas = esperadas[~np.isnan(esperadas)]
    actuales = actuales[~np.isnan(actuales)]
    
    percentiles = np.percentile(esperadas, np.linspace(0, 100, buckets+1))
    percentiles[0] = -np.inf
    percentiles[-1] = np.inf
    
    esperadas_bins = np.histogram(esperadas, bins=percentiles)[0]
    actuales_bins = np.histogram(actuales, bins=percentiles)[0]
    
    # Proporciones
    prop_esp = esperadas_bins / len(esperadas)
    prop_act = actuales_bins / len(actuales)
    
    # Evitar log(0)
    prop_esp = np.where(prop_esp == 0, 0.0001, prop_esp)
    prop_act = np.where(prop_act == 0, 0.0001, prop_act)
    
    psi = np.sum((prop_act - prop_esp) * np.log(prop_act / prop_esp))
    return psi

def detectar_drift_en_features(ref_df, new_df, features):
    """
    Devuelve un diccionario con PSI y KS p-valor para cada feature.
    """
    from scipy.stats import ks_2samp
    
    resultados = {}
    for col in features:
        ref_vals = ref_df[col].dropna()
        new_vals = new_df[col].dropna()
        
        # PSI
        psi = calcular_psi(ref_vals, new_vals)
        # KS test
        ks_stat, ks_p = ks_2samp(ref_vals, new_vals)
        
        resultados[col] = {
            "PSI": round(psi, 4),
            "KS_pvalor": round(ks_p, 4),
            "drift_severidad": "Alto" if psi > 0.25 else ("Moderado" if psi > 0.1 else "Bajo")
        }
    return resultados

def evaluar_caida_rendimiento(new_df, modelo_path=MODEL_PATH):
    """
    Evalúa el modelo en el nuevo dataset y compara con la métrica de referencia.
    Devuelve la diferencia de accuracy.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    
    modelo = joblib.load(MODEL_PATH)
    X_new = new_df[["antiguedad", "cargo_mensual", "reclamos"]]
    y_new = new_df["churn"]
    
    y_pred = modelo.predict(X_new)
    acc_nueva = accuracy_score(y_new, y_pred)
    
    # Cargar accuracy de referencia guardado en metadatos
    import json
    meta_path = MODELS_DIR / "modelo_churn_v1_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        acc_ref = meta["metricas"]["accuracy"]
    else:
        # fallback: leer del archivo docs/metricas_modelo.md
        metrics_md = PROJECT_ROOT / "docs" / "metricas_modelo.md"
        if metrics_md.exists():
            with open(metrics_md, "r") as f:
                for line in f:
                    if "Accuracy:" in line:
                        acc_ref = float(line.split(":")[1].strip())
                        break
        else:
            acc_ref = 0.825  # valor por defecto conocido
    
    diferencia = acc_ref - acc_nueva
    return {
        "accuracy_referencia": acc_ref,
        "accuracy_nuevo": acc_nueva,
        "caida_accuracy": round(diferencia, 4),
        "concept_drift_sospechoso": diferencia > 0.05
    }

def reporte_drift_completo(new_dataset_path, es_concept_drift=False):
    """
    Genera un reporte completo de drift comparando el dataset de referencia
    contra el nuevo dataset.
    """
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"No existe referencia: {REFERENCE_FILE}. Ejecute primero src/entrenar_modelo.py")
    
    ref_df = pd.read_csv(REFERENCE_FILE)
    new_df = pd.read_csv(new_dataset_path)
    
    features = ["antiguedad", "cargo_mensual", "reclamos"]
    
    print("="*60)
    print(f"Análisis de drift: {new_dataset_path.name}")
    print("="*60)
    
    # 1. Drift de características
    drift_features = detectar_drift_en_features(ref_df, new_df, features)
    print("\n--- Data drift por variable ---")
    for feat, res in drift_features.items():
        print(f"{feat}: PSI={res['PSI']} ({res['drift_severidad']}), KS p-valor={res['KS_pvalor']}")
    
    # 2. Evaluación de impacto en el modelo (concept drift)
    impacto = evaluar_caida_rendimiento(new_df)
    print("\n--- Impacto en rendimiento (posible concept drift) ---")
    print(f"Accuracy referencia: {impacto['accuracy_referencia']:.4f}")
    print(f"Accuracy nuevo conjunto: {impacto['accuracy_nuevo']:.4f}")
    print(f"Caída: {impacto['caida_accuracy']:.4f}")
    if impacto["concept_drift_sospechoso"]:
        print(">> ALERTA: Caída significativa de accuracy (>5%) - Posible concept drift")
    else:
        print("No se detectó caída significativa de accuracy.")
    
    return {
        "data_drift": drift_features,
        "impacto_rendimiento": impacto
    }

if __name__ == "__main__":
    # Ejemplo de uso con los datasets generados por drift_generator.py
    data_drift_path = DATA_DIR / "data_drift_dataset.csv"
    concept_drift_path = DATA_DIR / "concept_drift_dataset.csv"
    
    if data_drift_path.exists():
        reporte_drift_completo(data_drift_path)
    if concept_drift_path.exists():
        reporte_drift_completo(concept_drift_path)