# Proyecto Churn MLOps

Este proyecto corresponde a una práctica inicial del módulo de MLOps.

El objetivo es construir una estructura básica de trabajo para un proyecto de Machine Learning que permita:

- Preparar datos.
- Entrenar un modelo.
- Evaluar métricas.
- Guardar el modelo entrenado.
- Exponer el modelo mediante una API.
- Ejecutar pruebas básicas.

## Problema del proyecto

Se trabajará con un caso simplificado de predicción de abandono de clientes, conocido como churn.

El modelo intentará predecir si un cliente podría abandonar un servicio, utilizando variables como edad, antigüedad, saldo promedio, reclamos y uso de aplicación móvil.

## Estructura del proyecto

```text
proyecto_churn_mlops
|   .dockerignore
|   .gitignore
|   Dockerfile
|   INSTRUCCIONES_EJECUCION.md
|   README.md
|   requirements.txt
|               
+---api
|       main.py
|       main_sesion6_backup.py
|       __init__.py
|           
+---data
|      descripcion_dataset.md 
|
+---docs
|       metricas_modelo.md
|       
+---logs
|       monitor_api.log
|       
+---models
|       modelo_churn.pkl
|       modelo_churn_v1.joblib
|       modelo_churn_v1_metadata.json
|       
+---notebooks
|       .gitkeep
|       
+---src
|       drift_detector.py
|       drift_generator.py
|       entrenar_modelo.py
|       evaluar_modelo.py
|       preparar_datos.py
|       __init__.py
|       
\---tests
|       enviar_drift_api.py
|       simular_solicitudes.py
|       test_api.py
|       __init__.py
|   
```

## Carpetas principales

- `data`: contiene los datos del proyecto.
- `notebooks`: contiene análisis exploratorios.
- `src`: contiene los scripts principales del modelo.
- `models`: contiene el modelo entrenado.
- `api`: contiene la API del modelo.
- `tests`: contiene pruebas automáticas.
- `docs`: contiene documentación y métricas.

## Flujo inicial del proyecto

El flujo básico será:

1. Preparar los datos.
2. Entrenar el modelo.
3. Evaluar el modelo.
4. Guardar las métricas.
5. Crear una API básica.
6. Probar el funcionamiento inicial.

## Implementación de Drift

Este proyecto implementa un flujo completo de MLOps para predicción de churn, incluyendo:
- Entrenamiento de modelo (Logistic Regression).
- API predictiva con monitoreo básico (logs, latencia, métricas en memoria).
- **Detección de data drift y concept drift** (PSI, KS test, caída de accuracy).
- Generación de datasets sintéticos con deriva para simular entornos cambiantes.
Se utiliza el conjunto de entrenamiento guardado como `reference_training_data.csv` para comparar con nuevos lotes de datos.  
Métricas implementadas:
- **PSI (Population Stability Index)** – umbrales: <0.1 (bajo), 0.1-0.25 (moderado), >0.25 (alto).
- **Prueba de Kolmogorov‑Smirnov** – p-valor < 0.05 indica distribuciones diferentes.
- **Caída de accuracy** – diferencia >5% sugiere possible concept drift.

### Resumen de cambios y nuevas capacidades

| **Componente**               | **Acción realizada**                                                                 |
|------------------------------|--------------------------------------------------------------------------------------|
| `src/drift_generator.py`     | Nuevo archivo – genera datasets con data drift y concept drift.                     |
| `src/drift_detector.py`      | Nuevo archivo – calcula PSI, KS test, caída de accuracy, reporta deriva.            |
| `src/entrenar_modelo.py`     | Modificado: exporta `data/reference_training_data.csv` y estadísticas de referencia.|
| `src/evaluar_modelo.py`      | Modificado: integra llamada al detector de drift al finalizar la evaluación.        |
| `INSTRUCCIONES_EJECUCION.md` | Actualizado con pasos para generación y detección de drift.                         |
| `README.md`                  | Actualizado con nueva sección de drift.                                             |

Con estas adiciones, el proyecto cumple con:
- Generación de data drift y concept drift.
- Integración en el flujo existente (sin romper la funcionalidad anterior).
- Detección estadística de drift que complementa el monitoreo básico de la API.