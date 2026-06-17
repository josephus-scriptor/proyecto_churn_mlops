# **Instrucciones de ejecución – Generación y detección de drift**
Este documento explica paso a paso cómo **replicar escenarios de data drift y concept drift** en el proyecto `proyecto_churn_mlops`, utilizando los nuevos módulos `src/drift_generator.py` y `src/drift_detector.py`.

## Requisitos previos
- Proyecto clonado y ubicado en la carpeta `proyecto_churn_mlops`.
- Entorno virtual `.venv` creado y activado:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate          # Windows
  source .venv/bin/activate       # Linux/Mac
  ```

Dependencias instaladas:
```bash
pip install -r requirements.txt
```

Modelo entrenado y referencia de entrenamiento disponible.
Si aún no lo ha hecho, ejecute:
```bash
python src/entrenar_modelo.py
```

Este comando genera:
- `models/modelo_churn_v1.joblib`
- `data/reference_training_data.csv` (conjunto de entrenamiento de referencia)

## Paso 1 – Generar datasets con drift
Los scripts de generación producen dos archivos CSV en la carpeta data/:
- `data_drift_dataset.csv`  Cambios en la distribución de las características.
- `concept_drift_dataset.csv`  Cambios en la relación entre características y la variable objetivo.

Ejecute:
```bash
python src/drift_generator.py
```
Salida esperada:
```text
Referencia cargada desde ...\data\reference_training_data.csv
Data drift generado: ...\data\data_drift_dataset.csv
Concept drift generado: ...\data\concept_drift_dataset.csv
Opcional: Puede modificar los parámetros (número de registros, semillas) editando las funciones dentro de drift_generator.py.
```

## Paso 2 – Detectar y cuantificar el drift
El módulo `drift_detector.py` compara cada conjunto generado con la referencia de entrenamiento, calculando:
- PSI (Population Stability Index) para cada variable.
- Prueba de Kolmogorov Smirnov (p valor).
- Caída de accuracy del modelo sobre el conjunto con deriva.

Ejecute:
```bash
python src/drift_detector.py
```
Resultado esperado (extracto):
```text
============================================================
Análisis de drift: data_drift_dataset.csv
============================================================

--- Data drift por variable ---
antiguedad: PSI=0.4321 (Alto), KS p-valor=0.0000
cargo_mensual: PSI=0.5123 (Alto), KS p-valor=0.0000
reclamos: PSI=0.2987 (Alto), KS p-valor=0.0000

--- Impacto en rendimiento (posible concept drift) ---
Accuracy referencia: 0.8250
Accuracy nuevo conjunto: 0.7100
Caída: 0.1150
>> ALERTA: Caída significativa de accuracy (>5%) - Posible concept drift

============================================================
Análisis de drift: concept_drift_dataset.csv
============================================================

--- Data drift por variable ---
antiguedad: PSI=0.0234 (Bajo), KS p-valor=0.5432
cargo_mensual: PSI=0.0189 (Bajo), KS p-valor=0.6721
reclamos: PSI=0.0312 (Bajo), KS p-valor=0.4812

--- Impacto en rendimiento ---
Accuracy referencia: 0.8250
Accuracy nuevo conjunto: 0.6500
Caída: 0.1750
>> ALERTA: Caída significativa de accuracy (>5%) - Posible concept drift
```

Interpretación rápida:
- data_drift_dataset → alto PSI en todas las variables → data drift severo.
- concept_drift_dataset → bajo PSI pero caída de accuracy → concept drift.

## Paso 3 – Ejecutar la API monitorizada dentro de Docker

*Requisitos previos:*
- Cree un Dockerfile nuevo para el proyecto.
- Recomendación: Antes de ejecutar este paso, asegúrese de que la API esté activa:
```bash
python -m uvicorn api.main:app --reload
```
- Detener la API local con Ctrl + C para liberar el puerto 8000.
- Confirmar que Docker Desktop se encuentra activo.

1. Construir la imagen (Solo la primera vez).
```bash
docker build --no-cache -t churn-api-thenier .
```
2. Eliminar el contenedor anterior si existe.
```bash
docker rm -f churn-api-thenier
```
3. Ejecutar un contenedor nuevo.
```bash
docker run -d --name churn-api-thenier -p 127.0.0.1:8008:8008 churn-api-thenier
```
4. Verificar el contenedor.
```bash
docker ps
```

## Paso 4 – Simular el envío de datos drift a la API
Si desea observar cómo la API maneja valores atípicos y cómo se refleja en las métricas de monitoreo (/metrics), puede usar el script `tests/simular_solicitudes.py` (modificar el puerto a 8008). No obstante, para enviar masivamente los datos de drift.

Ejecute:
```bash
docker exec -it churn-api-thenier python tests/enviar_drift_api.py
```
Resultado esperado (extracto):
```text
200 []
200 ['antiguedad=75 fuera del rango histórico [1, 72]']
200 []
200 ['cargo_mensual=165.16754622395118 fuera del rango histórico [20.0, 150.0]']
200 ['antiguedad=93 fuera del rango histórico [1, 72]', 'cargo_mensual=203.38442243171465 fuera del rango histórico [20.0, 150.0]']
200 ['cargo_mensual=241.89940274026964 fuera del rango histórico [20.0, 150.0]']
```

## Paso 5 – Analizar los logs y las métricas de la API
Los logs se guardan en `logs/monitor_api.log`.
Ejecute:
```bash
docker exec churn-api-thenier cat logs/monitor_api.log | Select-String "Valores fuera de rango hist"
```
Resultado esperado (extracto):
```text
2026-06-15 17:08:14,032 | WARNING | Valores fuera de rango histÃ³rico: ['antiguedad=75 fuera del rango histÃ³rico [1, 72]']
2026-06-15 17:08:14,039 | WARNING | Valores fuera de rango histÃ³rico: ['cargo_mensual=165.16754622395118 fuera del rango histÃ³rico [20.0, 150.0]']
```
Para conocer el resto de los valores revisar el reporte [Metricas](http://127.0.0.1:8008/metrics) de la API.

## Resumen de comandos útiles
| **Acción**                                | **Comando**                              |
|-------------------------------------------|------------------------------------------|
| Entrenar modelo y crear referencia        | `python src/entrenar_modelo.py`          |
| Generar datasets con drift                | `python src/drift_generator.py`          |
| Detectar drift (PSI + KS + accuracy)      | `python src/drift_detector.py`           |
| Construir la imagen de Docker             | `docker build --no-cache -t churn-api-thenier .`|
| Eliminar el contenedor anterior si existe | `docker rm -f churn-api-thenier`         |
| Ejecutar un contenedor nuevo              | `docker run -d --name churn-api-thenier -p 127.0.0.1:8008:8008 churn-api-thenier`|
| Verificar el contenedor                   | `docker ps`|
| Simular tráfico con drift                 | `docker exec -it churn-api-thenier python tests/enviar_drift_api.py` |

