
# AI4I MLOps - Mantenimiento predictivo

Clasificación de fallas de máquinas utilizando el dataset [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset).

## Selección del modelo

Debido al desbalance de clases (3.39 % de fallas), se priorizaron métricas como Average Precision, recall y F1-score frente a accuracy.
Se compararon seis modelos mediante validación cruzada estratificada. Se seleccionó **HistGradientBoostingClassifier** por obtener el mejor Average Precision (0.7900), y posteriormente se optimizó mediante GridSearchCV (0.7922).
Resultados finales sobre el conjunto de prueba (20 %):

| Métrica | Valor |
|---|---:|
| Precision | 0.7067 |
| Recall | 0.7794 |
| F1-score | 0.7413 |
| Average Precision | 0.8479 |

El análisis y la comparación de modelos se encuentran en `notebooks/`.

## Ejecución local
Instalar dependencias:
```bash
uv sync --locked
```
Entrenar modelo:
```bash
uv run entrenar
```
Ejecutar pruebas:
```bash
uv run pytest -v
```
Iniciar FastAPI:
```bash
uv run fastapi dev src/ai4i_mlops/api.py
```
Iniciar Gradio en otra terminal:
```bash
uv run python app/gradio_app.py
```
## Integración MLOps

- **FastAPI:** servicio de predicción.
- **Gradio:** interfaz conectada a la API.
- **pytest:** pruebas de datos, modelo y API.
- **Docker:** construcción y ejecución de la aplicación.
- **GitHub Actions:** CI para entrenamiento y pruebas; CD para despliegue después de un CI exitoso.
- **Render:** despliegue de API y Gradio en dos Web Services.

## Despliegue

- **API:** https://ai4i-mlops-api.onrender.com/docs
- **Gradio:** https://ai4i-mlops-ui.onrender.com/
