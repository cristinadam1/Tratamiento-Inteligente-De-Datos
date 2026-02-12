# Tratamiento Inteligente de Datos

Repositorio de proyectos y prácticas desarrollados para la asignatura de Tratamiento Inteligente de Datos (Máster en Ingeniería Informática). El contenido cubre desde el preprocesamiento hasta la implementación de modelos de aprendizaje supervisado, no supervisado y técnicas de explicabilidad.

## Contenido del Repositorio

### Machine Learning y Análisis de Datos
* **House Pricing Kaggle.ipynb**: Implementación de un modelo de regresión para la predicción de precios de viviendas utilizando el dataset de la competición de Kaggle.
* **Procesamiento de Datos, Clustering y Reglas de Asociación.ipynb**: 
    * Preprocesamiento y limpieza (outliers, valores faltantes, normalización).
    * Segmentación de datos mediante K-means y DBSCAN.
    * Extracción de patrones con algoritmos de reglas de asociación (Apriori/FP-Growth).
* **Clasificación (Medical Triage)**: Modelado para la clasificación de niveles de urgencia médica (Acuity: Low, Medium, High) basado en signos vitales y datos demográficos.

### Procesamiento de Lenguaje Natural (NLP)
* **Text Mining Codigo.ipynb**: Clasificación binaria de noticias (Fake vs Real).
    * Limpieza de texto con Regex y NLTK (Stopwords, Lemmatization).
    * Vectorización mediante TF-IDF y Sentence Transformers (Embeddings).

### IA Explicable (XAI)
* **eXplainable Artificial Intelligence_ LIME & SHAP.pdf**: Trabajo de teoría sobre la interpretación de modelos de caja negra mediante técnicas de atribución de características.

## Tecnologías Utilizadas
* **Lenguaje**: Python
* **Librerías**: Pandas, NumPy, Scikit-learn, NLTK, Sentence-Transformers, Matplotlib, Seaborn.
* **Técnicas**: LIME, SHAP, K-means, DBSCAN, TF-IDF.

## Estructura de Archivos
```text
├── Datasets/                 # Conjuntos de datos locales y procesados
├── House Pricing Kaggle.ipynb # Regresión
├── Text Mining Codigo.ipynb  # NLP y Clasificación de texto
├── Procesamiento de Datos... # Clustering y Asociación
├── *.pdf                     # Documentación técnica y presentaciones (XAI, NLP, Clasificación)
