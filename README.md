# Optimización de Campañas de Telemarketing Bancario mediante Machine Learning

Este proyecto aplica y compara diversas técnicas de Machine Learning para predecir si un cliente bancario contratará un depósito a plazo fijo (`variable y`) tras una campaña de comunicación directa por teléfono. El objetivo de negocio es optimizar los recursos de la entidad financiera seleccionando perfiles comerciales de forma inteligente.

---

## Fuente de Datos

El conjunto de datos utilizado es **Bank Marketing**, extraído directamente del repositorio **UCI Machine Learning Repository** (ID: 222).
* **Instancias totales:** 45,211 registros.
* **Variables (16 características + 1 target):** Incluye datos demográficos del cliente (edad, tipo de trabajo, estado civil, nivel educativo), métricas financieras (balance anual medio, préstamos activos) y datos de contacto de campañas previas y actuales.
* **Variable Objetivo (`y`):** Clasificación binaria que indica si el cliente contrató el depósito (`yes` o `no`).

---

## Técnicas de Machine Learning y Preprocesamiento

A lo largo del desarrollo se han implementado las siguientes estrategias para transformar los datos y optimizar los modelos predictivos:

1. **Análisis Diagnóstico Inicial:** Evaluación de dimensiones (`.shape`), tipos de datos e identificación de nulos en variables clave como `job`, `education`, `contact` y `poutcome`.
2. **Filtrado Numérico Inicial:** Pruebas preliminares aislando únicamente las variables cuantitativas (`int64` y `float64`) para evaluar algoritmos de proximidad.
3. **Tratamiento Avanzado de Variables Categóricas:** Aplicación de **Target Encoding** para convertir variables de texto a valores numéricos ponderados según la relación directa con la variable objetivo.
4. **Tratamiento del Desbalance de Clases (SMOTE):** Debido al fuerte sesgo del dataset (muchas más respuestas "no" que "yes"), se utilizó **SMOTE** (*Synthetic Minority Over-sampling Technique*) exclusivamente sobre el conjunto de entrenamiento para balancear artificialmente la clase minoritaria.
5. **Entrenamiento de Modelos Base:** Implementación inicial de *K-Neighbors Classifier (KNN)*, *Regresión Logística* y *Árboles de Decisión (Decision Tree)*.
6. **Métodos de Ensamble (*Ensemble Learning*):** Escalado del proyecto mediante el uso de algoritmos potentes como *Bagging Classifier*, *Random Forest Classifier*, *AdaBoost Classifier* y *Gradient Boosting Classifier*.
7. **Ajuste de Hiperparámetros Personalizado:**
   * **GridSearchCV:** Búsqueda exhaustiva en rejilla con validación cruzada de 5 pliegues (`cv=5`).
   * **RandomizedSearchCV:** Búsqueda aleatoria masiva expandiendo los límites (evaluando hasta un espacio extendido de miles de árboles y profundidades de hasta 110 nodos).
   * **Métrica de Optimización:** Para maximizar la identificación de interesados evitándose valores nulos o sesgos en el target de texto, se configuró una función de puntuación personalizada con `make_scorer(f1_score, pos_label='yes')`.

---

## Tabla Comparativa de Resultados

Rendimiento definitivo medido sobre el conjunto de test independiente:

| Métrica | Regr. Logística (SMOTE) | Árbol Decisión (SMOTE) | Bagging Base | Bagging Grid Search | Bagging Random Search | Random Forest Base | Random Forest Grid Search | Random Forest Random Search | Gradient Boosting | AdaBoost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.8393 | 0.7787 | 0.8253 | 0.8479 | 0.8504 | 0.8533 | 0.8591 | **0.8603** | 0.8336 | 0.8190 |
| **Precision**| 0.4135 | 0.3289 | 0.3932 | 0.4202 | 0.4306 | 0.4354 | 0.4421 | **0.4514** | 0.4050 | 0.3814 |
| **Recall** | 0.7929 | 0.8020 | **0.8249** | 0.6856 | 0.7452 | 0.7287 | 0.6398 | 0.7324 | 0.8084 | 0.8048 |
| **F1-Score** | 0.5435 | 0.4665 | 0.5325 | 0.5211 | 0.5458 | 0.5451 | 0.5228 | **0.5585** | 0.5396 | 0.5175 |

### Conclusiones y Hallazgos Clave
* **El Modelo Más Equilibrado (Negocio Eficiente):** El algoritmo **Random Forest optimizado con Random Search** obtuvo el mayor F1-Score (**0.5585**), logrando la máxima precisión (**45.14%**) y un Accuracy del **86.03%**. Es la opción óptima si la entidad quiere ahorrar costes telefónicos, evitando el 55% de llamadas innecesarias mientras asegura capturar una gran proporción de los clientes positivos.
* **El Modelo de Captación Máxima (Volumen de Negocio):** Si el banco dispone de recursos de telemarketing masivos y prioriza capturar hasta el último depósito posible, el **Bagging Base** sigue destacando por registrar el **Recall más alto de todo el estudio (82.49%)**.

---

## Guía de Ejecución Secuencial del Notebook

Dado que las fases de modelado requirieron experimentación cruzada, para ejecutar el archivo `Proyecto-ML.ipynb` de forma lineal y libre de errores de dependencias de variables, asegúrese de seguir estrictamente este **orden lógico descendente (de arriba a abajo)**:

1. **Fase 1: Conexión a la API de UCI e Importación de Librerías:** Celdas iniciales que instalan `ucimlrepo`, cargan las dependencias principales (`pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`) y descargan el dataset original.
2. **Fase 2: Diagnóstico y Limpieza Inicial:** Ejecución de funciones automatizadas de exploración para revisar nulos, dimensiones de las matrices `X` e `y`, y una primera prueba rápida de rendimiento basándose solo en variables numéricas con KNN.
3. **Fase 3: Ingeniería de Características y Balanceo:** Bloque donde se aplica el tratamiento de cadenas de texto con *Target Encoding* y se realiza la separación de conjuntos (*train/test split*) previo a aplicar **SMOTE** sobre los datos de entrenamiento.
4. **Fase 4: Modelos Clasificadores Base:** Celdas correspondientes al entrenamiento y cálculo de métricas para la Regresión Logística y el Árbol de Decisión Inicial.
5. **Fase 5: Modelos de Ensamble (Avanzados):** Entrenamiento con hiperparámetros por defecto de los algoritmos Bagging, Random Forest, AdaBoost y Gradient Boosting.
6. **Fase 6: Optimización mediante GridSearchCV:** Bloque de código destinado al ajuste acotado e inspección de los mejores parámetros (ej. `n_estimators` y `max_depth`) en los modelos Bagging y Random Forest.
7. **Fase 7: Optimización Avanzada mediante RandomizedSearchCV:** Celdas finales que realizan la exploración en rangos extendidos (escalando parámetros hasta las 2,000 estimaciones). Al final del todo, se generan las matrices de confusión definitivas y el diccionario `datos_finalisimos` que compila la tabla de métricas comparativas del proyecto.

** Link a la presentación: https://docs.google.com/presentation/d/184lrEEX2bjzYz7fn8l8lR1MVCu40m3zdPbePshpAv5M/edit?usp=sharing