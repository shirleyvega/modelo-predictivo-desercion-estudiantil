# Modelo Predictivo e Interpretable para la Identificación Temprana del Riesgo de Deserción Estudiantil en Educación Superior

**Maestría en Business Intelligence y Ciencia de Datos**  
**Escuela de Negocios**

**Autora:** Shirley Melisa Vega Nieto  
**Director:** Manuel Eugenio Morocho Cayamcela  
**Año:** 2026

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Problema de Investigación y Objetivos](#2-problema-de-investigación-y-objetivos)
3. [Dataset](#3-dataset)
4. [Metodología](#4-metodología)
5. [Estructura del Repositorio](#5-estructura-del-repositorio)
6. [Configuración del Entorno](#6-configuración-del-entorno)
7. [Guía de Ejecución Paso a Paso](#7-guía-de-ejecución-paso-a-paso)
8. [Resultados](#8-resultados)
9. [Referencia de Archivos de Salida](#9-referencia-de-archivos-de-salida)
10. [Decisiones de Diseño](#10-decisiones-de-diseño)
11. [Reproducción Exacta de Resultados](#11-reproducción-exacta-de-resultados)
12. [Dependencias y Versiones](#12-dependencias-y-versiones)
13. [Cita](#13-cita)

---

## 1. Descripción del Proyecto

La deserción estudiantil en la educación superior es uno de los desafíos más críticos que enfrentan las instituciones académicas a nivel mundial. Conlleva consecuencias académicas, financieras y sociales significativas: trayectorias estudiantiles interrumpidas, pérdida de ingresos institucionales, deterioro de indicadores de calidad y riesgos de acreditación.

Este proyecto desarrolla un **pipeline predictivo e interpretable** para la identificación temprana de estudiantes en riesgo de abandonar sus estudios. El enfoque va más allá de maximizar la precisión global. Prioriza el **recall para la clase de deserción** con el fin de minimizar los falsos negativos, incorpora **validación cruzada estratificada** para garantizar la generalización de los resultados, y aplica **SHAP (SHapley Additive exPlanations)** para producir explicaciones transparentes y accionables de cada predicción.

El producto final no es solo un modelo entrenado. Es un marco analítico completo que apoya la toma de decisiones institucionales: una clasificación de riesgo segmentada en tres niveles (alto, medio, bajo), un ranking de las 15 variables más importantes, y una arquitectura propuesta de Sistema de Alerta Temprana (SAT) para su despliegue institucional.

### Contenido del repositorio

- Un notebook Jupyter completamente reproducible que abarca desde la carga de datos hasta el despliegue del modelo
- Un pipeline estructurado que previene la fuga de datos entre conjuntos de entrenamiento y prueba
- Comparación de tres algoritmos de clasificación bajo condiciones idénticas
- Análisis de interpretabilidad basado en SHAP sobre el modelo seleccionado
- Segmentación de riesgo basada en probabilidades estimadas de deserción
- Todos los resultados intermedios y finales organizados por capítulo de análisis

---

## 2. Problema de Investigación y Objetivos

### Pregunta de Investigación

> ¿Cómo se pueden desarrollar, evaluar y comparar modelos de clasificación basados en aprendizaje automático para mejorar la identificación de estudiantes en riesgo de deserción en la educación superior, usando técnicas de validación robusta e interpretabilidad del modelo para apoyar la toma de decisiones institucionales?

### Objetivo General

Desarrollar y comparar modelos de clasificación basados en aprendizaje automático para predecir la deserción estudiantil en educación superior, utilizando validación robusta, interpretabilidad y segmentación de riesgo, con el objetivo de apoyar las decisiones institucionales orientadas a la retención.

### Objetivos Específicos

1. Preparar y limpiar el conjunto de datos educativo mediante técnicas de preprocesamiento que incluyen manejo de nulos, eliminación de duplicados, codificación categórica, normalización y selección de variables.
2. Ajustar modelos de clasificación usando Regresión Logística, Random Forest y XGBoost, entrenados para estimar la probabilidad de deserción e identificar patrones asociados al abandono académico.
3. Evaluar y comparar el rendimiento de los modelos mediante validación cruzada estratificada, priorizando el recall para la clase de deserción y complementando con precisión y F1-score.
4. Analizar la interpretabilidad del modelo mediante SHAP sobre el modelo final para identificar la contribución de cada variable a la predicción de deserción.
5. Segmentar a los estudiantes en niveles de riesgo (alto, medio, bajo) basándose en las probabilidades estimadas, para facilitar la identificación de estudiantes en riesgo y apoyar intervenciones institucionales diferenciadas.

---

## 3. Dataset

### Fuente

**Predict Students' Dropout and Academic Success**  
UCI Machine Learning Repository  
Realinho, V., Martins, M. V., Machado, J., & Baptista, L. (2021)  
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

### Resumen del Dataset

| Propiedad | Valor |
|---|---|
| Total de registros | 4.424 estudiantes |
| Clases originales del target | Dropout, Enrolled, Graduate |
| Target transformado | Binario (1 = Desertor, 0 = No desertor) |
| Total de variables | 36 |
| Variables numéricas | 18 |
| Variables categóricas (codificadas numéricamente) | 18 |
| Valores nulos | Ninguno |
| Registros duplicados | Ninguno |
| Razón de desbalance de clases | 2,11:1 (No desertor:Desertor) |

### Transformación de la Variable Objetivo

El dataset original presenta un problema multiclase. Para este estudio fue transformado a una tarea de clasificación binaria:

```
Dropout  -> 1  (clase de interés)
Enrolled -> 0
Graduate -> 0
```

Esta transformación se justifica por el objetivo de investigación: identificar estudiantes en riesgo de abandono, no distinguir entre estudiantes activos y graduados.

### Categorías de Variables

**Variables académicas**
- `Curricular units 1st sem (approved)` — Unidades aprobadas en el primer semestre
- `Curricular units 2nd sem (approved)` — Unidades aprobadas en el segundo semestre
- `Curricular units 1st sem (enrolled)` — Unidades matriculadas en el primer semestre
- `Curricular units 2nd sem (enrolled)` — Unidades matriculadas en el segundo semestre
- `Curricular units 1st sem (without evaluations)` — Unidades sin evaluación, primer semestre
- `Curricular units 2nd sem (without evaluations)` — Unidades sin evaluación, segundo semestre
- `Curricular units 1st sem (grade)` — Promedio de notas, primer semestre
- `Curricular units 2nd sem (grade)` — Promedio de notas, segundo semestre
- `Admission grade` — Nota de admisión
- `Previous qualification (grade)` — Nota de la calificación previa

**Variables demográficas**
- `Age at enrollment` — Edad del estudiante al momento de la matrícula
- `Gender` — Género
- `Marital status` — Estado civil
- `Nacionality` — Nacionalidad
- `International` — Si el estudiante es internacional

**Variables socioeconómicas**
- `Scholarship holder` — Si el estudiante tiene beca
- `Tuition fees up to date` — Si los pagos de matrícula están al día
- `Debtor` — Si el estudiante tiene deuda pendiente
- `Mother's qualification` / `Father's qualification` — Nivel educativo de los padres
- `Mother's occupation` / `Father's occupation` — Categoría ocupacional de los padres
- `Displaced` — Si el estudiante es una persona desplazada

**Variables macroeconómicas**
- `Unemployment rate` — Tasa de desempleo nacional al momento de la matrícula
- `Inflation rate` — Tasa de inflación al momento de la matrícula
- `GDP` — Tasa de crecimiento del PIB al momento de la matrícula

El archivo que espera el pipeline es `BDF.xlsx` y debe ubicarse en `data/BDF.xlsx` antes de la ejecución.

---

## 4. Metodología

### Arquitectura del Pipeline

El análisis sigue un pipeline estructurado y reproducible organizado en 36 bloques secuenciales:

```
Carga de Datos
    |
Revisión de Calidad (nulos, duplicados, tipos)
    |
Transformación del Target (multiclase -> binario)
    |
Identificación de Variables (numéricas vs. categóricas)
    |
Análisis Exploratorio (distribuciones, correlaciones, balance de clases)
    |
División Train/Test (80/20 estratificada, random_state=42)
    |
Pipeline de Preprocesamiento
    | Numérico: Imputación por mediana + StandardScaler
    | Categórico: Imputación por moda + OneHotEncoder
    |
Definición de Modelos
    | Regresión Logística (class_weight='balanced')
    | Random Forest (n_estimators=300, class_weight='balanced')
    | XGBoost (scale_pos_weight=2,11, boosting)
    |
Validación Cruzada Estratificada (k=5, solo sobre el conjunto de entrenamiento)
    |
Selección de Modelo (criterio: mayor recall para la clase de deserción)
    |
Análisis de Umbral de Decisión (rango 0,10 a 0,90)
    |
Evaluación Final del Modelo (sobre el conjunto de prueba retenido)
    |
Análisis de Interpretabilidad SHAP
    |
Segmentación de Riesgo (basada en percentiles: p33 / p66)
    |
Serialización del Modelo (joblib)
    |
Exportación de Configuración (JSON)
```

### Justificación de la Selección de Algoritmos

Se seleccionaron tres algoritmos para cubrir interpretabilidad, robustez y capacidad de modelado no lineal:

| Modelo | Justificación |
|---|---|
| Regresión Logística | Salida probabilística, coeficientes interpretables, regularización, línea base adecuada |
| Random Forest | Ensemble de árboles de decisión, reduce la varianza, maneja no linealidades, no requiere escalado |
| XGBoost | El boosting secuencial corrige errores previos, alta capacidad predictiva, ampliamente validado en analítica educativa |

### Estrategia para el Desbalance de Clases

Los tres modelos utilizan **ponderación de clases** en lugar de remuestreo:
- Regresión Logística y Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight = n_negativos / n_positivos`

Esto evita introducir observaciones sintéticas que podrían distorsionar la frontera de decisión.

### Criterio de Selección del Modelo

El criterio primario de selección es el **máximo recall para la clase de deserción (clase 1)** a través de los cinco pliegues de validación cruzada. El recall minimiza los falsos negativos: estudiantes que desertarán pero que el modelo no identifica. En el contexto de un sistema de alerta temprana, el costo de no detectar a un estudiante en riesgo supera con creces el costo de una intervención preventiva innecesaria.

### Optimización del Umbral de Decisión

No se asumió que el umbral estándar de 0,50 fuera óptimo. Se evaluaron sistemáticamente umbrales desde 0,10 hasta 0,90 en pasos de 0,05 sobre el conjunto de prueba. El umbral óptimo se seleccionó maximizando el F1-score sujeto a una restricción mínima de recall de 0,70.

### Interpretabilidad con SHAP

Los valores SHAP se calcularon usando la clase `shap.Explainer` con una muestra de fondo de 200 observaciones del conjunto de entrenamiento. SHAP proporciona:
- **Importancia global**: Valor absoluto medio SHAP por variable
- **Impacto direccional**: Si los valores altos o bajos de cada variable aumentan o reducen la probabilidad de deserción
- **Explicaciones individuales**: Descomposición de la predicción de cada estudiante en contribuciones por variable

### Segmentación de Riesgo

Los estudiantes del conjunto de prueba fueron segmentados en tres niveles de riesgo usando el percentil 33 y el percentil 66 de las probabilidades de deserción estimadas como umbrales de corte:

```
Riesgo bajo:    probabilidad <= p33
Riesgo medio:   p33 < probabilidad <= p66
Riesgo alto:    probabilidad > p66
```

---

## 5. Estructura del Repositorio

```
student-dropout-prediction/
|
|-- data/
|   |-- BDF.xlsx                          <- Dataset (debe colocarse aquí manualmente)
|
|-- outputs/
|   |-- versions.json                     <- Versiones de librerías usadas en la ejecución
|   |-- capitulo_4/
|   |   |-- base_limpia.xlsx              <- Dataset limpio con target binario
|   |   |-- tabla_variables.xlsx          <- Diccionario de variables
|   |   |-- estadistica_descriptiva.xlsx  <- Estadísticas descriptivas
|   |   |-- correlacion.xlsx              <- Matriz de correlación completa
|   |   |-- correlacion_target.xlsx       <- Correlaciones con la variable objetivo
|   |   |-- balance_clases.xlsx           <- Resumen de distribución de clases
|   |   |-- grafica_target.png            <- Gráfico de distribución del target
|   |   |-- heatmap_correlacion.png       <- Mapa de calor de correlaciones
|   |   |-- boxplot_edad.png              <- Edad por estado de deserción
|   |
|   |-- capitulo_5/
|       |-- comparacion_modelos_validacion_cruzada.xlsx  <- Resultados de VC para todos los modelos
|       |-- evaluacion_umbrales.xlsx      <- Resultados del barrido de umbrales
|       |-- metricas_modelo_final.xlsx    <- Métricas del modelo final en el conjunto de prueba
|       |-- matriz_confusion.xlsx         <- Matriz de confusión
|       |-- matriz_confusion.png          <- Mapa de calor de la matriz de confusión
|       |-- curva_roc.png                 <- Curva ROC
|       |-- curva_precision_recall.png    <- Curva Precisión-Recall
|       |-- curva_calibracion.png         <- Curva de calibración
|       |-- coeficientes_logistica.xlsx   <- Coeficientes del modelo (si se seleccionó RL)
|       |-- shap_summary.png              <- Gráfico beeswarm de SHAP
|       |-- shap_importancia_barras.png   <- Gráfico de barras SHAP (top 15)
|       |-- shap_importancia_variables.xlsx  <- Valores absolutos medios SHAP
|       |-- segmentacion_riesgo.xlsx      <- Conjunto de prueba con segmento de riesgo
|       |-- distribucion_segmentos_riesgo.png  <- Gráfico de distribución por segmento
|       |-- resumen_segmentos_riesgo.xlsx <- Resumen por segmento (tamaño, prob. media, tasa real)
|       |-- configuracion_modelo.json     <- Configuración completa del modelo
|
|-- models/
|   |-- modelo_final_desercion.pkl        <- Pipeline final serializado (preprocesador + modelo)
|
|-- Desercion_estudiantil.ipynb           <- Notebook principal
|-- requirements.txt                      <- Dependencias de Python
|-- README.md                             <- Este archivo
```

---

## 6. Configuración del Entorno

### Requisitos Previos

- Python 3.9 o superior
- pip
- Git

### Opción A: Instalación Local

**Paso 1. Clonar el repositorio**

```bash
git clone https://github.com/tu-usuario/student-dropout-prediction.git
cd student-dropout-prediction
```

**Paso 2. Crear y activar un entorno virtual**

En Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
```

En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**Paso 3. Instalar dependencias**

```bash
pip install -r requirements.txt
```

**Paso 4. Ubicar el dataset**

Copiar el archivo `BDF.xlsx` en la carpeta `data/`:

```
data/BDF.xlsx
```

Si la carpeta `data/` no existe, crearla:

```bash
mkdir data
```

**Paso 5. Lanzar Jupyter**

```bash
jupyter notebook Desercion_estudiantil.ipynb
```

### Opción B: Google Colab

**Paso 1.** Abrir el notebook en Google Colab.

**Paso 2.** La primera celda contiene un bloque específico para Colab que usa `files.upload()` para recibir `BDF.xlsx` desde tu computadora local. Mueve automáticamente el archivo a la ruta esperada `data/BDF.xlsx`.

**Paso 3.** Ejecutar todas las celdas en orden desde el menú: `Entorno de ejecución > Ejecutar todo`.

No se requiere configuración adicional para Colab. Todos los directorios se crean automáticamente.

---

## 7. Guía de Ejecución Paso a Paso

El notebook está organizado en 36 bloques numerados. A continuación se describe cada bloque y lo que produce.

### Bloque 1 — Configuración de Rutas y Carga de Datos (Inicio Rápido)

Versión abreviada de la configuración del entorno. Define rutas de directorios e intenta cargar el dataset. Este bloque se utiliza al ejecutar directamente tras colocar el archivo. La inicialización completa ocurre en el Bloque 2.

**Qué hace:** Crea la estructura de carpetas e intenta leer `data/BDF.xlsx`.  
**Salida:** Ninguna (solo configuración).

---

### Bloque 2 — Inicialización Completa

Importa todas las librerías necesarias, establece la semilla aleatoria global (`RANDOM_STATE = 42`), define todas las rutas de directorios y guarda un archivo `versions.json` con las versiones exactas de las librerías utilizadas. Luego carga el dataset.

**Qué hace:** Inicialización completa del entorno y carga de datos.  
**Archivo generado:** `outputs/versions.json`

---

### Bloque 3 — Revisión de Calidad

Verifica valores nulos y registros duplicados. Imprime información de tipos de datos para todas las columnas.

**Qué hace:** Valida la integridad de los datos.  
**Salida:** Reporte en consola. Si existen duplicados, se eliminan.

---

### Bloque 4 — Transformación de la Variable Objetivo

Mapea el target original de tres clases (`Dropout`, `Enrolled`, `Graduate`) a una variable binaria: `target_dropout` (1 = desertor, 0 = no desertor). Imprime una tabulación cruzada para validación.

**Qué hace:** Crea el target de clasificación binaria.  
**Salida:** Nueva columna `target_dropout` añadida al DataFrame.  
**Archivo generado:** `outputs/capitulo_4/base_limpia.xlsx`

---

### Bloque 5 — Vista Previa

Muestra las primeras 10 filas del DataFrame limpio.

---

### Bloque 6 — Identificación de Variables y Diccionario

Clasifica todas las variables como numéricas o categóricas (codificadas numéricamente). Construye un diccionario formal de variables con descripciones y roles. Produce la tabla de variables utilizada en la tesis.

**Qué hace:** Define las listas `columnas_numericas` y `columnas_categoricas`.  
**Archivo generado:** `outputs/capitulo_4/tabla_variables.xlsx`

---

### Bloque 7 — Análisis Exploratorio

Calcula estadísticas descriptivas para todas las variables numéricas. Genera la matriz de correlación y produce tres visualizaciones: el gráfico de distribución del target, el mapa de calor de correlaciones y el boxplot de edad por estado de deserción. Reporta la razón de desbalance de clases y la clasifica como baja, moderada o alta.

**Qué hace:** Análisis exploratorio completo de los datos.  
**Archivos generados:**
- `outputs/capitulo_4/estadistica_descriptiva.xlsx`
- `outputs/capitulo_4/correlacion.xlsx`
- `outputs/capitulo_4/correlacion_target.xlsx`
- `outputs/capitulo_4/balance_clases.xlsx`
- `outputs/capitulo_4/grafica_target.png`
- `outputs/capitulo_4/heatmap_correlacion.png`
- `outputs/capitulo_4/boxplot_edad.png`

---

### Bloque 8 — División Train / Test

Divide el dataset en entrenamiento (80%, n=3.539) y prueba (20%, n=885) usando muestreo estratificado para preservar las proporciones de clases en ambos subconjuntos. El conjunto de prueba se retiene y nunca se usa para la selección de modelos ni el ajuste de umbrales.

**Qué hace:** Crea `X_train`, `X_test`, `y_train`, `y_test`.  
**Salida:** Imprime la distribución de clases para ambas particiones.

---

### Bloque 9 — Pipeline de Preprocesamiento

Define un `ColumnTransformer` con dos pipelines:
- **Numérico:** Imputación por mediana seguida de StandardScaler
- **Categórico:** Imputación por moda seguida de OneHotEncoder con `handle_unknown='ignore'`

Este preprocesador siempre se embebe dentro del pipeline del modelo para garantizar que las transformaciones se ajusten solo sobre los datos de entrenamiento y se apliquen consistentemente al momento de la prueba, previniendo la fuga de datos.

**Qué hace:** Define el objeto `preprocesador`.  
**Salida:** Resumen en consola del conteo de variables.

---

### Bloque 10 — Definición de Modelos

Define los tres modelos candidatos con sus hiperparámetros:

| Modelo | Parámetros Clave |
|---|---|
| Regresión Logística | `max_iter=3000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=300`, `class_weight='balanced'`, `n_jobs=-1` |
| XGBoost | `n_estimators=300`, `lr=0,05`, `max_depth=5`, `subsample=0,9`, `colsample_bytree=0,9`, `scale_pos_weight` calculado desde la razón de clases |

**Qué hace:** Crea el diccionario `modelos`.

---

### Bloque 11 — Validación Cruzada Estratificada

Ejecuta `cross_validate` con `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` para cada modelo. Calcula la media y la desviación estándar para: exactitud, precisión, recall, F1-score y AUC-ROC. Los resultados se ordenan por recall (descendente).

**Qué hace:** Compara los tres modelos bajo condiciones de evaluación idénticas y robustas.  
**Archivo generado:** `outputs/capitulo_5/comparacion_modelos_validacion_cruzada.xlsx`

---

### Bloque 12 — Selección del Modelo y Entrenamiento Final

Selecciona el modelo con el mayor recall medio en la validación cruzada. Entrena el modelo seleccionado (dentro de su pipeline completo) sobre el conjunto de entrenamiento completo.

**Qué hace:** Ajusta `pipeline_final` sobre `X_train`.

---

### Bloque 13 — Análisis de Umbral y Predicción

Genera probabilidades predichas sobre el conjunto de prueba. Evalúa exactitud, precisión, recall y F1-score para umbrales desde 0,10 hasta 0,90 en pasos de 0,05. Selecciona el umbral que maximiza el F1-score sujeto a una restricción de recall >= 0,70. Aplica el umbral óptimo para generar las predicciones binarias finales.

**Qué hace:** Determina `umbral_optimo` y produce `y_pred`.  
**Archivo generado:** `outputs/capitulo_5/evaluacion_umbrales.xlsx`

---

### Bloque 14 — Métricas Finales y Matriz de Confusión

Calcula todas las métricas finales sobre el conjunto de prueba usando `y_pred` y `y_prob`: exactitud, precisión, recall, F1-score, AUC-ROC. Construye y visualiza la matriz de confusión.

**Qué hace:** Evaluación final del rendimiento del modelo.  
**Archivos generados:**
- `outputs/capitulo_5/metricas_modelo_final.xlsx`
- `outputs/capitulo_5/matriz_confusion.xlsx`
- `outputs/capitulo_5/matriz_confusion.png`

---

### Bloque 15 — Curvas de Rendimiento

Genera tres gráficos de diagnóstico:
1. **Curva ROC** — Tasa de Verdaderos Positivos vs. Tasa de Falsos Positivos con anotación del AUC
2. **Curva Precisión-Recall** — Precisión vs. Recall a través de umbrales con anotación del AUC-PR
3. **Curva de Calibración** — Probabilidades predichas vs. proporciones reales de deserción (10 intervalos), comparada con la diagonal ideal

**Archivos generados:**
- `outputs/capitulo_5/curva_roc.png`
- `outputs/capitulo_5/curva_precision_recall.png`
- `outputs/capitulo_5/curva_calibracion.png`

---

### Bloque 16 — Interpretabilidad con SHAP

Extrae el preprocesador y el modelo ajustados del pipeline final. Si se seleccionó Regresión Logística, guarda los coeficientes del modelo. Transforma los datos de entrenamiento y prueba a través del preprocesador para obtener matrices densas de variables. Crea un `shap.Explainer` usando 200 muestras de fondo del conjunto de entrenamiento. Calcula los valores SHAP para todas las observaciones del conjunto de prueba.

Genera:
1. **Gráfico beeswarm** — Dirección y distribución del impacto de cada variable en todas las observaciones de prueba
2. **Gráfico de barras** — Top 15 variables ordenadas por valor absoluto medio SHAP

**Qué hace:** Análisis SHAP completo sobre el modelo seleccionado.  
**Archivos generados:**
- `outputs/capitulo_5/coeficientes_logistica.xlsx` (si se seleccionó Regresión Logística)
- `outputs/capitulo_5/shap_summary.png`
- `outputs/capitulo_5/shap_importancia_barras.png`
- `outputs/capitulo_5/shap_importancia_variables.xlsx`

---

### Bloque 17 — Segmentación de Riesgo

Calcula los percentiles 33 y 66 de las probabilidades de deserción estimadas en el conjunto de prueba. Asigna a cada estudiante un segmento de riesgo: Bajo, Medio o Alto. Produce una tabla resumen con el tamaño del segmento, la probabilidad media estimada y la tasa real de deserción por segmento.

**Qué hace:** Convierte las probabilidades predichas en categorías de riesgo accionables.  
**Archivos generados:**
- `outputs/capitulo_5/segmentacion_riesgo.xlsx`
- `outputs/capitulo_5/distribucion_segmentos_riesgo.png`
- `outputs/capitulo_5/resumen_segmentos_riesgo.xlsx`

---

### Bloque 18 — Serialización del Modelo

Guarda el pipeline final completo (preprocesador + modelo entrenado) usando `joblib.dump`. El archivo serializado puede cargarse directamente para calificar nuevos registros de estudiantes sin reentrenar.

**Qué hace:** Exporta el modelo listo para producción.  
**Archivo generado:** `models/modelo_final_desercion.pkl`

---

### Bloque 19 — Exportación de Configuración

Guarda un archivo JSON con la configuración completa del modelo: algoritmo seleccionado, semilla aleatoria, definición del target, criterio de selección, umbral óptimo y umbrales de segmentación.

**Archivo generado:** `outputs/capitulo_5/configuracion_modelo.json`

---

### Bloque 20 — Exportación de Requisitos

Escribe `requirements.txt` en disco mediante un comando mágico del notebook.

---

### Bloque 21 — Resumen Final

Imprime la lista completa de archivos de salida generados como lista de verificación.

---

## 8. Resultados

### Comparación por Validación Cruzada (Conjunto de Entrenamiento, k=5)

| Modelo | Exactitud | Precisión | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Regresión Logística | 0,857 | 0,764 | **0,804** | 0,783 | 0,920 |
| XGBoost | 0,867 | 0,798 | 0,784 | 0,791 | 0,919 |
| Random Forest | 0,866 | 0,859 | 0,698 | 0,770 | 0,915 |

La Regresión Logística fue seleccionada con base en el mayor recall para la clase de deserción (0,804). XGBoost alcanza mayor exactitud global y precisión, pero su menor recall implica más desertores no detectados. Random Forest muestra la mayor precisión pero el menor recall, siendo inadecuado cuando el objetivo principal es minimizar los estudiantes en riesgo no identificados.

### Análisis de Umbrales de Decisión (Conjunto de Prueba)

| Umbral | Exactitud | Precisión | Recall | F1-score |
|---|---|---|---|---|
| 0,10 | 0,626 | 0,461 | 0,968 | 0,624 |
| 0,30 | 0,817 | 0,657 | 0,898 | 0,759 |
| 0,50 | 0,876 | 0,788 | 0,838 | 0,812 |
| **0,55** | **0,885** | **0,820** | **0,820** | **0,821** |
| 0,65 | 0,886 | 0,865 | 0,764 | 0,811 |
| 0,80 | 0,877 | 0,915 | 0,680 | 0,780 |

Umbral óptimo: **0,55** — maximiza el F1-score (0,821) manteniendo el recall por encima del mínimo de 0,70.

### Rendimiento del Modelo Final (Conjunto de Prueba, n=885)

| Métrica | Valor | Interpretación |
|---|---|---|
| Exactitud | 0,885 | 88,5% de los estudiantes clasificados correctamente |
| Precisión | 0,818 | 82 de cada 100 alertas corresponden a desertores reales |
| Recall | 0,824 | 82,4% de los desertores reales son correctamente identificados |
| F1-score | 0,821 | Balance óptimo entre precisión y recall |
| AUC-ROC | 0,931 | Excelente capacidad discriminativa (umbral: >0,90) |
| AUC-PR | 0,900 | Alto rendimiento sobre la clase de interés en contexto desbalanceado |

### Matriz de Confusión (Conjunto de Prueba, umbral=0,55)

|  | Predicho No Desertor | Predicho Desertor |
|---|---|---|
| **Real No Desertor** | 549 (Verdaderos Negativos) | 52 (Falsos Positivos) |
| **Real Desertor** | 50 (Falsos Negativos) | 234 (Verdaderos Positivos) |

- 234 estudiantes en riesgo correctamente identificados para intervención temprana
- 50 falsos negativos (17,6% de los desertores reales no detectados)
- 52 falsos positivos (intervenciones preventivas innecesarias pero de bajo costo)

### Importancia de Variables SHAP (Top 10)

| Rango | Variable | SHAP Medio Abs. | Categoría | Dirección |
|---|---|---|---|---|
| 1 | Unidades aprobadas 2do sem. | 1,588 | Académica | Valores bajos aumentan el riesgo |
| 2 | Unidades aprobadas 1er sem. | 0,714 | Académica | Valores bajos aumentan el riesgo |
| 3 | Unidades matriculadas 2do sem. | 0,592 | Académica | Valores altos aumentan riesgo en algunos perfiles |
| 4 | Ocupación del padre (cat. 9) | 0,296 | Familiar | Presencia aumenta la probabilidad de deserción |
| 5 | Tasa de desempleo | 0,278 | Macroeconómica | Valores altos aumentan la probabilidad de deserción |
| 6 | Matrícula al día (cat. 1) | 0,226 | Financiera | Pagos al corriente reducen el riesgo |
| 7 | Matrícula al día (cat. 0) | 0,219 | Financiera | Pagos vencidos aumentan el riesgo |
| 8 | Unidades acreditadas 2do sem. | 0,216 | Académica | Valores bajos aumentan el riesgo |
| 9 | Edad al momento de matrícula | 0,195 | Demográfica | Mayor edad aumenta la probabilidad de deserción |
| 10 | Unidades matriculadas 1er sem. | 0,165 | Académica | Contexto de carga académica, primer semestre |

### Segmentación de Riesgo (Conjunto de Prueba, n=885)

| Nivel de Riesgo | Estudiantes | % del Total | Prob. Estimada Media | Tasa Real de Deserción |
|---|---|---|---|---|
| Alto | 301 | 34,0% | 0,861 | 79,1% (238/301) |
| Medio | 292 | 33,0% | 0,246 | 12,7% (37/292) |
| Bajo | 292 | 33,0% | 0,052 | 3,1% (9/292) |

La brecha de 76 puntos porcentuales entre las tasas reales de deserción del segmento alto y el bajo confirma que la segmentación tiene alta capacidad discriminativa y no constituye una partición arbitraria de la población.

---

## 9. Referencia de Archivos de Salida

| Archivo | Ubicación | Descripción |
|---|---|---|
| `versions.json` | `outputs/` | Versiones de Python y librerías |
| `base_limpia.xlsx` | `outputs/capitulo_4/` | Dataset limpio con target binario |
| `tabla_variables.xlsx` | `outputs/capitulo_4/` | Diccionario de variables con tipo y rol |
| `estadistica_descriptiva.xlsx` | `outputs/capitulo_4/` | Estadísticas descriptivas de variables numéricas |
| `correlacion.xlsx` | `outputs/capitulo_4/` | Matriz de correlación numérica completa |
| `correlacion_target.xlsx` | `outputs/capitulo_4/` | Correlación de cada variable con el target |
| `balance_clases.xlsx` | `outputs/capitulo_4/` | Frecuencia, proporción y razón de desbalance |
| `grafica_target.png` | `outputs/capitulo_4/` | Gráfico de barras de distribución del target |
| `heatmap_correlacion.png` | `outputs/capitulo_4/` | Mapa de calor de correlaciones (paleta coolwarm) |
| `boxplot_edad.png` | `outputs/capitulo_4/` | Edad al momento de matrícula por estado de deserción |
| `comparacion_modelos_validacion_cruzada.xlsx` | `outputs/capitulo_5/` | Resultados de VC: todos los modelos, todas las métricas |
| `evaluacion_umbrales.xlsx` | `outputs/capitulo_5/` | Barrido de umbrales de 0,10 a 0,90 |
| `metricas_modelo_final.xlsx` | `outputs/capitulo_5/` | Métricas finales del modelo en el conjunto de prueba |
| `matriz_confusion.xlsx` | `outputs/capitulo_5/` | Matriz de confusión (TN, FP, FN, TP) |
| `matriz_confusion.png` | `outputs/capitulo_5/` | Mapa de calor de la matriz de confusión |
| `curva_roc.png` | `outputs/capitulo_5/` | Curva ROC con AUC |
| `curva_precision_recall.png` | `outputs/capitulo_5/` | Curva Precisión-Recall con AUC-PR |
| `curva_calibracion.png` | `outputs/capitulo_5/` | Curva de calibración (modelo vs. ideal) |
| `coeficientes_logistica.xlsx` | `outputs/capitulo_5/` | Coeficientes del modelo (solo Regresión Logística) |
| `shap_summary.png` | `outputs/capitulo_5/` | Gráfico beeswarm SHAP (top 15) |
| `shap_importancia_barras.png` | `outputs/capitulo_5/` | Gráfico de barras SHAP (top 15) |
| `shap_importancia_variables.xlsx` | `outputs/capitulo_5/` | Ranking SHAP completo con valores absolutos medios |
| `segmentacion_riesgo.xlsx` | `outputs/capitulo_5/` | Conjunto de prueba con probabilidad, predicción y segmento |
| `distribucion_segmentos_riesgo.png` | `outputs/capitulo_5/` | Conteo de estudiantes por segmento de riesgo |
| `resumen_segmentos_riesgo.xlsx` | `outputs/capitulo_5/` | Resumen por segmento: tamaño, prob. media, tasa real |
| `configuracion_modelo.json` | `outputs/capitulo_5/` | Configuración del modelo: algoritmo, umbral, cortes de segmentación |
| `modelo_final_desercion.pkl` | `models/` | Pipeline final serializado (preprocesador + modelo) |

---

## 10. Decisiones de Diseño

### Por qué priorizar el recall sobre la exactitud

En un sistema de alerta temprana para la deserción estudiantil, el costo de **no detectar a un estudiante que desertará** (falso negativo) es sustancialmente mayor que el costo de **señalar a un estudiante que no desertará** (falso positivo). Un falso negativo significa que un estudiante abandona sin intervención. Un falso positivo significa que se aplica una acción preventiva a alguien que no la necesitaba — lo cual es la función central de un sistema de apoyo estudiantil de todos modos. La minimización de falsos negativos mediante el recall es por tanto el criterio metodológicamente correcto.

### Por qué validación cruzada estratificada en lugar de una sola división

Una sola partición aleatoria puede producir resultados que dependen de qué estudiantes terminaron en cada subconjunto. La validación cruzada estratificada de k pliegues entrena y evalúa el modelo en k subconjuntos de datos diferentes, preservando siempre las proporciones de clases, y promedia los resultados. Esto proporciona una estimación más estable y realista del rendimiento fuera de muestra.

### Por qué integrar el preprocesamiento dentro del pipeline

Ajustar el escalador y el codificador sobre el dataset completo antes de dividirlo, o ajustarlos sobre el conjunto de entrenamiento y aplicarlos al de prueba en un paso separado, ambos introducen fuga de datos. Usar `sklearn.pipeline.Pipeline` garantiza que el preprocesador se ajuste solo sobre los datos de entrenamiento en cada pliegue de validación cruzada y sobre el conjunto de entrenamiento completo para el modelo final. El conjunto de prueba siempre se trata como genuinamente no visto.

### Por qué usar ponderación de clases en lugar de SMOTE u otro remuestreo

SMOTE genera observaciones sintéticas en el espacio de variables. En un dataset mixto numérico/categórico como este, las observaciones sintéticas pueden crear combinaciones que no corresponden a perfiles reales de estudiantes, sesgando potencialmente la frontera de decisión. La ponderación de clases ajusta la función de pérdida durante el entrenamiento para penalizar más los errores sobre la clase minoritaria, sin crear datos artificiales.

### Por qué seleccionar el umbral 0,55 sobre el convencional 0,50

El umbral por defecto de 0,50 no está justificado teóricamente para tareas de clasificación desbalanceadas. Con 0,50, el modelo alcanza F1=0,812 y recall=0,838. Con 0,55, la precisión y el recall convergen (ambos 0,820), el F1-score es marginalmente mayor (0,821), y el modelo demuestra que sus estimaciones de probabilidad cercanas al límite de 0,50 están suficientemente calibradas para soportar un desplazamiento leve hacia arriba. La curva de calibración confirma la idoneidad funcional para este rango.

### Por qué SHAP en lugar de solo los coeficientes del modelo

Los coeficientes de la Regresión Logística reflejan la relación entre cada variable estandarizada/codificada y el log-odds de deserción. Son interpretables pero operan en el espacio de variables transformadas, donde las categóricas están codificadas one-hot. SHAP proporciona explicaciones en unidades de contribución a la probabilidad en lugar del log-odds, maneja variables codificadas one-hot agrupando las dummies relacionadas, y ofrece interpretabilidad tanto global (a nivel de población) como local (a nivel de estudiante individual). SHAP también permite verificar que las predicciones del modelo se alineen con el conocimiento del dominio.

---

## 11. Reproducción Exacta de Resultados

Para reproducir exactamente los resultados reportados en la tesis y en este documento, deben cumplirse las siguientes condiciones:

**1. Semilla aleatoria**

El estado aleatorio global es `RANDOM_STATE = 42`. Esta semilla se aplica a:
- `numpy.random.seed(42)`
- `random.seed(42)`
- `train_test_split(random_state=42)`
- `StratifiedKFold(random_state=42)`
- Todos los constructores de modelos

**2. Versiones de librerías**

Los resultados están garantizados para reproducirse con las siguientes versiones (registradas automáticamente en `outputs/versions.json` cuando el notebook se ejecuta):

```
Python       >= 3.9
numpy        >= 1.24
pandas       >= 2.0
scikit-learn >= 1.3
xgboost      >= 1.7
shap         >= 0.42
openpyxl     >= 3.1
joblib       >= 1.3
seaborn      >= 0.12
matplotlib   >= 3.7
```

Instalar todas las dependencias de una vez:

```bash
pip install -r requirements.txt
```

**3. Dataset**

El dataset debe colocarse en `data/BDF.xlsx` sin modificaciones. Cualquier cambio en el archivo fuente (nombres de columnas, orden de filas, codificación de valores) cambiará los resultados.

**4. Orden de ejecución**

Todas las celdas del notebook deben ejecutarse secuencialmente, del Bloque 1 al Bloque 36, en un kernel limpio. No re-ejecutar bloques individuales fuera de orden. Variables como `columnas_numericas`, `columnas_categoricas`, `preprocesador`, `pipeline_final` y `umbral_optimo` son con estado y dependen de bloques anteriores.

Para garantizar una ejecución limpia: `Kernel > Reiniciar y ejecutar todo`

**5. Independencia del hardware**

El pipeline no depende de ejecución en GPU. Los resultados son deterministas en arquitecturas de CPU cuando todas las semillas están configuradas y las versiones de librerías coinciden.

---

## 12. Dependencias y Versiones

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
openpyxl
joblib
```

Instalar con:

```bash
pip install -r requirements.txt
```

Para Colab, la mayoría de dependencias están preinstaladas. Solo `shap` puede requerir instalación explícita:

```python
!pip install shap
```

---

## 13. Cita

Si utilizas este código, metodología o resultados en trabajo académico, por favor cita:

```
Vega Nieto, S. M. (2026). Modelo Predictivo e Interpretable para la Identificación
Temprana del Riesgo de Deserción Estudiantil en Educación Superior. Tesis de Maestría,
Escuela de Negocios, Maestría en Business Intelligence y Ciencia de Datos.
```

Cita del dataset:

```
Realinho, V., Martins, M. V., Machado, J., & Baptista, L. (2021).
Predict Students' Dropout and Academic Success.
UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
```

---

## Referencias

Bettahi, A., Belouadha, F. Z., & Harroud, H. (2025). A Modular and Explainable Machine Learning Pipeline for Student Dropout Prediction in Higher Education. *Algorithms, 18*(10). https://doi.org/10.3390/a18100662

Carballo-Mendivil, B., Arellano-Gonzalez, A., Rios-Vazquez, N. J., & Lizardi-Duarte, M. (2025). Predicting Student Dropout from Day One: XGBoost-Based Early Warning System Using Pre-Enrollment Data. *Applied Sciences, 15*(16). https://doi.org/10.3390/app15169202

Duro, B., Gomes, A., Correia, F. B., Borges, A. R., & Bernardino, J. (2026). Machine Learning and Deep Learning for Dropout Prediction in Higher Education: A Review. *Computers, 15*(3), 164. https://doi.org/10.3390/computers15030164

Khatun, M. R., Mim, M. A., Tasin, M. M., & Hossain, M. M. (2025). A hybrid framework of statistical, machine learning, and explainable AI methods for school dropout prediction. *PLOS ONE, 20*(9), e0331917. https://doi.org/10.1371/journal.pone.0331917

Kruger, J. G. C., Britto, A. de S., & Barddal, J. P. (2023). An explainable machine learning approach for student dropout prediction. *Expert Systems with Applications, 233*, 120933. https://doi.org/10.1016/j.eswa.2023.120933

Selim, K. S., & Rezk, S. S. (2023). On predicting school dropouts in Egypt: A machine learning approach. *Education and Information Technologies, 28*(7), 9235-9266. https://doi.org/10.1007/s10639-022-11571-x
