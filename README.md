<!-- Cabecera animada tipo capsule-render -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:f12711,50:f6d365,100:12c2e9&height=180&section=header&text=Business%20Analytics%20para%20Startups%20🚀&fontSize=50&fontAlign=50&fontColor=fff&desc=Innova,%20emprende,%20escala&descAlign=50&descSize=28&descAlignY=70" alt="Business Analytics para Startups Banner"/>
</p>

<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Google%20Colab-ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/Licencia-GPL--3.0-1A6600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Estado-Activo-00BFFF?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LATAM-Contexto%20Regional-2C3E50?style=for-the-badge"/>

<br/><br/>

### Herramientas analíticas en Python para tomar decisiones basadas en datos, preparar valuaciones tecnológicas y escalar negocios en América Latina.

**Módulos completos · Código comentado · Contexto LATAM · Google Colab ready**

</div>

[![----------------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#tabla-de-contenidos)

## ➤ Tabla de contenidos

- [¿Qué es este repositorio?](#-qué-es-este-repositorio)
- [¿Para quién está pensado?](#-para-quién-está-pensado)
- [Módulos disponibles](#-módulos-disponibles)
  - [M1 · Análisis Exploratorio de Datos (EDA)](#m1--análisis-exploratorio-de-datos-eda)
  - [M2 · Analítica Predictiva con XGBoost](#m2--analítica-predictiva-con-xgboost)
  - [M3 · Analítica de Marketing — Modelo LTV/CAC](#m3--analítica-de-marketing--modelo-ltvcac)
  - [M4 · Segmentación de Clientes con KMeans](#m4--segmentación-de-clientes-con-kmeans)
  - [M5 · Demand Forecasting con Prophet](#m5--demand-forecasting-con-prophet)
  - [M6 · Valuación de Tecnología — Método VC](#m6--valuación-de-tecnología--método-vc)
- [Integración con Claude (Skill + Agente)](#-integración-con-claude-skill--agente)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Cómo empezar](#-cómo-empezar)
- [Convenciones de código](#-convenciones-de-código)
- [Benchmarks de referencia LATAM](#-benchmarks-de-referencia-latam)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#qué-es-este-repositorio)

## ➤ ¿Qué es este repositorio?

Este repositorio contiene una colección de módulos de **business analytics** diseñados para startups, investigadores, gestores tecnológicos y tomadores de decisión en América Latina. Cada módulo es un notebook de Python completamente ejecutable en Google Colab, con datos simulados en contexto LATAM, comentarios explicativos en español y visualizaciones listas para presentar.

El objetivo no es solo mostrar cómo correr un modelo: es explicar **qué pregunta de negocio responde cada herramienta** y cómo interpretar los resultados para tomar decisiones concretas, negociar con inversores, preparar rondas de financiamiento o estructurar procesos de transferencia tecnológica.

> **Nota sobre transferencia tecnológica:** los módulos M3, M5 y M6 están especialmente orientados al **empaquetamiento y valuación de tecnologías universitarias**, cubriendo las métricas que oficinas de transferencia tecnológica (OTT/TTO) y fondos de capital de riesgo solicitan en procesos de licenciamiento, spinoff y rondas de inversión.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#para-quién)

## ➤ ¿Para quién está pensado?

| Perfil | Uso recomendado |
|---|---|
| 🚀 **Founders y emprendedores** | Validar unit economics, preparar pitch con datos sólidos |
| 🏛️ **Gestores tecnológicos (OTT/TTO)** | Valuar tecnologías universitarias para licenciamiento o spinoff |
| 📊 **Analistas de datos** | Plantillas de código comentado para proyectos reales |
| 🎓 **Investigadores y docentes** | Material para cursos de innovación, emprendimiento y analítica |
| 💼 **Consultores** | Adaptar modelos a contextos específicos de clientes en LATAM |
| 🏦 **Inversores ángel y fondos** | Verificar métricas y supuestos de los proyectos evaluados |

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#módulos)

## ➤ Módulos disponibles

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#m1)

### M1 · Análisis Exploratorio de Datos (EDA)

> 📂 `Code Colabs/Modulo_1_EDA_LATAM.ipynb`

**Pregunta que responde:** ¿Qué tengo, qué calidad tienen mis datos y qué patrones iniciales puedo identificar?

El punto de entrada obligatorio antes de cualquier modelado. Cubre ocho pasos progresivos: carga del dataset, análisis descriptivo, detección y tratamiento de valores faltantes, identificación de outliers por método IQR, análisis de correlación, métricas de overview por país y etapa, y una comparativa inicial de modelos benchmark (regresión lineal vs. random forest).

**Caso de uso simulado:** portafolio de proyectos de biotecnología en LATAM — inversión en I+D, tiempos de desarrollo, tasas de éxito y valor potencial de mercado.

**Librerías:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy` · `sklearn`

**Outputs generados:**
- `eda_descriptivo_latam.png` — distribuciones y boxplots por país
- `eda_missing_values.png` — mapa de calor de valores faltantes
- `eda_outliers.png` — boxplots por variable numérica
- `eda_correlacion.png` — matriz de correlación
- `eda_model_selection.png` — comparativa de modelos benchmark (R²)

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#m2)

### M2 · Analítica Predictiva con XGBoost

> 📂 `Code Colabs/Modulo_2_XGBoost_LATAM.ipynb`

**Pregunta que responde:** ¿Qué clientes se irán? ¿Qué proyectos tienen mayor probabilidad de recibir funding? ¿Qué leads van a convertir?

Implementa un clasificador XGBoost con manejo de datos desbalanceados, validación cruzada, curva ROC e interpretabilidad mediante SHAP values. El caso de uso simula la predicción de funding exitoso para startups LATAM, con variables de tracción, equipo, mercado y etapa de desarrollo.

**Librerías:** `xgboost` · `shap` · `sklearn` · `pandas` · `matplotlib`

> ⚠️ Requiere instalación adicional: `!pip install xgboost shap`

**Outputs generados:**
- `xgboost_funding_latam.png` — curva ROC + importancia de features (SHAP)

**Benchmarks LATAM incluidos:**
- AUC > 0.75 como umbral de utilidad para priorización de deal-flow en fondos
- Comparativa por sector: FinTech lidera probabilidad de funding en la región

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#m3)

### M3 · Analítica de Marketing — Modelo LTV/CAC

> 📂 `Code Colabs/Modulo_3_LTV_CAC_LATAM.ipynb`

**Pregunta que responde:** ¿Cuánto vale cada cliente? ¿Qué canal de adquisición es rentable? ¿Mis unit economics son suficientes para levantar capital?

Calcula LTV ajustado por margen, CAC por canal, ratio LTV/CAC, payback period y curvas de retención. El caso de uso simula una FinTech LATAM con cinco canales de adquisición: referidos, social ads, influencers, outbound B2B y App Store (ASO).

**Fórmula central:**

```
LTV = (ARPU mensual × Margen bruto) / Churn mensual
Ratio LTV/CAC → umbral saludable para inversores: > 3x
Payback period → criterio Serie A en LATAM: < 12 meses
```

**Outputs generados:**
- `ltv_cac_latam.png` — ratio LTV/CAC · curvas de retención · scatter CAC vs LTV · payback por canal

**Aplicación en transferencia tecnológica:** el modelo LTV/CAC se adapta directamente para estimar el valor de un contrato de licenciamiento: ARPU = regalía mensual esperada, churn = probabilidad de no renovación, CAC = costo del proceso de negociación y due diligence.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#m4)

### M4 · Segmentación de Clientes con KMeans

> 📂 `Code Colabs/Modulo_4_KMeans_RFM_LATAM.ipynb`

**Pregunta que responde:** ¿Cómo se agrupan mis clientes? ¿A quién debo retener, reactivar o dejar ir?

Implementa análisis RFM (Recencia, Frecuencia, Monetario) combinado con KMeans clustering. Incluye método del codo (elbow) y silhouette score para selección óptima de K, etiquetado automático de segmentos con nombres de negocio interpretables y estrategias de acción por segmento.

**Segmentos identificados:**
- 🏆 **Campeones** — alta frecuencia, compra reciente, alto ticket
- ⭐ **Leales** — frecuencia media-alta, estables
- 📈 **Potenciales** — buenos indicadores pero baja frecuencia aún
- 😴 **Hibernando** — compraron bien antes, ahora inactivos
- ❌ **Perdidos** — baja recencia, baja frecuencia, bajo ticket

**Outputs generados:**
- `segmentacion_kmeans_latam.png` — elbow · silhouette · scatter recencia vs monetario · tamaño de clusters

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#m5)

### M5 · Demand Forecasting con Prophet

> 📂 `Code Colabs/Modulo_5_Prophet_LATAM.ipynb`

**Pregunta que responde:** ¿Cuánto venderé el próximo trimestre? ¿Cómo afectan las estacionalidades y eventos especiales a mi demanda?

Implementa Facebook Prophet con estacionalidades semanales, anuales y una componente quincenal específica de LATAM. Incorpora holidays regionales (Buen Fin, Black Friday, Cyber Lunes, vacaciones navideñas) y proyecta 90 días hacia adelante con intervalos de confianza al 90%.

> ⚠️ Requiere instalación adicional: `!pip install prophet`

**Estacionalidades LATAM capturadas:**
- Efecto quincena (días 1-3 y 14-17 de cada mes): +20-30% sobre la media
- Buen Fin México (tercera semana de noviembre): +100% sobre la línea base
- Caída B2B en fines de semana: factor 0.40 sobre días hábiles
- Vacaciones navideñas (24 dic – 8 ene): -40% en SaaS B2B

**Métrica de validación:** MAPE (Mean Absolute Percentage Error). Umbral recomendado para planning financiero: MAPE < 15%.

**Outputs generados:**
- `demand_forecast_latam.png` — serie histórica + forecast 90 días + componente de tendencia

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#m6)

### M6 · Valuación de Tecnología — Método VC

> 📂 `Code Colabs/Modulo_6_Valuacion_VC_LATAM.ipynb`

**Pregunta que responde:** ¿Cuánto vale mi empresa o tecnología? ¿Qué porcentaje de equity debo entregar en una ronda? ¿Mis supuestos son realistas para el mercado LATAM?

Implementa el **método VC (Venture Capital Method)** con tres escenarios ponderados (conservador, base, optimista), análisis de sensibilidad cruzado (ARR exit × múltiplo EV/ARR) y simulación Monte Carlo con 10.000 iteraciones para obtener la distribución completa de valuaciones posibles.

**Parámetros clave:**

```
IRR exigido por VCs LATAM:    25-35%
Múltiplo EV/ARR SaaS B2B:     5-10x (LATAM: 30-50% menor que Silicon Valley)
Horizonte típico de exit:     5 años
Pre-money = Exit Value / (1 + IRR)^N − Inversión buscada
```

**Outputs generados:**
- `valuacion_vc_latam.png` — distribución Monte Carlo · escenarios bar chart · heatmap de sensibilidad · dilución del founder

**Aplicación directa en transferencia tecnológica:**
Este módulo es el núcleo del empaquetamiento tecnológico para OTTs universitarias. Permite calcular el rango de pre-money de un spinoff, estimar el equity que debería tomar un fondo de innovación o un licenciatario estratégico, y presentar tres escenarios documentados en cualquier proceso de negociación de IP.

> ⚠️ Disclaimer: los resultados son ilustrativos. Validar siempre con transacciones comparables recientes en LATAM y consultar un asesor financiero especializado.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#claude)

## ➤ Integración con Claude (Skill + Agente)

Este repositorio incluye dos archivos para integrar los módulos como herramientas de inteligencia artificial en Claude Projects:

| Archivo | Propósito |
|---|---|
| [`SKILL.md`](./SKILL.md) | Define el motor técnico: plantillas de código, convenciones, contexto LATAM por módulo y estándares de output. Requiere YAML frontmatter para activarse en Claude Projects. |
| [`AGENT.md`](./AGENT.md) | Define el comportamiento del agente BAA (Business Analytics Agent): árbol de decisión para selección de módulo, personalización por sector, flujo de diagnóstico y criterios de calidad de respuesta. |

**Cómo activar:**
1. Crear un proyecto en Claude.ai
2. Subir `SKILL.md` y `AGENT.md` a Project Knowledge
3. El agente detectará automáticamente qué módulo usar según la consulta del usuario

**El agente BAA puede:**
- Diagnosticar qué módulo aplica según el problema de negocio descrito
- Adaptar el código al sector y país del usuario (FinTech, EdTech, SaaS B2B, etc.)
- Interpretar resultados en lenguaje de negocio, no solo estadístico
- Comparar métricas del usuario con benchmarks reales de LATAM

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#estructura)

## ➤ Estructura del repositorio

```
business-analytics-para-startups/
│
├── Code Colabs/
│   ├── Modulo_1_EDA_LATAM.ipynb
│   ├── Modulo_2_XGBoost_LATAM.ipynb
│   ├── Modulo_3_LTV_CAC_LATAM.ipynb
│   ├── Modulo_4_KMeans_RFM_LATAM.ipynb
│   ├── Modulo_5_Prophet_LATAM.ipynb
│   └── Modulo_6_Valuacion_VC_LATAM.ipynb
│
├── SKILL.md          ← Skill para Claude Projects (con YAML frontmatter)
├── AGENT.md          ← Agente BAA para Claude Projects
├── README.md         ← Este archivo
└── LICENSE
```

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#cómo-empezar)

## ➤ Cómo empezar

### Opción 1 — Google Colab (recomendada)

1. Abre cualquier notebook desde este repositorio haciendo clic en el badge de Colab que aparece al inicio de cada archivo `.ipynb`
2. Ejecuta la celda de instalación de dependencias adicionales al inicio del notebook
3. Corre las celdas en orden: cada sección está numerada y tiene comentarios explicativos

### Opción 2 — Entorno local

```bash
# Clonar el repositorio
git clone https://github.com/scientificbroker/business-analytics-para-startups.git
cd business-analytics-para-startups

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias base
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Instalar dependencias adicionales para módulos avanzados
pip install xgboost shap prophet
```

### Dependencias por módulo

| Módulo | Preinstalado en Colab | Requiere instalación adicional |
|---|---|---|
| M1 EDA | ✅ | — |
| M2 XGBoost | ❌ | `pip install xgboost shap` |
| M3 LTV/CAC | ✅ | — |
| M4 KMeans | ✅ | — |
| M5 Prophet | ❌ | `pip install prophet` |
| M6 VC Method | ✅ | — |

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#convenciones)

## ➤ Convenciones de código

Todo el código del repositorio sigue un conjunto de convenciones consistentes para facilitar la lectura, adaptación y uso en presentaciones:

```python
# 1. Semilla de reproducibilidad al inicio de cada notebook
np.random.seed(42)

# 2. Distribución de países LATAM en datos simulados
pais = np.random.choice(
    ['México', 'Brasil', 'Argentina', 'Colombia', 'Chile'],
    n, p=[0.25, 0.30, 0.20, 0.15, 0.10]
)

# 3. Headers de sección con separadores visuales
# ═══════════════════════════════════════════════════════════
# 📊 MÓDULO N: NOMBRE DEL ANÁLISIS — CONTEXTO LATAM
# ═══════════════════════════════════════════════════════════

# 4. Secciones internas numeradas
# ─── SECCIÓN 1: DESCRIPCIÓN ──────────────────────────────

# 5. Comentarios que explican el PORQUÉ, no solo el QUÉ
# XGBoost usa scale_pos_weight para compensar el desbalance
# entre clases (startups funded vs. no funded son minoría)

# 6. Print statements con contexto de negocio
print(f"🎯 LTV/CAC: {ratio:.1f}x  |  Benchmark LATAM: > 3x para Serie A")

# 7. Guardar todas las visualizaciones
plt.savefig('nombre_descriptivo.png', dpi=150, bbox_inches='tight')

# 8. Sección final de conclusiones accionables
print("✅ CONCLUSIÓN:")
print("   • Punto 1 con implicación para la startup")
```

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#benchmarks)

## ➤ Benchmarks de referencia LATAM

Los módulos incluyen estos benchmarks como referencia para interpretar resultados:

### Unit economics

| Métrica | Por debajo del mínimo | Saludable | Excelente |
|---|---|---|---|
| LTV/CAC ratio | < 1x | 3x | > 5x |
| Payback period | > 18 meses | 12 meses | < 6 meses |
| Churn mensual (B2C) | > 20% | 10-15% | < 8% |
| Churn mensual (B2B) | > 10% | 5-8% | < 3% |

### Valuación SaaS B2B (Serie A, LATAM 2023-2024)

| Sector | Múltiplo EV/ARR típico | Fuente referencial |
|---|---|---|
| FinTech B2B | 5-8x | Kaszek, ALLVP portfolio |
| EdTech | 4-7x | Latitud Fund Report |
| HealthTech | 5-9x | Softbank LATAM Fund II |
| SaaS B2B genérico | 5-10x | PitchBook LATAM |
| AgriTech | 3-6x | BID Lab portfolio |

### Forecasting

| MAPE | Interpretación |
|---|---|
| < 10% | Excelente — apto para compromisos contractuales |
| 10-15% | Bueno — apto para planning financiero interno |
| 15-25% | Aceptable — útil para estimaciones de rango |
| > 25% | Revisar modelo o calidad de datos |

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#contribuciones)

## ➤ Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz fork del repositorio
2. Crea una rama descriptiva: `git checkout -b feature/modulo-N-mejora`
3. Asegúrate de que el código sigue las convenciones del repositorio (ver sección anterior)
4. Incluye al menos dos visualizaciones y una sección de conclusiones accionables
5. Abre un Pull Request describiendo el cambio y los resultados obtenidos

### Ideas de contribución

- Adaptar módulos existentes a sectores específicos (AgriTech, GovTech, LegalTech)
- Agregar versiones en R de los módulos existentes
- Incorporar datos reales anonimizados de startups LATAM con su consentimiento
- Traducir comentarios del código al inglés para audiencias internacionales
- Agregar módulo de análisis de cohortes (M7 potencial)
- Agregar módulo de optimización de precios con modelos de elasticidad (M8 potencial)

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#licencia)

## ➤ Licencia

Este proyecto está bajo la licencia **GPL-3.0**. Puedes usar, modificar y distribuir el código libremente siempre que mantengas la misma licencia en los trabajos derivados y atribuyas la fuente original.

---

<div align="center">

Desarrollado con foco en el ecosistema de innovación de América Latina.

Si este repositorio te fue útil, considera dejar una ⭐ en GitHub.

<br/>

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-)

</div>
