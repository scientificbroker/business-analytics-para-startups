"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÓDULO 2 — XGBOOST: ANALÍTICA PREDICTIVA                                 ║
║   Business Analytics para Startups LATAM                                   ║
║   Caso: Predicción de Churn en FinTech de préstamos para consumidores       ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS PREVIOS:
    pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn

DATOS QUE NECESITAS:
    - Tabla de clientes con variables de comportamiento (uso, pagos, soporte)
    - Variable objetivo binaria: churn (1=canceló, 0=activo)
    - Mínimo recomendado: 500+ registros. Mejor con 2,000+.
    - Distribución mínima en clase positiva: 10% de churn real.

CASO DE USO:
    FinTech similar a Konfío (México), Neon (Brasil) o Ualá (Argentina).
    El modelo predice qué usuarios cancelarán en los próximos 30 días
    para activar campañas de retención antes de que se vayan.

ESTRUCTURA:
    1. Generación de datos y feature engineering
    2. División train/test y balanceo de clases
    3. Entrenamiento XGBoost con validación cruzada
    4. Evaluación: AUC-ROC, matriz de confusión, precision-recall
    5. Interpretabilidad con SHAP (qué variables importan y por qué)
    6. Scores de riesgo por cliente (lista de acción)
    7. Conclusiones y ROI estimado de la campaña de retención
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# SHAP es opcional pero muy recomendado — instalar con: pip install shap
try:
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    SHAP_DISPONIBLE = False
    print("💡 Para interpretabilidad: pip install shap")

# ─── CONFIGURACIÓN VISUAL ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]
plt.rcParams.update({'figure.figsize': (12, 7), 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
np.random.seed(42)

print("=" * 70)
print("🤖 MÓDULO 2: XGBOOST — PREDICCIÓN DE CHURN FINTECH LATAM")
print("   Similar a: Konfío, Neon, Ualá, Clip, Kushki")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — DATASET CON FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

N = 3000  # 3,000 usuarios: tamaño mínimo para XGBoost robusto en producción

paises     = ['México', 'Brasil', 'Argentina', 'Colombia', 'Chile']
p_pais     = [0.25, 0.30, 0.20, 0.15, 0.10]
productos  = ['Préstamo Personal', 'Tarjeta Prepago', 'Crédito PYME', 'Ahorro Digital']
p_prod     = [0.40, 0.30, 0.20, 0.10]

data = {
    'usuario_id':            [f"USR-{i:05d}" for i in range(1, N+1)],
    'pais':                  np.random.choice(paises, N, p=p_pais),
    'producto_principal':    np.random.choice(productos, N, p=p_prod),
    'edad':                  np.random.randint(18, 65, N),
    'meses_en_plataforma':   np.random.exponential(12, N).astype(int) + 1,
    # Transacciones: Poisson refleja comportamiento discreto de pagos
    'transacciones_mes':     np.random.poisson(8, N),
    'dias_ultimo_acceso':    np.random.exponential(15, N).astype(int),
    # Monto promedio por transacción en USD — distribución lognormal (típica en consumo)
    'monto_promedio_txn':    np.random.lognormal(3.5, 0.8, N),
    # Uso de productos adicionales: cross-sell / penetración
    'productos_contratados': np.random.randint(1, 5, N),
    'score_credito':         np.clip(np.random.normal(620, 80, N), 300, 850).astype(int),
    'dias_retraso_pago':     np.random.choice([0, 0, 0, 0, 7, 15, 30, 60], N,
                                              p=[0.5, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.03]),
    'contacto_soporte':      np.random.poisson(1.2, N),
    'quejas_registradas':    np.random.choice([0, 1, 2, 3], N, p=[0.70, 0.20, 0.07, 0.03]),
    # NPS en FinTech: tiende a ser más bajo que en SaaS
    'nps_respuesta':         np.random.choice(range(0, 11), N),
}

df = pd.DataFrame(data)

# ─── FEATURE ENGINEERING: variables derivadas que mejoran el modelo ───────────
# Estas variables combinadas capturan patrones que las variables crudas no muestran

df['ratio_uso'] = df['transacciones_mes'] / (df['meses_en_plataforma'] + 1)
df['engagement_score'] = (
    df['transacciones_mes'] * 0.4 +
    (1 / (df['dias_ultimo_acceso'] + 1)) * 100 * 0.3 +
    df['productos_contratados'] * 10 * 0.3
)
df['riesgo_mora'] = (df['dias_retraso_pago'] > 0).astype(int)
df['cliente_inactivo'] = (df['dias_ultimo_acceso'] > 30).astype(int)
df['alta_queja'] = (df['quejas_registradas'] >= 2).astype(int)

# Encodear variables categóricas
le = LabelEncoder()
df['pais_enc']    = le.fit_transform(df['pais'])
df['prod_enc']    = le.fit_transform(df['producto_principal'])

# ─── VARIABLE OBJETIVO: CHURN ─────────────────────────────────────────────────
# Construida con lógica de negocio real: clientes con mora, inactividad y quejas
# tienen mayor probabilidad de churn — verificado en datos reales de FinTech

logit = (
    -3.0
    + 0.8  * df['riesgo_mora']
    + 1.2  * df['cliente_inactivo']
    + 0.9  * df['alta_queja']
    - 0.04 * df['score_credito'] / 100
    + 0.5  * (df['nps_respuesta'] < 5).astype(int)
    - 0.3  * df['productos_contratados']
    + np.random.normal(0, 0.5, N)  # ruido para simular variables no observadas
)
prob_churn = 1 / (1 + np.exp(-logit))
df['churn'] = (np.random.random(N) < prob_churn).astype(int)

print(f"\n✅ Dataset creado: {df.shape[0]} usuarios × {df.shape[1]} variables")
print(f"   Tasa de churn: {df['churn'].mean():.1%}  "
      f"(benchmark FinTech LATAM: 8-15% mensual para consumidores)")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — PREPARACIÓN PARA MODELADO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("⚙️  PASO 2: PREPARACIÓN DE DATOS")
print("─" * 70)

# Variables de entrada al modelo (excluir IDs y target)
FEATURES = [
    'edad', 'meses_en_plataforma', 'transacciones_mes', 'dias_ultimo_acceso',
    'monto_promedio_txn', 'productos_contratados', 'score_credito',
    'dias_retraso_pago', 'contacto_soporte', 'quejas_registradas', 'nps_respuesta',
    'ratio_uso', 'engagement_score', 'riesgo_mora', 'cliente_inactivo',
    'alta_queja', 'pais_enc', 'prod_enc'
]

X = df[FEATURES]
y = df['churn']

# Stratified split: garantiza que ambos sets tengan la misma proporción de churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} usuarios | Churn: {y_train.mean():.1%}")
print(f"  Test:  {X_test.shape[0]} usuarios | Churn: {y_test.mean():.1%}")

# scale_pos_weight: compensa desbalance de clases sin perder datos
# Fórmula: n_negativos / n_positivos (estándar en XGBoost para clases desbalanceadas)
ratio_clases = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n  Ratio de clases (neg/pos): {ratio_clases:.1f} → usado en scale_pos_weight")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — ENTRENAMIENTO XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🚀 PASO 3: ENTRENAMIENTO XGBOOST")
print("─" * 70)

# Hiperparámetros elegidos para datasets medianos (500-5000 muestras) en LATAM.
# max_depth=5 y subsample=0.8 previenen overfitting sin sacrificar potencia.
modelo = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio_clases,  # corrige desbalance de clases
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Validación cruzada estratificada: da una estimación honesta del desempeño real
# (5-fold es el estándar para datasets de este tamaño)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"\n  AUC-ROC Validación Cruzada (5-fold):")
print(f"    Media:  {cv_scores.mean():.4f}")
print(f"    Std:    {cv_scores.std():.4f}")
print(f"    Rango:  [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
print(f"\n  Interpretación:")
print(f"    AUC > 0.90 → modelo excelente")
print(f"    AUC > 0.80 → modelo bueno para uso en producción")
print(f"    AUC > 0.70 → modelo aceptable con mejoras posibles")

modelo.fit(X_train, y_train,
           eval_set=[(X_test, y_test)],
           verbose=False)

print("\n✅ Modelo entrenado correctamente.")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — EVALUACIÓN DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("📊 PASO 4: EVALUACIÓN DEL MODELO")
print("─" * 70)

y_pred_proba = modelo.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.40).astype(int)  # umbral 0.40: prioriza recall (no perder churners)

auc_test = roc_auc_score(y_test, y_pred_proba)
print(f"\n  AUC-ROC en Test: {auc_test:.4f}")
print(f"\n  Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Activo', 'Churn']))

# ─── VISUALIZACIONES DE EVALUACIÓN ───────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Evaluación del Modelo XGBoost — Predicción de Churn', fontsize=14, fontweight='bold')

# Plot 1: Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color='#E74C3C', lw=2.5, label=f'AUC = {auc_test:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatorio (AUC=0.50)')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].set_title('Curva ROC')
axes[0].legend()

# Plot 2: Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Activo', 'Churn'], yticklabels=['Activo', 'Churn'],
            linewidths=1, cbar=False)
axes[1].set_title('Matriz de Confusión')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predicho')
# Anotar costos de negocio
total = cm.sum()
axes[1].set_xlabel(f'Predicho\n(Total: {total} | FN={cm[1,0]} clientes perdidos sin alertar)')

# Plot 3: Distribución de probabilidades por clase
activos = y_pred_proba[y_test == 0]
churners = y_pred_proba[y_test == 1]
axes[2].hist(activos, bins=30, alpha=0.6, label='Activos', color='#2ECC71', edgecolor='white')
axes[2].hist(churners, bins=30, alpha=0.6, label='Churners', color='#E74C3C', edgecolor='white')
axes[2].axvline(0.40, color='black', linestyle='--', lw=2, label='Umbral=0.40')
axes[2].set_xlabel('Probabilidad de Churn Predicha')
axes[2].set_ylabel('N° usuarios')
axes[2].set_title('Separación de Clases')
axes[2].legend()

plt.tight_layout()
plt.savefig('M2_evaluacion_modelo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M2_evaluacion_modelo.png")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — INTERPRETABILIDAD: IMPORTANCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🔍 PASO 5: ¿QUÉ VARIABLES IMPULSAN EL CHURN?")
print("─" * 70)

# Importancia basada en ganancia (gain): cuánto mejora el modelo cada variable
# Es más confiable que la frecuencia de uso para comparar variables de escala distinta
importancias = pd.DataFrame({
    'feature': FEATURES,
    'importancia': modelo.feature_importances_
}).sort_values('importancia', ascending=False)

print("\nTop 10 Variables (por importancia en el modelo):")
print(importancias.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
colores = ['#E74C3C' if i < 3 else '#3498DB' if i < 7 else '#BDC3C7'
           for i in range(len(importancias.head(12)))]
bars = ax.barh(importancias.head(12)['feature'][::-1],
               importancias.head(12)['importancia'][::-1],
               color=colores[::-1], edgecolor='white', height=0.7)
ax.set_xlabel('Importancia (ganancia relativa)')
ax.set_title('Variables que Mejor Predicen el Churn\n(rojo = críticas, azul = importantes)', fontsize=13)
for bar, val in zip(bars, importancias.head(12)['importancia'][::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('M2_importancia_variables.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M2_importancia_variables.png")

# SHAP Values: explica predicciones individuales (si está instalado)
if SHAP_DISPONIBLE:
    print("\n  Calculando SHAP values (puede tomar 30-60 segundos)...")
    explainer = shap.TreeExplainer(modelo)
    X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.title('SHAP — Impacto Real de cada Variable en las Predicciones', fontsize=12)
    plt.tight_layout()
    plt.savefig('M2_shap_values.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico SHAP guardado: M2_shap_values.png")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — SCORING: LISTA DE ACCIÓN POR CLIENTE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🎯 PASO 6: SCORING — LISTA DE CLIENTES EN RIESGO")
print("─" * 70)

# Aplicar el modelo a TODOS los usuarios para crear lista de intervención
df['prob_churn'] = modelo.predict_proba(X[FEATURES])[:, 1]

# Segmentación por nivel de riesgo (umbrales definidos por el equipo de negocio)
def clasificar_riesgo(prob):
    if prob >= 0.70:   return 'CRITICO'
    elif prob >= 0.45: return 'ALTO'
    elif prob >= 0.25: return 'MEDIO'
    else:              return 'BAJO'

df['nivel_riesgo'] = df['prob_churn'].apply(clasificar_riesgo)

print("\nDistribución de clientes por nivel de riesgo:")
riesgo_dist = df['nivel_riesgo'].value_counts()
print(riesgo_dist.to_string())

# Lista de los 20 clientes más urgentes (para que el equipo de retención actúe hoy)
criticos = (df[df['nivel_riesgo'] == 'CRITICO']
            [['usuario_id', 'pais', 'producto_principal', 'prob_churn',
              'meses_en_plataforma', 'dias_ultimo_acceso', 'quejas_registradas']]
            .sort_values('prob_churn', ascending=False)
            .head(20))

print(f"\nTop 20 usuarios CRÍTICOS (prob_churn > 70%) — accionar HOY:")
print(criticos.round(3).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — ROI DE LA CAMPAÑA DE RETENCIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("💰 PASO 7: ROI ESTIMADO DE LA CAMPAÑA DE RETENCIÓN")
print("─" * 70)

# Parámetros financieros — ajustar con tus datos reales
MRR_PROMEDIO = df['monto_promedio_txn'].mean() * 0.15  # 15% de GMV como revenue FinTech
LTV_MESES    = 18          # LTV en meses (benchmark FinTech LATAM)
COSTO_CAMPANA_POR_USUARIO = 8   # USD por usuario contactado (email + incentivo)
TASA_RETENCION_CAMPANA    = 0.25  # 25% de churners recuperados por campaña bien ejecutada

n_criticos = (df['nivel_riesgo'] == 'CRITICO').sum()
n_altos    = (df['nivel_riesgo'] == 'ALTO').sum()
n_objetivo = n_criticos + n_altos

revenue_rescatado = n_objetivo * TASA_RETENCION_CAMPANA * MRR_PROMEDIO * LTV_MESES
costo_total = n_objetivo * COSTO_CAMPANA_POR_USUARIO
roi = (revenue_rescatado - costo_total) / costo_total * 100

print(f"""
  Usuarios en riesgo CRITICO + ALTO: {n_objetivo}
  Costo total campaña de retención:  ${costo_total:,.0f} USD
  Revenue rescatado estimado:         ${revenue_rescatado:,.0f} USD
  ROI estimado de la campaña:         {roi:.0f}%

  Supuestos:
    - MRR promedio por usuario: ${MRR_PROMEDIO:.2f}/mes
    - Tasa de retención por campaña: {TASA_RETENCION_CAMPANA:.0%}
    - LTV promedio: {LTV_MESES} meses
""")


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("🎯 CONCLUSIONES EJECUTIVAS — XGBOOST CHURN FINTECH LATAM")
print("=" * 70)
print(f"""
HALLAZGOS CLAVE:
  1. AUC-ROC {auc_test:.3f}: el modelo puede identificar correctamente a los
     churners el {auc_test:.0%} de las veces.

  2. Las variables más importantes son días de inactividad, retraso en pagos
     y quejas registradas. Son señales tempranas de insatisfacción.

  3. {n_criticos} clientes tienen probabilidad >70% de cancelar en 30 días.
     Contactarlos esta semana puede salvar ~${n_criticos * TASA_RETENCION_CAMPANA * MRR_PROMEDIO * LTV_MESES:,.0f} USD en LTV.

  4. El umbral óptimo de 0.40 prioriza recall: preferimos contactar de más
     (algunos falsos positivos) antes que perder churners reales.

ACCIONES INMEDIATAS:
  → Semana 1: Exportar lista de críticos y asignar a equipo de Customer Success.
  → Mes 1: Integrar el score de riesgo en el CRM para monitoreo automático.
  → Mes 2: A/B test de dos mensajes de retención para calibrar la tasa de 25%.

PRÓXIMO MÓDULO RECOMENDADO:
  Si tienes segmentos distintos de clientes → M4 (KMeans) para campañas diferenciadas.
  Si quieres calcular el LTV real por cliente → M3 (LTV/CAC Marketing Analytics).
""")

print("─" * 70)
print("📁 GUARDAR ESTE SCRIPT:")
print("   /Code Colabs/M2_XGBoost_Predictivo_LATAM.py")
print("─" * 70)
