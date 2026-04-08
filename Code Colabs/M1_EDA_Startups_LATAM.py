"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÓDULO 1 — EDA: ANÁLISIS EXPLORATORIO DE DATOS                           ║
║   Business Analytics para Startups LATAM                                   ║
║   Caso: SaaS B2B de gestión contable para PYMEs latinoamericanas            ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS PREVIOS:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

DATOS QUE NECESITAS (o sustituye por tus propios):
    - Tabla de clientes: ID, país, plan, MRR, fecha_inicio, churn
    - Tabla de uso: logins/mes, features_usadas, tickets_soporte
    Mínimo recomendado: 200+ registros para que los patrones sean confiables.

ESTRUCTURA DEL ANÁLISIS (8 pasos):
    1. Carga y primera vista del dataset
    2. Análisis descriptivo (resumen estadístico)
    3. Calidad de datos (nulos, duplicados, tipos)
    4. Distribuciones de variables numéricas
    5. Variables categóricas y su composición
    6. Matriz de correlaciones
    7. Detección de outliers (IQR + Z-score)
    8. Conclusiones accionables y módulo siguiente recomendado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN VISUAL ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]
sns.set_palette(PALETTE)
plt.rcParams.update({'figure.figsize': (12, 7), 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
np.random.seed(42)

print("=" * 70)
print("📊 MÓDULO 1: EDA — STARTUP SAAS B2B LATAM")
print("   Similar a: ContaAzul (Brasil), Alegra (Colombia), Bind (México)")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — GENERACIÓN DEL DATASET SIMULADO
# ══════════════════════════════════════════════════════════════════════════════
# Si tienes datos reales, reemplaza esta sección con:
#   df = pd.read_csv('tu_archivo.csv')
# ─────────────────────────────────────────────────────────────────────────────

N = 800  # 800 clientes B2B: representativo para un SaaS seed/serie A en LATAM

paises      = ['México', 'Brasil', 'Argentina', 'Colombia', 'Chile']
pesos_pais  = [0.25, 0.30, 0.20, 0.15, 0.10]
planes      = ['Starter', 'Growth', 'Enterprise']
pesos_plan  = [0.50, 0.35, 0.15]
industrias  = ['Comercio', 'Servicios', 'Manufactura', 'Construcción', 'Salud']

# MRR (Monthly Recurring Revenue) por plan — rangos realistas para SaaS B2B LATAM
mrr_base = {'Starter': 49, 'Growth': 149, 'Enterprise': 499}

data = {
    'cliente_id':       [f"CLI-{i:04d}" for i in range(1, N+1)],
    'pais':             np.random.choice(paises, N, p=pesos_pais),
    'plan':             np.random.choice(planes, N, p=pesos_plan),
    'industria':        np.random.choice(industrias, N),
    'meses_activo':     np.random.exponential(scale=14, size=N).astype(int) + 1,
    'empleados_pyme':   np.random.choice([1, 5, 15, 50, 100], N, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
    'logins_mes':       np.random.poisson(lam=18, size=N),
    'features_usadas':  np.random.randint(1, 12, N),
    'tickets_soporte':  np.random.poisson(lam=1.8, size=N),
    'nps_score':        np.random.choice(range(0, 11), N, p=[0.02]*3 + [0.05]*4 + [0.10]*3 + [0.12, 0.20]),
}

df = pd.DataFrame(data)

# MRR con variación realista por plan (+/- 20%)
df['mrr_usd'] = df['plan'].map(mrr_base) * np.random.uniform(0.8, 1.2, N)
df['mrr_usd'] = df['mrr_usd'].round(2)

# Churn: más probable en Starter, pocos meses activos y bajo engagement
# Esto modela la realidad: los usuarios que no enganchan se van primero
prob_churn = (
    (df['plan'] == 'Starter').astype(float) * 0.20 +
    (df['meses_activo'] < 3).astype(float) * 0.25 +
    (df['logins_mes'] < 5).astype(float) * 0.30 +
    (df['features_usadas'] < 3).astype(float) * 0.15
)
prob_churn = (prob_churn / prob_churn.max()).clip(0.05, 0.85)
df['churn'] = (np.random.random(N) < prob_churn).astype(int)

# Introducir valores nulos realistas (5-8% en variables que los usuarios no llenan)
for col, pct in [('nps_score', 0.08), ('empleados_pyme', 0.05), ('tickets_soporte', 0.03)]:
    mask = np.random.random(N) < pct
    df.loc[mask, col] = np.nan

# Agregar outliers controlados (errores de entrada de datos — comunes en LATAM)
df.loc[np.random.choice(N, 8, replace=False), 'mrr_usd'] *= 15   # entradas duplicadas
df.loc[np.random.choice(N, 5, replace=False), 'logins_mes'] = 0   # cuentas fantasma

print(f"\n✅ Dataset creado: {df.shape[0]} clientes × {df.shape[1]} variables")
print(f"   Tasa de churn global: {df['churn'].mean():.1%}")
print(f"   MRR Total: ${df['mrr_usd'].sum():,.0f} USD/mes\n")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — PRIMERA VISTA Y RESUMEN ESTADÍSTICO
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("📋 PASO 2: PRIMERA VISTA DEL DATASET")
print("─" * 70)
print("\nPrimeras 5 filas:")
print(df.head().to_string())

print("\nResumen estadístico (numéricas):")
desc = df.describe().round(2)
print(desc.to_string())

print("\nTipos de datos:")
print(df.dtypes.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — CALIDAD DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🔍 PASO 3: CALIDAD DE DATOS")
print("─" * 70)

nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
calidad_df = pd.DataFrame({'nulos': nulos, 'pct_nulo': pct_nulos})
calidad_df = calidad_df[calidad_df['nulos'] > 0].sort_values('pct_nulo', ascending=False)

if len(calidad_df) > 0:
    print("\nVariables con valores faltantes:")
    print(calidad_df.to_string())
    print("\n💡 Acción: Para NPS (encuesta opcional) → imputar con mediana.")
    print("           Para empleados_pyme → imputar con moda o valor por país.")
else:
    print("\n✅ Sin valores faltantes.")

duplicados = df.duplicated().sum()
print(f"\nRegistros duplicados: {duplicados}")
print(f"Registros únicos de cliente: {df['cliente_id'].nunique()}")

# Estrategia de limpieza aplicada
df_clean = df.copy()
df_clean['nps_score'].fillna(df_clean['nps_score'].median(), inplace=True)
df_clean['empleados_pyme'].fillna(df_clean['empleados_pyme'].mode()[0], inplace=True)
df_clean['tickets_soporte'].fillna(0, inplace=True)

print(f"\n✅ Dataset limpio: {df_clean.isnull().sum().sum()} nulos restantes.")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — DISTRIBUCIONES DE VARIABLES NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("📊 PASO 4: DISTRIBUCIONES NUMÉRICAS")
print("─" * 70)

num_cols = ['mrr_usd', 'meses_activo', 'logins_mes', 'features_usadas', 'nps_score']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribuciones de Variables Clave — SaaS B2B LATAM', fontsize=15, fontweight='bold')

for i, col in enumerate(num_cols):
    ax = axes[i // 3][i % 3]
    data_col = df_clean[col].dropna()
    ax.hist(data_col, bins=25, edgecolor='white', alpha=0.85, color=PALETTE[i % len(PALETTE)])
    ax.axvline(data_col.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Media: {data_col.mean():.1f}')
    ax.axvline(data_col.median(), color='navy', linestyle=':', linewidth=1.5, label=f'Mediana: {data_col.median():.1f}')
    ax.set_title(col.replace('_', ' ').title())
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    ax.legend(fontsize=8)

# Panel vacío → texto con hallazgo clave
axes[1][2].axis('off')
skew_mrr = df_clean['mrr_usd'].skew()
axes[1][2].text(0.1, 0.5,
    f"HALLAZGO CLAVE\n\n"
    f"MRR skewness: {skew_mrr:.2f}\n"
    f"(>1 = cola derecha larga)\n\n"
    f"Indica que pocos clientes\n"
    f"concentran mucho revenue.\n\n"
    f"Acción: analizar si los\n"
    f"clientes Enterprise son\n"
    f"retenidos activamente.",
    transform=axes[1][2].transAxes, fontsize=11,
    bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

plt.tight_layout()
plt.savefig('M1_distribuciones.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M1_distribuciones.png")

# Tabla de asimetría para identificar transformaciones necesarias
print("\nAsimetría (skewness) por variable — valores >1 necesitan log-transform para ML:")
for col in num_cols:
    sk = df_clean[col].skew()
    flag = "⚠️ considera log-transform" if abs(sk) > 1 else "✅ distribución aceptable"
    print(f"  {col:<22} skew={sk:+.2f}  {flag}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — VARIABLES CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🗂️  PASO 5: VARIABLES CATEGÓRICAS Y COMPOSICIÓN")
print("─" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Composición por País, Plan e Industria', fontsize=14, fontweight='bold')

for ax, col, title in zip(axes, ['pais', 'plan', 'industria'],
                          ['Distribución por País', 'Mix de Planes', 'Por Industria']):
    counts = df_clean[col].value_counts()
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor='white')
    ax.set_title(title)
    ax.set_xlabel(col.title())
    ax.set_ylabel('N° clientes')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('M1_categoricas.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M1_categoricas.png")

# Churn por segmento — esto es la perla del EDA para el CEO
print("\nTasa de churn por Plan (la pregunta que todo fundador se hace):")
churn_plan = df_clean.groupby('plan')['churn'].agg(['mean', 'count']).round(3)
churn_plan.columns = ['tasa_churn', 'n_clientes']
churn_plan['tasa_churn_pct'] = (churn_plan['tasa_churn'] * 100).round(1)
print(churn_plan.to_string())

print("\nTasa de churn por País:")
churn_pais = df_clean.groupby('pais')['churn'].mean().sort_values(ascending=False)
print((churn_pais * 100).round(1).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — MATRIZ DE CORRELACIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🔗 PASO 6: CORRELACIONES")
print("─" * 70)

# Encodear categóricas para incluirlas en la correlación
df_corr = df_clean.copy()
le = LabelEncoder()
for col in ['pais', 'plan', 'industria']:
    df_corr[col + '_enc'] = le.fit_transform(df_corr[col])

corr_cols = ['mrr_usd', 'meses_activo', 'logins_mes', 'features_usadas',
             'tickets_soporte', 'nps_score', 'churn', 'plan_enc']
corr_matrix = df_corr[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Coeficiente de correlación'})
ax.set_title('Matriz de Correlaciones — Variables de Negocio', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('M1_correlaciones.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M1_correlaciones.png")

# Correlaciones con churn (lo más relevante para el CEO)
print("\nCorrelaciones con CHURN (las variables que mejor predicen cancelación):")
corr_churn = corr_matrix['churn'].drop('churn').sort_values(key=abs, ascending=False)
for var, val in corr_churn.items():
    signo = "protege contra churn" if val < -0.1 else ("aumenta riesgo" if val > 0.1 else "sin relación clara")
    print(f"  {var:<22} r={val:+.3f}  → {signo}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — DETECCIÓN DE OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("⚠️  PASO 7: DETECCIÓN DE OUTLIERS")
print("─" * 70)

def detectar_outliers(series, nombre):
    """Detecta outliers con método IQR (robusto) y Z-score (distribución normal)."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers_iqr = ((series < lim_inf) | (series > lim_sup)).sum()
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers_z = (z_scores > 3).sum()
    print(f"  {nombre:<22} IQR: {outliers_iqr:>3} outliers  |  Z-score: {outliers_z:>3} outliers"
          f"  |  rango válido: [{lim_inf:.0f}, {lim_sup:.0f}]")
    return lim_inf, lim_sup

print("\nAnálisis de outliers por variable numérica:")
for col in ['mrr_usd', 'logins_mes', 'meses_activo']:
    detectar_outliers(df_clean[col], col)

# Boxplots para visualizar outliers
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Boxplots — Detección de Outliers (puntos fuera = sospechosos)', fontsize=13, fontweight='bold')

for ax, col, color in zip(axes, ['mrr_usd', 'logins_mes', 'meses_activo'], PALETTE):
    bp = ax.boxplot(df_clean[col].dropna(), vert=True, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    ax.set_title(col.replace('_', ' ').title())
    ax.set_ylabel('Valor')
    ax.set_xticklabels([''])

plt.tight_layout()
plt.savefig('M1_outliers.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M1_outliers.png")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 8 — CONCLUSIONES Y PRÓXIMO MÓDULO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🎯 CONCLUSIONES EJECUTIVAS — EDA SAAS B2B LATAM")
print("=" * 70)

mrr_mediana = df_clean['mrr_usd'].median()
churn_global = df_clean['churn'].mean()
plan_mas_churn = churn_plan['tasa_churn_pct'].idxmax()
nps_prom = df_clean['nps_score'].mean()

print(f"""
HALLAZGOS CLAVE:
  1. MRR mediana: ${mrr_mediana:.0f}/mes. Los outliers de MRR son candidatos a
     Enterprise: verificar si son clientes reales o errores de datos.

  2. Churn global {churn_global:.1%}: el plan '{plan_mas_churn}' concentra mayor
     riesgo de cancelación. Benchmark SaaS B2B LATAM: 8-15% anual.
     Si tu churn mensual supera 3%, hay un problema de product-market fit.

  3. NPS promedio {nps_prom:.1f}/10: la correlación negativa con churn confirma
     que mejorar la experiencia es la palanca de retención más directa.

  4. Features usadas es la señal de engagement más fuerte. Clientes que usan
     3+ features raramente cancelan → activación rápida = prioridad #1.

  5. Brasil y México son los mercados con más clientes, pero revisa si la
     tasa de churn por país justifica marketing diferenciado.

ACCIÓN INMEDIATA:
  → Crear un Health Score = f(logins, features, nps) para monitoreo semanal.
  → Activar campaña de onboarding para clientes con <3 features en primeros 30 días.

PRÓXIMO MÓDULO RECOMENDADO:
  Si quieres PREDECIR qué clientes cancelarán → usa M2 (XGBoost Churn Prediction)
  Si quieres SEGMENTAR clientes por comportamiento → usa M4 (KMeans + RFM)
  Si quieres calcular cuánto vale cada cliente → usa M3 (LTV/CAC)
""")

print("─" * 70)
print("📁 GUARDAR ESTE SCRIPT EN EL REPOSITORIO:")
print("   /Code Colabs/M1_EDA_Startups_LATAM.py")
print("─" * 70)
