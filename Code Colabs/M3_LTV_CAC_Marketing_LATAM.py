"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÓDULO 3 — LTV/CAC: MARKETING ANALYTICS                                  ║
║   Business Analytics para Startups LATAM                                   ║
║   Caso: EdTech de cursos profesionales online (B2C + B2B)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS PREVIOS:
    pip install pandas numpy matplotlib seaborn

DATOS QUE NECESITAS:
    - Tabla de clientes: ID, canal de adquisición, fecha_inicio, fecha_churn
    - Tabla de pagos: cliente_id, fecha, monto
    - Tabla de costos por canal: canal, inversión_mes, n_clientes_adquiridos
    Mínimo recomendado: 3 meses de cohortes activas con 50+ clientes cada una.

CASO DE USO:
    EdTech similar a Platzi (Colombia/LATAM), Crehana (Perú) o Aprende Institute.
    Objetivo: Entender qué canales son rentables y cuánto podemos gastar por cliente.

FÓRMULAS CLAVE:
    LTV  = ARPU × Gross Margin % / Churn Rate mensual
    CAC  = Inversión en Marketing / Nuevos Clientes Adquiridos
    LTV/CAC ratio ≥ 3x → canal rentable (benchmark industria)
    Payback Period = CAC / (ARPU × Gross Margin %)

ESTRUCTURA:
    1. Datos de clientes y cohortes de adquisición
    2. Cálculo de ARPU, churn y LTV por canal
    3. Cálculo de CAC por canal y período
    4. Análisis LTV/CAC y Payback Period
    5. Análisis de cohortes (retención mes a mes)
    6. Dashboard ejecutivo y recomendaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN VISUAL ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE_CANALES = {
    'Orgánico/SEO':     '#2ECC71',
    'Paid Social':      '#E74C3C',
    'Referidos':        '#3498DB',
    'Email Marketing':  '#F39C12',
    'Paid Search':      '#9B59B6',
}
plt.rcParams.update({'figure.figsize': (13, 7), 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
np.random.seed(42)

print("=" * 70)
print("💵 MÓDULO 3: LTV/CAC — EDTECH LATAM")
print("   Similar a: Platzi, Crehana, Aprende Institute, Descomplica")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — DATASET DE CLIENTES Y TRANSACCIONES
# ══════════════════════════════════════════════════════════════════════════════

CANALES = list(PALETTE_CANALES.keys())
# Distribución de canales: SEO y referidos son el MVP para EdTech LATAM
P_CANALES = [0.30, 0.25, 0.20, 0.15, 0.10]

N_CLIENTES = 1200

# Características del canal influyen en la calidad del cliente que trae
PERFIL_CANAL = {
    # (ARPU_media, churn_mensual, costo_acq_usd)
    'Orgánico/SEO':     (42, 0.08, 12),
    'Paid Social':      (35, 0.16, 45),
    'Referidos':        (48, 0.06, 18),
    'Email Marketing':  (40, 0.10, 8),
    'Paid Search':      (38, 0.14, 52),
}

paises = ['México', 'Brasil', 'Argentina', 'Colombia', 'Chile', 'Perú']
p_pais = [0.22, 0.28, 0.15, 0.18, 0.10, 0.07]
planes  = ['Mensual', 'Trimestral', 'Anual']
p_plan  = [0.50, 0.30, 0.20]

# Generar fechas de adquisición: últimos 18 meses
fecha_inicio_periodo = datetime(2024, 10, 1)
fecha_fin_periodo    = datetime(2026, 4, 1)
delta_dias = (fecha_fin_periodo - fecha_inicio_periodo).days

clientes = []
for i in range(N_CLIENTES):
    canal   = np.random.choice(CANALES, p=P_CANALES)
    arpu_b, churn_b, _ = PERFIL_CANAL[canal]

    fecha_adq = fecha_inicio_periodo + timedelta(days=int(np.random.uniform(0, delta_dias)))
    plan      = np.random.choice(planes, p=p_plan)

    # ARPU varía por plan: anual tiene descuento pero mayor commitment
    factor_plan = {'Mensual': 1.0, 'Trimestral': 0.90, 'Anual': 0.75}[plan]
    arpu = arpu_b * factor_plan * np.random.uniform(0.85, 1.15)

    # Churn mensual varía por calidad del canal
    churn_m = churn_b * np.random.uniform(0.8, 1.2)

    # Meses activos (tiempo de vida observado)
    meses_max = (fecha_fin_periodo - fecha_adq).days // 30
    meses_activo = max(1, min(int(np.random.geometric(churn_m)), meses_max))

    churned = meses_activo < meses_max

    clientes.append({
        'cliente_id':    f"CLI-{i+1:05d}",
        'canal':         canal,
        'plan':          plan,
        'pais':          np.random.choice(paises, p=p_pais),
        'fecha_adq':     fecha_adq,
        'meses_activo':  meses_activo,
        'arpu_usd':      round(arpu, 2),
        'churn_rate_m':  round(churn_m, 4),
        'churned':       int(churned),
    })

df = pd.DataFrame(clientes)
df['revenue_total'] = df['arpu_usd'] * df['meses_activo']
df['mes_adq'] = df['fecha_adq'].dt.to_period('M').astype(str)

print(f"\n✅ Dataset creado: {len(df)} clientes")
print(f"   Revenue total generado: ${df['revenue_total'].sum():,.0f} USD")
print(f"   Churn global: {df['churned'].mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — TABLA DE COSTOS DE MARKETING POR CANAL
# ══════════════════════════════════════════════════════════════════════════════

# Inversión mensual en marketing por canal (últimos 6 meses promediados)
costos_marketing = {
    'canal':                CANALES,
    'inversion_mensual':    [3500, 12000, 4200, 1800, 9500],  # USD/mes
    'n_clientes_mes_prom':  [290, 266, 240, 180, 100],         # nuevos clientes/mes promedio
}
df_costos = pd.DataFrame(costos_marketing)
df_costos['cac_usd'] = (df_costos['inversion_mensual'] / df_costos['n_clientes_mes_prom']).round(2)

print("\nTabla de Costos de Adquisición por Canal (CAC):")
print(df_costos.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — CÁLCULO DE LTV Y LTV/CAC POR CANAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("📐 PASO 3: CÁLCULO LTV/CAC POR CANAL")
print("─" * 70)

# Gross Margin EdTech: 65-75% (infraestructura cloud, docentes, soporte)
GROSS_MARGIN = 0.68

# Métricas por canal
metricas_canal = (df.groupby('canal')
                    .agg(
                        n_clientes   = ('cliente_id', 'count'),
                        arpu_medio   = ('arpu_usd', 'mean'),
                        churn_medio  = ('churn_rate_m', 'mean'),
                        revenue_prom = ('revenue_total', 'mean'),
                    )
                    .reset_index())

# LTV = (ARPU × Gross Margin) / Churn mensual
metricas_canal['ltv_usd'] = (
    metricas_canal['arpu_medio'] * GROSS_MARGIN / metricas_canal['churn_medio']
).round(2)

# Payback period: cuántos meses para recuperar el CAC
metricas_canal = metricas_canal.merge(df_costos[['canal', 'cac_usd']], on='canal')
metricas_canal['ltv_cac_ratio'] = (metricas_canal['ltv_usd'] / metricas_canal['cac_usd']).round(2)
metricas_canal['payback_meses'] = (
    metricas_canal['cac_usd'] / (metricas_canal['arpu_medio'] * GROSS_MARGIN)
).round(1)

# Semáforo de rentabilidad
def semaforo(ratio):
    if ratio >= 3:   return '🟢 RENTABLE'
    elif ratio >= 2: return '🟡 MARGINAL'
    else:            return '🔴 NO RENTABLE'

metricas_canal['estado'] = metricas_canal['ltv_cac_ratio'].apply(semaforo)

print("\nTabla Maestra LTV/CAC por Canal:")
cols_show = ['canal', 'n_clientes', 'arpu_medio', 'churn_medio',
             'ltv_usd', 'cac_usd', 'ltv_cac_ratio', 'payback_meses', 'estado']
print(metricas_canal[cols_show].round(2).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — VISUALIZACIÓN LTV/CAC DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Dashboard LTV/CAC — EdTech LATAM\nAnálisis de Rentabilidad por Canal de Adquisición',
             fontsize=15, fontweight='bold', y=0.98)

gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: LTV vs CAC por canal (scatter con tamaño = n_clientes)
ax1 = fig.add_subplot(gs[0, 0])
colores = [PALETTE_CANALES[c] for c in metricas_canal['canal']]
sc = ax1.scatter(metricas_canal['cac_usd'], metricas_canal['ltv_usd'],
                 s=metricas_canal['n_clientes'] * 0.8, c=colores, alpha=0.8, edgecolors='white', lw=1)
for _, row in metricas_canal.iterrows():
    ax1.annotate(row['canal'].split('/')[0], (row['cac_usd'], row['ltv_usd']),
                 textcoords='offset points', xytext=(5, 5), fontsize=8)
# Línea LTV/CAC = 3x (umbral de rentabilidad)
max_cac = metricas_canal['cac_usd'].max() * 1.2
x_line = np.linspace(0, max_cac, 100)
ax1.plot(x_line, x_line * 3, 'r--', lw=1.5, label='LTV/CAC = 3x (umbral)')
ax1.plot(x_line, x_line * 1, 'orange', linestyle=':', lw=1.5, label='LTV/CAC = 1x (break even)')
ax1.set_xlabel('CAC (USD)')
ax1.set_ylabel('LTV (USD)')
ax1.set_title('LTV vs CAC por Canal\n(tamaño = # clientes)')
ax1.legend(fontsize=8)

# Plot 2: Ratio LTV/CAC por canal
ax2 = fig.add_subplot(gs[0, 1])
sorted_df = metricas_canal.sort_values('ltv_cac_ratio', ascending=True)
bar_colors = ['#2ECC71' if r >= 3 else '#F39C12' if r >= 2 else '#E74C3C'
              for r in sorted_df['ltv_cac_ratio']]
bars = ax2.barh(sorted_df['canal'], sorted_df['ltv_cac_ratio'],
                color=bar_colors, edgecolor='white', height=0.6)
ax2.axvline(3, color='green', linestyle='--', lw=2, label='Meta: 3x')
ax2.axvline(1, color='red', linestyle=':', lw=1.5, label='Break-even: 1x')
ax2.set_xlabel('Ratio LTV/CAC')
ax2.set_title('Ratio LTV/CAC por Canal\n(verde ≥ 3x = rentable)')
ax2.legend(fontsize=9)
for bar, val in zip(bars, sorted_df['ltv_cac_ratio']):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}x', va='center', fontsize=9, fontweight='bold')

# Plot 3: Payback Period
ax3 = fig.add_subplot(gs[0, 2])
sorted_pb = metricas_canal.sort_values('payback_meses')
pb_colors = ['#2ECC71' if p <= 6 else '#F39C12' if p <= 12 else '#E74C3C'
             for p in sorted_pb['payback_meses']]
bars3 = ax3.bar(sorted_pb['canal'], sorted_pb['payback_meses'],
                color=pb_colors, edgecolor='white')
ax3.axhline(6,  color='green',  linestyle='--', lw=1.5, label='Meta: 6 meses')
ax3.axhline(12, color='orange', linestyle=':', lw=1.5, label='Máximo saludable: 12 meses')
ax3.set_ylabel('Meses para recuperar CAC')
ax3.set_title('Payback Period por Canal\n(verde ≤ 6 meses = excelente)')
ax3.tick_params(axis='x', rotation=30)
ax3.legend(fontsize=8)
for bar, val in zip(bars3, sorted_pb['payback_meses']):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f'{val:.0f}m', ha='center', fontsize=9)

# Plot 4: Revenue acumulado proyectado por canal
ax4 = fig.add_subplot(gs[1, :2])
meses = np.arange(1, 25)
for _, row in metricas_canal.iterrows():
    # Modelo de retención: revenue mes M = n_clientes × ARPU × (1-churn)^M
    rev_acum = np.cumsum([row['n_clientes'] * row['arpu_medio'] * (1 - row['churn_medio'])**m
                          for m in meses])
    ax4.plot(meses, rev_acum / 1000, label=row['canal'],
             color=PALETTE_CANALES[row['canal']], lw=2.5, marker='o', markersize=3)
ax4.set_xlabel('Mes')
ax4.set_ylabel('Revenue Acumulado (USD miles)')
ax4.set_title('Proyección de Revenue Acumulado por Canal (24 meses)')
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))

# Plot 5: Resumen ejecutivo en texto
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
mejor_canal  = metricas_canal.loc[metricas_canal['ltv_cac_ratio'].idxmax(), 'canal']
peor_canal   = metricas_canal.loc[metricas_canal['ltv_cac_ratio'].idxmin(), 'canal']
mejor_ratio  = metricas_canal['ltv_cac_ratio'].max()
peor_ratio   = metricas_canal['ltv_cac_ratio'].min()

resumen = (
    f"RESUMEN EJECUTIVO\n\n"
    f"Gross Margin asumido: {GROSS_MARGIN:.0%}\n"
    f"Benchmark LATAM EdTech:\n"
    f"  LTV/CAC meta: ≥ 3.0x\n"
    f"  Payback meta: ≤ 6 meses\n\n"
    f"MEJOR canal: {mejor_canal}\n"
    f"  LTV/CAC = {mejor_ratio:.1f}x\n\n"
    f"PEOR canal: {peor_canal}\n"
    f"  LTV/CAC = {peor_ratio:.1f}x\n\n"
    f"Acción: escalar presupuesto\n"
    f"en {mejor_canal} y revisar\n"
    f"estrategia de {peor_canal}."
)
ax5.text(0.05, 0.95, resumen, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#EBF5FB', alpha=0.9))

plt.savefig('M3_ltv_cac_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Dashboard guardado: M3_ltv_cac_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — ANÁLISIS DE COHORTES DE RETENCIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("📅 PASO 5: ANÁLISIS DE COHORTES (RETENCIÓN MES A MES)")
print("─" * 70)

# Construir tabla de cohortes: % de clientes activos por mes desde adquisición
df['cohort'] = df['fecha_adq'].dt.to_period('M').astype(str)

# Simular retención por cohorte basada en distribución de meses activos
COHORTES = sorted(df['cohort'].unique())[-8:]  # últimas 8 cohortes
MAX_MES   = 8

cohort_data = {}
for cohorte in COHORTES:
    clientes_cohorte = df[df['cohort'] == cohorte]
    n_total = len(clientes_cohorte)
    fila = [100.0]  # mes 0: 100% activos por definición
    for mes in range(1, MAX_MES + 1):
        activos = (clientes_cohorte['meses_activo'] >= mes).sum()
        fila.append(round(activos / n_total * 100, 1))
    cohort_data[cohorte] = fila

cohort_df = pd.DataFrame(cohort_data, index=[f'M+{i}' for i in range(MAX_MES + 1)]).T

print("\nTabla de Retención por Cohorte (% clientes activos):")
print(cohort_df.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
mask = cohort_df.isnull()
sns.heatmap(cohort_df, annot=True, fmt='.0f', cmap='YlGn',
            mask=mask, ax=ax, linewidths=0.5,
            vmin=0, vmax=100,
            cbar_kws={'label': '% Clientes Activos'})
ax.set_title('Análisis de Cohortes — Retención Mensual (%)\nEdTech LATAM', fontsize=13, fontweight='bold')
ax.set_xlabel('Mes desde Adquisición')
ax.set_ylabel('Cohorte de Adquisición')
plt.tight_layout()
plt.savefig('M3_cohortes_retencion.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Cohortes guardadas: M3_cohortes_retencion.png")

# Retención promedio en M+3 y M+6 (métricas que piden los VCs)
m3_prom = cohort_df['M+3'].mean()
m6_prom = cohort_df['M+6'].mean() if 'M+6' in cohort_df.columns else None
print(f"\nRetención promedio M+3: {m3_prom:.1f}%  (Benchmark LATAM EdTech: ~60-70%)")
if m6_prom:
    print(f"Retención promedio M+6: {m6_prom:.1f}%  (Benchmark LATAM EdTech: ~45-60%)")


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🎯 CONCLUSIONES EJECUTIVAS — LTV/CAC EDTECH LATAM")
print("=" * 70)

rentables = metricas_canal[metricas_canal['ltv_cac_ratio'] >= 3]['canal'].tolist()
no_rent   = metricas_canal[metricas_canal['ltv_cac_ratio'] < 2]['canal'].tolist()

print(f"""
HALLAZGOS CLAVE:
  1. Canales rentables (LTV/CAC ≥ 3x): {', '.join(rentables) if rentables else 'Ninguno aún'}
     Escalar inversión en estos canales tiene ROI demostrado.

  2. Canales a revisar (LTV/CAC < 2x): {', '.join(no_rent) if no_rent else 'Ninguno en esta zona'}
     Evaluar si el problema es el CAC alto o la retención baja.
     Solución: mejorar targeting (baja CAC) o mejorar onboarding (mejora LTV).

  3. El canal Referidos tiene el churn más bajo: los clientes que llegan
     recomendados por otros tienen más fit con el producto.
     → Diseñar un programa de referidos activo con incentivos.

  4. Retención M+3 del {m3_prom:.0f}%: si está por debajo del 60%, hay un problema
     de onboarding o de valor percibido en las primeras semanas.

ACCIONES INMEDIATAS:
  → Calcular estos mismos KPIs con tus datos reales cada mes.
  → Si LTV/CAC < 3x en todos los canales, no escales marketing: arregla retención primero.
  → Define un MRR objetivo y trabaja hacia atrás: ¿cuántos clientes necesito con qué CAC?

PRÓXIMO MÓDULO RECOMENDADO:
  Para segmentar clientes por valor real → M4 (KMeans + RFM)
  Para proyectar revenue de los próximos 12 meses → M5 (Prophet Forecasting)
""")

print("─" * 70)
print("📁 GUARDAR ESTE SCRIPT:")
print("   /Code Colabs/M3_LTV_CAC_Marketing_LATAM.py")
print("─" * 70)
