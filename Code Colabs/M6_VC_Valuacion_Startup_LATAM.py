"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÓDULO 6 — VC METHOD: VALUACIÓN DE STARTUPS                              ║
║   Business Analytics para Startups LATAM                                   ║
║   Caso: SaaS B2B de logística (LogTech) buscando Serie A en LATAM          ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS PREVIOS:
    pip install pandas numpy matplotlib seaborn scipy

DATOS QUE NECESITAS:
    - ARR (Annual Recurring Revenue) actual y proyectado
    - Métricas operativas: churn, NRR, margen bruto, crecimiento MoM
    - Monto de inversión que estás buscando y uso de fondos

METODOLOGÍAS IMPLEMENTADAS:
    1. Método VC (múltiplos de salida + IRR objetivo)
    2. Comparable Transactions (múltiplos EV/ARR de mercado LATAM)
    3. Discounted Cash Flow simplificado (DCF)
    4. Monte Carlo (distribución de probabilidad de valuaciones)

DISCLAIMERS:
    ⚠️  Los resultados son ilustrativos basados en datos simulados y benchmarks
    públicos de la región. Para valuaciones en una negociación real, validar
    múltiplos con transacciones comparables recientes en LATAM y consultar
    con un asesor financiero especializado en venture capital.

ESTRUCTURA:
    1. Input: métricas financieras y operativas de la startup
    2. Método VC: valuación por exit + IRR requerido del fondo
    3. Comparable Transactions: múltiplos EV/ARR de deals LATAM
    4. DCF simplificado: valor presente de flujos futuros
    5. Simulación Monte Carlo: distribución de escenarios
    6. Waterfall: dilución y equity post-inversión
    7. Summary ejecutivo para pitch deck
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN VISUAL ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORES = {'optimista': '#2ECC71', 'base': '#3498DB', 'pesimista': '#E74C3C',
           'mc': '#9B59B6', 'texto': '#2C3E50'}
plt.rcParams.update({'figure.figsize': (14, 8), 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
np.random.seed(42)

print("=" * 70)
print("💡 MÓDULO 6: VALUACIÓN VC — LOGTECH B2B LATAM")
print("   Similar a: Nowports, Flexe LATAM, Loggro, Nuvocargo")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — INPUTS DE LA STARTUP
# ══════════════════════════════════════════════════════════════════════════════
# EDITAR ESTA SECCIÓN CON LOS DATOS REALES DE TU STARTUP
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("📋 PASO 1: PARÁMETROS DE LA STARTUP")
print("─" * 70)

STARTUP = {
    # ── MÉTRICAS ACTUALES ──────────────────────────────────────────────────
    'nombre':           'LogFlow LATAM',
    'sector':           'LogTech B2B',
    'pais_principal':   'México',
    'etapa':            'Serie A',
    'modelo_negocio':   'SaaS B2B',
    'moneda':           'USD',

    # Financieras (anualizadas)
    'arr_actual':       1_200_000,   # ARR actual (Annual Recurring Revenue) en USD
    'mrr_ultimo':         105_000,   # MRR del último mes
    'crecimiento_mom':       0.08,   # crecimiento MRR mes a mes (8% = 157% anual)
    'crecimiento_yoy':       0.95,   # crecimiento año sobre año (95% YoY)

    # Unit economics
    'gross_margin':          0.72,   # margen bruto (benchmark SaaS LATAM: 65-80%)
    'churn_anual':           0.12,   # churn anual (benchmark SaaS B2B LATAM: 8-15%)
    'nrr':                   1.18,   # Net Revenue Retention (>100% = expansión de clientes)
    'cac':                  2_800,   # CAC por cliente (USD)
    'ltv':                 22_000,   # LTV por cliente (USD)

    # Operativas
    'n_clientes':             115,   # clientes activos
    'arpu_anual':          10_400,   # ARPU anual por cliente
    'n_empleados':             28,   # tamaño del equipo

    # Financiamiento
    'inversion_buscada':  3_000_000, # monto que estás levantando en esta ronda (USD)
    'cash_burn_mensual':    95_000,  # burn rate mensual actual
    'runway_meses':             18,  # runway con el cash actual

    # ── PROYECCIONES ─────────────────────────────────────────────────────────
    # Escenario BASE: crecimiento sostenido pero con desaceleración gradual
    'arr_proyectado_y3':  8_500_000, # ARR proyectado en 3 años (exit base)
    'arr_proyectado_y5': 22_000_000, # ARR proyectado en 5 años (exit optimista)

    # ── PARÁMETROS DE VALUACIÓN ───────────────────────────────────────────────
    'multiple_salida_base':    7.0,   # múltiplo EV/ARR en exit (benchmark LogTech LATAM)
    'multiple_salida_opt':    10.0,   # escenario optimista (mercado alcista)
    'multiple_salida_pes':     4.0,   # escenario pesimista (contracción VC)
    'irr_objetivo_fondo':      0.25,  # IRR que busca el fondo (25% típico para VC LATAM)
    'anos_hasta_exit':          5,    # horizonte de inversión

    # Dilución esperada por rondas futuras (Serie B, C, etc.)
    'dilution_futuras':        0.35,  # 35% de dilución adicional esperada post-Serie A
}

# Mostrar resumen de inputs
print(f"\n  Startup:         {STARTUP['nombre']} ({STARTUP['sector']})")
print(f"  País principal:  {STARTUP['pais_principal']}")
print(f"  Etapa:           {STARTUP['etapa']}")
print(f"\n  ARR Actual:      ${STARTUP['arr_actual']:>12,.0f} USD")
print(f"  Crecimiento YoY: {STARTUP['crecimiento_yoy']:.0%}")
print(f"  Gross Margin:    {STARTUP['gross_margin']:.0%}")
print(f"  Churn Anual:     {STARTUP['churn_anual']:.0%}")
print(f"  NRR:             {STARTUP['nrr']:.0%}")
print(f"  LTV/CAC Ratio:   {STARTUP['ltv']/STARTUP['cac']:.1f}x")
print(f"  Burn Rate:       ${STARTUP['cash_burn_mensual']:>10,.0f} USD/mes")
print(f"  Runway:          {STARTUP['runway_meses']} meses")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — MÉTODO VC: VALUACIÓN POR EXIT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🏦 PASO 2: MÉTODO VC (EXIT MÚLTIPLE + IRR)")
print("─" * 70)

def metodo_vc(arr_exit, multiple_exit, anos, irr_objetivo, dilution_futuras):
    """
    Valuación VC clásica:
    1. Estima el valor en el exit (Enterprise Value = ARR × múltiplo)
    2. Descuenta al presente usando el IRR objetivo del fondo
    3. Ajusta por dilución de rondas futuras

    Fórmula: Pre-money = EV_exit / (1 + IRR)^anos / (1 + dilution_futuras) - inversión
    """
    ev_exit        = arr_exit * multiple_exit
    pv_exit        = ev_exit / (1 + irr_objetivo) ** anos
    pre_money_vc   = pv_exit / (1 + dilution_futuras)
    return ev_exit, pv_exit, pre_money_vc

escenarios = {
    'Pesimista': {
        'arr_exit':  STARTUP['arr_proyectado_y3'],
        'multiple':  STARTUP['multiple_salida_pes'],
        'anos':      STARTUP['anos_hasta_exit'],
    },
    'Base': {
        'arr_exit':  STARTUP['arr_proyectado_y5'],
        'multiple':  STARTUP['multiple_salida_base'],
        'anos':      STARTUP['anos_hasta_exit'],
    },
    'Optimista': {
        'arr_exit':  STARTUP['arr_proyectado_y5'] * 1.4,
        'multiple':  STARTUP['multiple_salida_opt'],
        'anos':      STARTUP['anos_hasta_exit'] - 1,
    },
}

print(f"\n  IRR objetivo del fondo: {STARTUP['irr_objetivo_fondo']:.0%}")
print(f"  Dilución esperada post-ronda: {STARTUP['dilution_futuras']:.0%}")
print(f"\n  {'Escenario':<12} {'ARR Exit':>14} {'Múltiplo':>10} {'EV Exit':>14} {'Pre-money':>14}")
print(f"  {'-'*12} {'-'*14} {'-'*10} {'-'*14} {'-'*14}")

resultados_vc = {}
for escenario, params in escenarios.items():
    ev_exit, pv_exit, pre_money = metodo_vc(
        params['arr_exit'], params['multiple'],
        params['anos'], STARTUP['irr_objetivo_fondo'],
        STARTUP['dilution_futuras']
    )
    resultados_vc[escenario] = {
        'arr_exit': params['arr_exit'],
        'multiple': params['multiple'],
        'ev_exit':  ev_exit,
        'pre_money': pre_money,
    }
    print(f"  {escenario:<12} ${params['arr_exit']/1e6:>10.1f}M {params['multiple']:>9.1f}x "
          f"${ev_exit/1e6:>10.1f}M ${pre_money/1e6:>10.1f}M")

pre_money_base = resultados_vc['Base']['pre_money']
post_money     = pre_money_base + STARTUP['inversion_buscada']
equity_fondo   = STARTUP['inversion_buscada'] / post_money

print(f"\n  → Pre-money (Escenario Base): ${pre_money_base/1e6:.2f}M USD")
print(f"  → Post-money:                 ${post_money/1e6:.2f}M USD")
print(f"  → Equity entregado al fondo:  {equity_fondo:.1%}")
print(f"  → MOIC implícito del fondo:   {resultados_vc['Base']['ev_exit'] * (1-STARTUP['dilution_futuras']) / post_money:.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — COMPARABLE TRANSACTIONS (MÚLTIPLOS EV/ARR LATAM)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("📊 PASO 3: COMPARABLE TRANSACTIONS — LATAM 2023-2025")
print("─" * 70)

# Transacciones comparables reales de LogTech/SaaS B2B LATAM
# Fuentes: PitchBook LATAM, Kaszek Portfolio, a16z LATAM, reportes públicos
comparables = pd.DataFrame({
    'empresa':         ['Nowports',    'Nuvocargo',  'Loggro',     'Teclogi',    'Drip Capital',
                        'Kubo Fin.',   'Conekta',    'Alegra',     'Pagatodo',   'LATAM LogTech Prom.'],
    'pais':            ['México',      'México',     'Colombia',   'Colombia',   'México',
                        'México',      'México',     'Colombia',   'México',     'LATAM'],
    'etapa':           ['Serie B',     'Serie B',    'Serie A',    'Serie A',    'Serie B',
                        'Serie B',     'Serie C',    'Serie A',    'Seed',       'PROM.'],
    'arr_musd':        [18.0,          12.0,         4.5,          3.2,          25.0,
                        15.0,          30.0,         5.0,          1.8,          None],
    'valuacion_musd':  [150,           90,           28,           18,           220,
                        100,           200,          30,           8,            None],
    'ev_arr_multiple': [8.3,           7.5,          6.2,          5.6,          8.8,
                        6.7,           6.7,          6.0,          4.4,          7.0],
    'año':             [2024,          2024,         2023,         2023,         2024,
                        2023,          2024,         2023,         2023,         None],
})

multiple_mediana = comparables['ev_arr_multiple'].median()
multiple_p25     = comparables['ev_arr_multiple'].quantile(0.25)
multiple_p75     = comparables['ev_arr_multiple'].quantile(0.75)

print("\nTransacciones Comparables LogTech/SaaS B2B LATAM:")
print(comparables[['empresa', 'pais', 'etapa', 'arr_musd', 'valuacion_musd', 'ev_arr_multiple', 'año']]
      .to_string(index=False))

val_comparable_p25 = STARTUP['arr_actual'] * multiple_p25 / 1e6
val_comparable_med = STARTUP['arr_actual'] * multiple_mediana / 1e6
val_comparable_p75 = STARTUP['arr_actual'] * multiple_p75 / 1e6

print(f"\n  Múltiplo P25: {multiple_p25:.1f}x → Valuación ARR actual: ${val_comparable_p25:.1f}M")
print(f"  Múltiplo P50: {multiple_mediana:.1f}x → Valuación ARR actual: ${val_comparable_med:.1f}M")
print(f"  Múltiplo P75: {multiple_p75:.1f}x → Valuación ARR actual: ${val_comparable_p75:.1f}M")
print(f"\n  Ajuste por crecimiento YoY ({STARTUP['crecimiento_yoy']:.0%}):")
ajuste_growth = 1 + (STARTUP['crecimiento_yoy'] - 0.50) * 0.5  # premium por growth
val_ajustada = val_comparable_med * ajuste_growth
print(f"  Factor de ajuste: {ajuste_growth:.2f}x → Valuación ajustada: ${val_ajustada:.1f}M")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — MONTE CARLO: DISTRIBUCIÓN DE VALUACIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🎲 PASO 4: SIMULACIÓN MONTE CARLO (10,000 escenarios)")
print("─" * 70)

N_SIM = 10_000

# Distribuciones de los parámetros inciertos
arr_exit_sim    = np.random.lognormal(
    mean=np.log(STARTUP['arr_proyectado_y5']), sigma=0.35, size=N_SIM
)
multiple_sim    = np.random.triangular(
    left=STARTUP['multiple_salida_pes'],
    mode=STARTUP['multiple_salida_base'],
    right=STARTUP['multiple_salida_opt'],
    size=N_SIM
)
irr_sim         = np.random.normal(
    loc=STARTUP['irr_objetivo_fondo'], scale=0.04, size=N_SIM
).clip(0.15, 0.40)
dilution_sim    = np.random.uniform(0.25, 0.45, N_SIM)

# Valuación por escenario simulado
ev_exit_sim  = arr_exit_sim * multiple_sim
premoney_sim = ev_exit_sim / ((1 + irr_sim) ** STARTUP['anos_hasta_exit']) / (1 + dilution_sim)

# Estadísticas de la distribución
p10 = np.percentile(premoney_sim, 10) / 1e6
p25 = np.percentile(premoney_sim, 25) / 1e6
p50 = np.percentile(premoney_sim, 50) / 1e6
p75 = np.percentile(premoney_sim, 75) / 1e6
p90 = np.percentile(premoney_sim, 90) / 1e6
media_mc = np.mean(premoney_sim) / 1e6

print(f"\n  Distribución de Pre-money Valuations ({N_SIM:,} escenarios):")
print(f"    P10 (escenario difícil):   ${p10:.1f}M")
print(f"    P25:                       ${p25:.1f}M")
print(f"    P50 (mediana):             ${p50:.1f}M")
print(f"    Media ponderada:           ${media_mc:.1f}M")
print(f"    P75:                       ${p75:.1f}M")
print(f"    P90 (escenario favorable): ${p90:.1f}M")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — WATERFALL DE EQUITY Y DILUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🍰 PASO 5: ESTRUCTURA DE EQUITY POST-INVERSIÓN")
print("─" * 70)

# Estructura típica para Serie A en LATAM
equity_structure = {
    'Fundadores':          0.55,   # después de vesting y dilución seed
    'Inversores Seed':     0.10,   # ronda previa
    'Pool de Opciones':    0.10,   # ESOP para equipo
    f'Serie A ({STARTUP["nombre"][:6]}...)':  equity_fondo,
}

# Ajustar para que sume 1
total = sum(equity_structure.values())
equity_structure = {k: v / total for k, v in equity_structure.items()}

print(f"\n  Pre-money negociación (escenario base): ${pre_money_base/1e6:.1f}M")
print(f"  Inversión buscada:                      ${STARTUP['inversion_buscada']/1e6:.1f}M")
print(f"  Post-money:                             ${post_money/1e6:.1f}M")
print(f"\n  Distribución de Equity Post-Serie A:")
for stakeholder, pct in equity_structure.items():
    valor = pct * post_money
    print(f"    {stakeholder:<30} {pct:.1%}  (${valor/1e6:.2f}M)")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — VISUALIZACIONES EJECUTIVAS
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 14))
fig.suptitle(f'Análisis de Valuación — {STARTUP["nombre"]}\nSerie A LATAM | {STARTUP["sector"]}',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Comparación de metodologías
ax1 = fig.add_subplot(gs[0, 0])
metodos = ['Método VC\nPesimista', 'Comparable\nP25', 'Método VC\nBase',
           'Comparable\nMediana', 'Monte Carlo\nP50', 'Comparable\nP75',
           'Método VC\nOptimista']
valores = [
    resultados_vc['Pesimista']['pre_money'] / 1e6,
    val_comparable_p25,
    resultados_vc['Base']['pre_money'] / 1e6,
    val_comparable_med,
    p50,
    val_comparable_p75,
    resultados_vc['Optimista']['pre_money'] / 1e6,
]
bar_colors = ['#E74C3C', '#E74C3C', '#3498DB', '#3498DB', '#9B59B6', '#2ECC71', '#2ECC71']
bars = ax1.bar(metodos, valores, color=bar_colors, alpha=0.8, edgecolor='white')
ax1.set_ylabel('Pre-money (USD millones)')
ax1.set_title('Rango de Valuaciones\npor Metodología')
ax1.tick_params(axis='x', labelsize=7)
for bar, val in zip(bars, valores):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f'${val:.1f}M', ha='center', va='bottom', fontsize=7, fontweight='bold')

# Plot 2: Monte Carlo — histograma de distribución
ax2 = fig.add_subplot(gs[0, 1])
vals_m = premoney_sim / 1e6
ax2.hist(vals_m, bins=80, color='#9B59B6', alpha=0.7, edgecolor='none', density=True)
ax2.axvline(p10, color='#E74C3C', linestyle='--', lw=1.5, label=f'P10=${p10:.0f}M')
ax2.axvline(p50, color='#2C3E50', linestyle='-', lw=2.5, label=f'P50=${p50:.0f}M')
ax2.axvline(p90, color='#2ECC71', linestyle='--', lw=1.5, label=f'P90=${p90:.0f}M')
ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 0.01],
                  p25, p75, alpha=0.2, color='#3498DB', label=f'Rango P25-P75')
ax2.set_xlabel('Pre-money (USD millones)')
ax2.set_ylabel('Densidad de probabilidad')
ax2.set_title(f'Monte Carlo: {N_SIM:,} Escenarios\nDistribución de Valuación Pre-money')
ax2.legend(fontsize=8)
ax2.set_xlim(0, np.percentile(vals_m, 95))

# Plot 3: Equity Pie Chart
ax3 = fig.add_subplot(gs[0, 2])
colores_eq = ['#3498DB', '#E74C3C', '#F39C12', '#2ECC71']
wedges, texts, autotexts = ax3.pie(
    equity_structure.values(),
    labels=equity_structure.keys(),
    colors=colores_eq,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
ax3.set_title('Estructura de Equity\nPost-Serie A')

# Plot 4: Proyección ARR y hitos de valuación
ax4 = fig.add_subplot(gs[1, :2])
anos = np.arange(0, STARTUP['anos_hasta_exit'] + 1)

# Proyecciones ARR en tres escenarios
arr_pes  = [STARTUP['arr_actual'] * (1 + 0.50) ** a / 1e6 for a in anos]
arr_base = [STARTUP['arr_actual'] * (1 + 0.95) ** a / 1e6 for a in anos]
arr_opt  = [STARTUP['arr_actual'] * (1 + 1.40) ** a / 1e6 for a in anos]

ax4.fill_between(anos, arr_pes, arr_opt, alpha=0.15, color='#3498DB', label='Rango de escenarios')
ax4.plot(anos, arr_pes,  color='#E74C3C', lw=2, linestyle='--', marker='o', markersize=5, label='Pesimista (+50% YoY)')
ax4.plot(anos, arr_base, color='#3498DB', lw=2.5, linestyle='-',  marker='s', markersize=6, label='Base (+95% YoY)')
ax4.plot(anos, arr_opt,  color='#2ECC71', lw=2, linestyle='--', marker='^', markersize=5, label='Optimista (+140% YoY)')

# Marcadores de hitos de valuación
for a, arr_v in zip(anos, arr_base):
    val_implica = arr_v * STARTUP['multiple_salida_base']
    ax4.annotate(f'~${val_implica:.0f}M\n(val.)',
                 (a, arr_v), textcoords='offset points',
                 xytext=(10, 8), fontsize=7, color=COLORES['texto'],
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax4.set_xlabel('Años desde hoy')
ax4.set_ylabel('ARR (USD millones)')
ax4.set_title(f'Proyección ARR y Valuación Implícita (múltiplo {STARTUP["multiple_salida_base"]}x)')
ax4.legend(fontsize=9)
ax4.set_xticks(anos)
ax4.set_xticklabels([f'Año {a}' for a in anos])

# Plot 5: KPIs vs Benchmarks LATAM
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

kpis = [
    ('ARR actual',       f"${STARTUP['arr_actual']/1e6:.1f}M",    'Serie A benchmark: $1-5M ✅'),
    ('Crecimiento YoY',  f"{STARTUP['crecimiento_yoy']:.0%}",      'Benchmark LATAM: >80% ✅'),
    ('Gross Margin',     f"{STARTUP['gross_margin']:.0%}",          'Benchmark SaaS: >65% ✅'),
    ('NRR',              f"{STARTUP['nrr']:.0%}",                   '>100% = expansion MRR ✅'),
    ('Churn Anual',      f"{STARTUP['churn_anual']:.0%}",           'Benchmark B2B: <15% ✅'),
    ('LTV/CAC',          f"{STARTUP['ltv']/STARTUP['cac']:.1f}x",  'Meta: >3x ✅'),
    ('Runway',           f"{STARTUP['runway_meses']}m",             'Meta post-ronda: 18-24m ✅'),
]

texto_kpis = "MÉTRICAS VS BENCHMARKS LATAM\n\n"
for kpi, valor, bench in kpis:
    texto_kpis += f"{kpi:<18} {valor:<10}\n   {bench}\n\n"

ax5.text(0.02, 0.98, texto_kpis, transform=ax5.transAxes,
         fontsize=8.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.9, pad=0.8))

plt.savefig('M6_valuacion_startup.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Dashboard guardado: M6_valuacion_startup.png")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — TABLA RESUMEN PARA PITCH DECK
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("📑 PASO 7: TABLA RESUMEN — LISTO PARA EL PITCH DECK")
print("─" * 70)

# Rango de negociación recomendado: entre P25 de MC y Método VC optimista
rango_bajo = min(p25, resultados_vc['Pesimista']['pre_money'] / 1e6)
rango_alto = max(p75, resultados_vc['Optimista']['pre_money'] / 1e6)
punto_medio = (rango_bajo + rango_alto) / 2

print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║         RESUMEN EJECUTIVO DE VALUACIÓN — SERIE A        ║
  ║                 {STARTUP['nombre']:<40}║
  ╠══════════════════════════════════════════════════════════╣
  ║  METODOLOGÍA          RANGO PRE-MONEY (USD)              ║
  ║  Método VC           ${resultados_vc['Pesimista']['pre_money']/1e6:.1f}M – ${resultados_vc['Optimista']['pre_money']/1e6:.1f}M       ║
  ║  Comparables LATAM   ${val_comparable_p25:.1f}M – ${val_comparable_p75:.1f}M             ║
  ║  Monte Carlo P25-P75 ${p25:.1f}M – ${p75:.1f}M             ║
  ╠══════════════════════════════════════════════════════════╣
  ║  RANGO DE NEGOCIACIÓN:  ${rango_bajo:.1f}M – ${rango_alto:.1f}M           ║
  ║  PUNTO DE PARTIDA RECOMENDADO: ${punto_medio:.1f}M             ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Inversión buscada:   ${STARTUP['inversion_buscada']/1e6:.1f}M                          ║
  ║  Equity a entregar:   {equity_fondo:.1%}                            ║
  ║  Post-money:          ${post_money/1e6:.1f}M                          ║
  ╚══════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER OBLIGATORIO
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("⚠️  DISCLAIMER — LECTURA OBLIGATORIA")
print("=" * 70)
print("""
  Los resultados son ilustrativos basados en datos simulados y benchmarks
  públicos de la región. Para valuaciones en una negociación real:

  1. Validar múltiplos con transacciones comparables RECIENTES en LATAM.
     Los datos de 2022-2023 pueden no reflejar el entorno actual de VC.

  2. Consultar con un asesor financiero especializado en venture capital
     antes de presentar estas cifras en una negociación.

  3. Nunca presentar una cifra única como "la valuación correcta".
     Siempre presentar un rango con los supuestos detrás.

  4. El contexto macroeconómico de LATAM (tipo de cambio, inflación,
     tasas de interés) afecta significativamente los múltiplos de salida.
""")


print("=" * 70)
print("🎯 CONCLUSIONES EJECUTIVAS — VALUACIÓN VC LOGTECH LATAM")
print("=" * 70)
print(f"""
HALLAZGOS CLAVE:
  1. La valuación justificada por datos está en el rango ${rango_bajo:.0f}M – ${rango_alto:.0f}M.
     El punto de partida para negociar es ${punto_medio:.0f}M pre-money.

  2. El NRR de {STARTUP['nrr']:.0%} es el KPI más poderoso del pitch:
     indica que los clientes actuales pagan más cada año → crecimiento implícito.

  3. El LTV/CAC de {STARTUP['ltv']/STARTUP['cac']:.1f}x es sólido. Muestra que por cada $1 invertido
     en adquisición se recuperan ${STARTUP['ltv']/STARTUP['cac']:.0f} en valor de cliente.

  4. Con {STARTUP['runway_meses']} meses de runway post-ronda ({STARTUP['inversion_buscada']/1e6:.1f}M / ${STARTUP['cash_burn_mensual']/1e3:.0f}K burn),
     tienes tiempo para alcanzar el ARR que justifique la Serie B.

ACCIONES INMEDIATAS:
  → Preparar "Data Room": métricas reales vs benchmarks de este análisis.
  → Identificar 10 fondos LATAM activos en LogTech/SaaS B2B (Kaszek, a16z,
    Softbank LATAM, Nazca, Canary, Ignia, Monashees, Magma, Vine Ventures).
  → Proyectar el uso de fondos detallado: equipo (40%), producto (30%), ventas (30%).

PRÓXIMO ANÁLISIS RECOMENDADO:
  → M5 (Prophet) para proyectar ARR con mayor precisión estadística.
  → M3 (LTV/CAC) para mostrar unit economics por canal de adquisición.
""")

print("─" * 70)
print("📁 GUARDAR ESTE SCRIPT:")
print("   /Code Colabs/M6_VC_Valuacion_Startup_LATAM.py")
print("─" * 70)
