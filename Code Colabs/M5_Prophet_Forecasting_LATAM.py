"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÓDULO 5 — PROPHET: DEMAND FORECASTING                                   ║
║   Business Analytics para Startups LATAM                                   ║
║   Caso: AgriTech B2B de insumos agrícolas con estacionalidad regional      ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS PREVIOS:
    pip install prophet pandas numpy matplotlib seaborn

    En Google Colab: !pip install prophet

DATOS QUE NECESITAS:
    - Serie temporal diaria o semanal de ventas/pedidos/usuarios
    - Mínimo recomendado: 2 ciclos completos de tu estacionalidad principal
      (si es anual → mínimo 2 años; si es semanal → mínimo 2-3 meses)
    - Columnas requeridas por Prophet: 'ds' (fecha) y 'y' (valor a pronosticar)

CASO DE USO:
    AgriTech similar a Agrofy (Argentina/Brasil), Kubo Financiero agro (México),
    o Agrosmart (Brasil). Objetivo: pronosticar demanda de insumos agrícolas
    para optimizar inventario y presupuesto de compras con 6 meses de anticipación.

    La estacionalidad agrícola en LATAM tiene picos en:
    - Brasil/Argentina: siembra soya/maíz (sep-dic) y cosecha (feb-abr)
    - México: ciclo primavera-verano (abr-ago) y otoño-invierno (oct-feb)
    - Colombia/Perú: sin estación invernal marcada, ciclos de lluvia

ESTRUCTURA:
    1. Generación de serie temporal con estacionalidad agrícola LATAM
    2. Análisis visual de la serie histórica
    3. Entrenamiento del modelo Prophet
    4. Pronóstico a 6 meses con intervalos de confianza
    5. Descomposición de tendencia + estacionalidad + festivos
    6. Evaluación del modelo (MAPE, MAE)
    7. Escenarios de planning (optimista, base, pesimista)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Prophet — instalación: pip install prophet
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_OK = True
except ImportError:
    print("⚠️  Prophet no instalado. Ejecuta: pip install prophet")
    print("     En Google Colab: !pip install prophet")
    PROPHET_OK = False

# ─── CONFIGURACIÓN VISUAL ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize': (14, 6), 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
np.random.seed(42)

print("=" * 70)
print("📦 MÓDULO 5: PROPHET — DEMAND FORECASTING AGRITECH LATAM")
print("   Similar a: Agrofy, Kubo Agro, Agrosmart, Granular")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — SERIE TEMPORAL CON ESTACIONALIDAD AGRÍCOLA
# ══════════════════════════════════════════════════════════════════════════════

# Serie semanal: 3 años históricos (suficiente para capturar 3 ciclos anuales)
FECHA_INICIO = pd.Timestamp('2023-01-01')
FECHA_FIN    = pd.Timestamp('2026-03-31')
FECHAS       = pd.date_range(start=FECHA_INICIO, end=FECHA_FIN, freq='W')

n = len(FECHAS)

# Tendencia de crecimiento (AgriTech LATAM creció ~25-30% anual en 2022-2025)
tendencia = np.linspace(80, 180, n)  # ventas base USD miles

# Estacionalidad anual agrícola: pico en Q1 (cosecha) y Q3/Q4 (siembra)
# Modelo: suma de componentes sinusoidales para capturar dos picos por año
semanas = np.arange(n)
estacionalidad = (
    30 * np.sin(2 * np.pi * semanas / 52)           # ciclo anual principal
    + 15 * np.sin(4 * np.pi * semanas / 52 + 0.5)   # ciclo semestral (dos temporadas)
    + 8  * np.sin(2 * np.pi * semanas / 13)          # ciclo trimestral (pagos de cosecha)
)

# Ruido aleatorio (variabilidad real: clima, logística, tipo de cambio LATAM)
ruido = np.random.normal(0, 12, n)

# Eventos especiales: ferias agrícolas (Expo Agroalimentaria México = noviembre)
eventos = np.zeros(n)
for i, fecha in enumerate(FECHAS):
    if fecha.month == 11 and fecha.week in [45, 46]:  # Expo Agro México
        eventos[i] = 35
    if fecha.month == 3 and fecha.week in [10, 11]:   # Agroactiva Argentina
        eventos[i] = 25
    if fecha.month == 8 and fecha.week in [32, 33]:   # Agrishow Brasil
        eventos[i] = 20

# Combinar componentes
ventas = (tendencia + estacionalidad + ruido + eventos).clip(min=5)

df_ts = pd.DataFrame({'ds': FECHAS, 'y': ventas.round(2)})
df_ts['y_smoothed'] = df_ts['y'].rolling(4, center=True).mean()  # media móvil 4 semanas

print(f"\n✅ Serie temporal generada: {len(df_ts)} semanas ({FECHA_INICIO.date()} → {FECHA_FIN.date()})")
print(f"   Ventas promedio: ${df_ts['y'].mean():.1f}K USD/semana")
print(f"   Ventas máximas:  ${df_ts['y'].max():.1f}K USD/semana")
print(f"   Ventas mínimas:  ${df_ts['y'].min():.1f}K USD/semana")

# Split: últimas 12 semanas como set de validación
df_train = df_ts[df_ts['ds'] < '2026-01-01'].copy()
df_test  = df_ts[df_ts['ds'] >= '2026-01-01'].copy()
print(f"\n   Train: {len(df_train)} semanas | Test: {len(df_test)} semanas")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — ANÁLISIS VISUAL DE LA SERIE HISTÓRICA
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Análisis de Serie Temporal — Ventas AgriTech LATAM (USD miles/semana)',
             fontsize=14, fontweight='bold')

# Serie completa con tendencia
ax1 = axes[0]
ax1.fill_between(df_ts['ds'], df_ts['y'], alpha=0.3, color='#2ECC71')
ax1.plot(df_ts['ds'], df_ts['y'], alpha=0.6, color='#27AE60', lw=1, label='Ventas semanales')
ax1.plot(df_ts['ds'], df_ts['y_smoothed'], color='#E74C3C', lw=2.5, label='Media móvil 4 sem.')
ax1.axvline(pd.Timestamp('2026-01-01'), color='navy', linestyle='--', lw=2, label='Inicio período test')
ax1.set_ylabel('Ventas (USD miles)')
ax1.set_title('Serie Histórica de Ventas con Tendencia')
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

# Descomposición manual: tendencia vs ciclo anual
ax2 = axes[1]
# Ventas por semana del año (promedio): muestra la estacionalidad
df_ts['semana_anio'] = df_ts['ds'].dt.isocalendar().week.astype(int)
estac_prom = df_ts.groupby('semana_anio')['y'].mean()
ax2.bar(estac_prom.index, estac_prom.values, color='#3498DB', alpha=0.7, edgecolor='white')
ax2.set_xlabel('Semana del Año')
ax2.set_ylabel('Ventas Promedio (USD miles)')
ax2.set_title('Patrón de Estacionalidad Semanal (promedio histórico)')
ax2.axhline(estac_prom.mean(), color='red', linestyle='--', lw=1.5,
            label=f'Promedio global: ${estac_prom.mean():.1f}K')
ax2.legend()

plt.tight_layout()
plt.savefig('M5_serie_historica.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: M5_serie_historica.png")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — ENTRENAMIENTO PROPHET
# ══════════════════════════════════════════════════════════════════════════════

if not PROPHET_OK:
    print("\n⚠️  Saltando entrenamiento: instala Prophet primero.")
else:
    print("\n" + "─" * 70)
    print("🚀 PASO 3: ENTRENAMIENTO PROPHET")
    print("─" * 70)

    # Configuración del modelo:
    # - changepoint_prior_scale: flexibilidad de la tendencia (0.05=rígida, 0.5=muy flexible)
    #   Para AgriTech con crecimiento sostenido, usamos 0.15 (moderado)
    # - seasonality_prior_scale: fuerza de la estacionalidad (10=fuerte para agro)
    # - interval_width: 80% de confianza (más estrecho = más útil para planning)

    modelo = Prophet(
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        interval_width=0.80,
        yearly_seasonality=True,
        weekly_seasonality=False,  # sin efecto día de la semana en datos semanales
        daily_seasonality=False,
    )

    # Estacionalidad agrícola personalizada (ciclo semestral + trimestral)
    modelo.add_seasonality(name='semestral', period=26, fourier_order=5)
    modelo.add_seasonality(name='trimestral', period=13, fourier_order=3)

    # Festivos agrícolas LATAM (eventos que generan spikes de demanda)
    festivos_agro = pd.DataFrame({
        'holiday': ['expo_agro_mx'] * 3 + ['agroactiva_arg'] * 3 + ['agrishow_br'] * 3,
        'ds': pd.to_datetime([
            '2023-11-06', '2024-11-04', '2025-11-03',
            '2023-03-12', '2024-03-10', '2025-03-09',
            '2023-08-07', '2024-08-05', '2025-08-04'
        ]),
        'lower_window': [0] * 9,
        'upper_window': [6] * 9,
    })
    modelo.add_country_holidays(country_name='MX')

    modelo.fit(df_train)
    print("✅ Modelo entrenado con datos históricos.")


    # ══════════════════════════════════════════════════════════════════════════
    # PASO 4 — PRONÓSTICO A 6 MESES
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 70)
    print("🔮 PASO 4: PRONÓSTICO A 6 MESES")
    print("─" * 70)

    # Crear dataframe futuro: 26 semanas hacia adelante (6 meses)
    future = modelo.make_future_dataframe(periods=26, freq='W')
    forecast = modelo.predict(future)

    # Filtrar solo el período futuro (sin histórico)
    forecast_futuro = forecast[forecast['ds'] > df_train['ds'].max()].copy()

    print(f"\n  Pronóstico {forecast_futuro['ds'].min().date()} → {forecast_futuro['ds'].max().date()}")
    print(f"  Ventas proyectadas promedio: ${forecast_futuro['yhat'].mean():.1f}K USD/sem")
    print(f"  Rango 80% confianza: [${forecast_futuro['yhat_lower'].mean():.1f}K, ${forecast_futuro['yhat_upper'].mean():.1f}K]")
    print(f"\n  Ventas proyectadas próximos 6 meses (total):")
    print(f"  Escenario Base:      ${forecast_futuro['yhat'].sum():.0f}K USD")
    print(f"  Escenario Pesimista: ${forecast_futuro['yhat_lower'].sum():.0f}K USD")
    print(f"  Escenario Optimista: ${forecast_futuro['yhat_upper'].sum():.0f}K USD")


    # ══════════════════════════════════════════════════════════════════════════
    # PASO 5 — VISUALIZACIÓN DEL PRONÓSTICO
    # ══════════════════════════════════════════════════════════════════════════

    fig, ax = plt.subplots(figsize=(18, 8))

    # Histórico
    ax.fill_between(df_train['ds'], df_train['y'], alpha=0.2, color='#3498DB')
    ax.plot(df_train['ds'], df_train['y'], color='#2980B9', lw=1.2, alpha=0.8, label='Histórico real')

    # Test (si hay)
    if len(df_test) > 0:
        ax.scatter(df_test['ds'], df_test['y'], color='#E74C3C', s=30, zorder=5, label='Datos test')

    # Pronóstico: período de entrenamiento
    forecast_hist = forecast[forecast['ds'] <= df_train['ds'].max()]
    ax.plot(forecast_hist['ds'], forecast_hist['yhat'], color='#27AE60', lw=1.5,
            linestyle='--', alpha=0.6, label='Ajuste del modelo')

    # Pronóstico: futuro con bandas de confianza
    ax.fill_between(forecast_futuro['ds'],
                    forecast_futuro['yhat_lower'],
                    forecast_futuro['yhat_upper'],
                    alpha=0.3, color='#F39C12', label='Banda de confianza 80%')
    ax.plot(forecast_futuro['ds'], forecast_futuro['yhat'],
            color='#E67E22', lw=3, label='Pronóstico base', zorder=4)
    ax.plot(forecast_futuro['ds'], forecast_futuro['yhat_lower'],
            color='#E74C3C', lw=1.5, linestyle=':', label='Escenario pesimista', zorder=4)
    ax.plot(forecast_futuro['ds'], forecast_futuro['yhat_upper'],
            color='#2ECC71', lw=1.5, linestyle=':', label='Escenario optimista', zorder=4)

    ax.axvline(df_train['ds'].max(), color='black', linestyle='--', lw=2, label='Hoy (inicio forecast)')

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas (USD miles)')
    ax.set_title('Pronóstico de Demanda — AgriTech LATAM\n(6 meses hacia adelante con intervalos de confianza)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    plt.savefig('M5_forecast_6meses.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Forecast guardado: M5_forecast_6meses.png")

    # Descomposición de componentes (el "por qué" de las predicciones)
    fig_comp = modelo.plot_components(forecast)
    fig_comp.suptitle('Descomposición del Modelo: Tendencia + Estacionalidades',
                      fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('M5_componentes_modelo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Componentes guardados: M5_componentes_modelo.png")


    # ══════════════════════════════════════════════════════════════════════════
    # PASO 6 — EVALUACIÓN DEL MODELO (MAPE)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 70)
    print("📊 PASO 6: EVALUACIÓN DEL MODELO")
    print("─" * 70)

    if len(df_test) > 0:
        # Merge pronóstico con valores reales del test set
        eval_df = df_test.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left'
        ).dropna(subset=['yhat'])

        if len(eval_df) > 0:
            eval_df['error_abs']    = np.abs(eval_df['y'] - eval_df['yhat'])
            eval_df['error_pct']    = eval_df['error_abs'] / eval_df['y'] * 100
            eval_df['dentro_banda'] = ((eval_df['y'] >= eval_df['yhat_lower']) &
                                        (eval_df['y'] <= eval_df['yhat_upper']))

            mape = eval_df['error_pct'].mean()
            mae  = eval_df['error_abs'].mean()
            pct_dentro = eval_df['dentro_banda'].mean() * 100

            print(f"\n  MAPE:  {mape:.1f}%  (error porcentual promedio)")
            print(f"  MAE:   {mae:.1f}K USD/semana  (error absoluto promedio)")
            print(f"  Datos dentro de la banda 80%: {pct_dentro:.0f}% (esperado: ~80%)")
            print(f"\n  Interpretación del MAPE:")
            print(f"    < 10% → excelente para planning de inventario")
            print(f"    10-20% → bueno, dentro del rango aceptable")
            print(f"    20-30% → aceptable, considerar más datos o features")
            print(f"    > 30%  → revisar calidad de datos o cambios estructurales")
        else:
            print("  (período de test sin superposición con el forecast)")


    # ══════════════════════════════════════════════════════════════════════════
    # PASO 7 — TABLA DE PLANNING POR MES
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 70)
    print("📅 PASO 7: TABLA DE PLANNING MENSUAL")
    print("─" * 70)

    forecast_futuro['mes'] = forecast_futuro['ds'].dt.to_period('M').astype(str)
    planning = forecast_futuro.groupby('mes').agg(
        ventas_base       = ('yhat',       'sum'),
        ventas_pesimista  = ('yhat_lower', 'sum'),
        ventas_optimista  = ('yhat_upper', 'sum'),
    ).round(1)

    # Presupuesto de compras: 60% del ingreso proyectado (costo de insumos AgriTech)
    COGS_PCT = 0.60
    planning['compras_recomendadas_kUSD'] = (planning['ventas_base'] * COGS_PCT).round(1)

    print("\nPlan de Ventas y Compras — Próximos 6 Meses (USD miles):")
    print(planning.to_string())
    planning.to_csv('M5_plan_ventas_6meses.csv')
    print("\n✅ Plan exportado: M5_plan_ventas_6meses.csv")


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🎯 CONCLUSIONES EJECUTIVAS — DEMAND FORECASTING AGRITECH LATAM")
print("=" * 70)
print("""
HALLAZGOS CLAVE:
  1. La demanda de insumos agrícolas tiene DOS picos anuales claros en LATAM:
     - Siembra (Q3-Q4): aumento del 30-40% sobre la media
     - Cosecha (Q1-Q2): segundo pico con compras de post-cosecha
     Anticipar 6-8 semanas antes para optimizar inventario.

  2. El modelo captura tendencia de crecimiento (~25% anual).
     Si este crecimiento se desacelera, el modelo lo detectará en 2-3 meses.

  3. Los festivos agrícolas (Expo Agro, AgroActiva, Agrishow) generan spikes
     de ventas del 20-35% en esa semana. Planear inventario y equipo de ventas.

  4. La banda de confianza del 80% define el rango de planning:
     - Compras mínimas: escenario pesimista
     - Producción/capacidad máxima: escenario optimista

ACCIONES INMEDIATAS:
  → Compartir la tabla de planning con el equipo de supply chain.
  → Configurar alertas: si ventas reales caen >15% debajo del pronóstico 2 semanas
    seguidas, activar protocolo de revisión de demanda.
  → Actualizar el modelo mensualmente con nuevos datos.

PRÓXIMO MÓDULO RECOMENDADO:
  Para calcular el valor de tu empresa con esta proyección → M6 (Valuación VC).
  Para segmentar clientes que impulsan los picos de temporada → M4 (KMeans).
""")

print("─" * 70)
print("📁 GUARDAR ESTE SCRIPT:")
print("   /Code Colabs/M5_Prophet_Forecasting_LATAM.py")
print("─" * 70)
