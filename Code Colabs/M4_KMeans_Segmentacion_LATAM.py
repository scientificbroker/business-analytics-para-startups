"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÓDULO 4 — KMEANS: SEGMENTACIÓN DE CLIENTES                              ║
║   Business Analytics para Startups LATAM                                   ║
║   Caso: Marketplace de delivery de alimentos (B2C)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS PREVIOS:
    pip install scikit-learn pandas numpy matplotlib seaborn

DATOS QUE NECESITAS:
    - Historial de transacciones por cliente: fecha, monto, frecuencia
    - Mínimo recomendado: 100+ clientes activos (50 por cluster esperado)
    - Columnas mínimas: cliente_id, fecha_ultima_compra, n_pedidos, monto_total

CASO DE USO:
    Marketplace similar a Rappi, iFood o PedidosYa en mercados LATAM.
    Objetivo: segmentar usuarios por comportamiento de compra (modelo RFM)
    para diseñar estrategias de marketing diferenciadas por perfil.

MODELO RFM (estándar global adaptado a LATAM):
    R (Recency)  = días desde el último pedido
    F (Frequency) = número de pedidos en los últimos 90 días
    M (Monetary)  = gasto total en los últimos 90 días (USD)

ESTRUCTURA:
    1. Generación de datos transaccionales
    2. Cálculo de métricas RFM
    3. Normalización y selección de K óptimo (Elbow + Silhouette)
    4. Entrenamiento KMeans y asignación de segmentos
    5. Perfilado de segmentos con nombres accionables
    6. Visualizaciones y estrategias por segmento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN VISUAL ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
# Colores por segmento — diseñados para ser distintos y accesibles
COLORES_SEGMENTO = {
    0: '#E74C3C',  # Rojo: en riesgo
    1: '#2ECC71',  # Verde: campeones
    2: '#3498DB',  # Azul: potencial
    3: '#F39C12',  # Naranja: recientes/nuevos
    4: '#9B59B6',  # Púrpura: hibernando
}
plt.rcParams.update({'figure.figsize': (13, 7), 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
np.random.seed(42)

print("=" * 70)
print("🔖 MÓDULO 4: KMEANS — SEGMENTACIÓN MARKETPLACE LATAM")
print("   Similar a: Rappi, iFood, PedidosYa, Cornershop")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — DATOS TRANSACCIONALES
# ══════════════════════════════════════════════════════════════════════════════

N_CLIENTES = 1500
FECHA_HOY  = datetime(2026, 4, 8)
VENTANA_RFM_DIAS = 90  # análisis sobre los últimos 90 días

paises    = ['México', 'Brasil', 'Argentina', 'Colombia', 'Chile']
p_pais    = [0.25, 0.30, 0.20, 0.15, 0.10]
ciudades  = ['CDMX', 'São Paulo', 'Buenos Aires', 'Bogotá', 'Santiago',
             'Guadalajara', 'Río de Janeiro', 'Córdoba', 'Medellín', 'Valparaíso']

# Simular 4 perfiles naturales que queremos que el modelo descubra
# Esto replica la realidad observada en marketplaces LATAM
PERFILES = [
    # (nombre, n_usuarios, recency_max, freq_media, monto_medio_ticket)
    ('Campeones',   200, 7,   15, 28),   # compran seguido, gastaron mucho recientemente
    ('Leales',      350, 20,  8,  22),   # buenos clientes, no los más frecuentes
    ('En_riesgo',   400, 65,  5,  18),   # antes compraban, ahora no tanto
    ('Hibernando',  550, 85,  2,  12),   # casi no compran, poco gasto
]

transacciones = []
for perfil, n_users, rec_max, freq_m, ticket_m in PERFILES:
    for i in range(n_users):
        # Recency: días desde último pedido
        recency = max(1, int(np.random.uniform(1, rec_max * 1.5)))
        # Frecuencia: pedidos en ventana de 90 días
        freq = max(1, int(np.random.poisson(freq_m)))
        # Monto: distribución log-normal (pocos pedidos grandes, muchos pequeños)
        monto = round(np.random.lognormal(
            np.log(ticket_m * freq), 0.4
        ), 2)

        transacciones.append({
            'cliente_id':        f"{perfil[:3].upper()}-{i:05d}",
            'pais':              np.random.choice(paises, p=p_pais),
            'ciudad':            np.random.choice(ciudades),
            'recency_dias':      recency,
            'freq_90d':          freq,
            'monto_90d_usd':     monto,
            'ticket_prom_usd':   round(monto / freq, 2),
            'n_categorias':      np.random.randint(1, 7),
            'usar_app_movil':    np.random.choice([1, 0], p=[0.85, 0.15]),
            'usa_descuentos':    np.random.choice([1, 0], p=[0.40, 0.60]),
            'nps_ultimo':        np.random.choice(range(5, 11), p=[0.05,0.05,0.10,0.20,0.30,0.30]),
            'perfil_real':       perfil,  # solo para validación, NO entra al modelo
        })

df = pd.DataFrame(transacciones)
print(f"\n✅ Dataset creado: {len(df)} clientes con métricas RFM")
print(f"   Gasto promedio 90 días: ${df['monto_90d_usd'].mean():.2f} USD")
print(f"   Frecuencia promedio: {df['freq_90d'].mean():.1f} pedidos/90 días\n")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — PREPARACIÓN DE FEATURES RFM
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("📐 PASO 2: FEATURES RFM")
print("─" * 70)

# Variables para el clustering (solo las 3 dimensiones RFM puras)
FEATURES_CLUSTER = ['recency_dias', 'freq_90d', 'monto_90d_usd']

# Normalización: StandardScaler garantiza que las 3 variables tengan igual peso
# sin esto, 'monto' dominaría por tener valores más grandes que 'recency'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES_CLUSTER])

print("Estadísticas RFM antes de normalizar:")
print(df[FEATURES_CLUSTER].describe().round(2).to_string())
print("\n✅ Variables normalizadas con StandardScaler (media=0, std=1)")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — SELECCIÓN DE K ÓPTIMO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🔍 PASO 3: SELECCIÓN DE K ÓPTIMO (ELBOW + SILHOUETTE)")
print("─" * 70)

# Probar K de 2 a 8 y medir inercia y silhouette
K_RANGE = range(2, 9)
inercias, silhouettes = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_, sample_size=500))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Selección del Número Óptimo de Clusters', fontsize=14, fontweight='bold')

# Elbow Method
axes[0].plot(list(K_RANGE), inercias, 'bo-', lw=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inercia (WCSS)')
axes[0].set_title('Método del Codo\n(buscar el "quiebre" de la curva)')
axes[0].set_xticks(list(K_RANGE))

# Silhouette Score
axes[1].plot(list(K_RANGE), silhouettes, 'rs-', lw=2, markersize=8)
mejor_k_idx = silhouettes.index(max(silhouettes))
mejor_k     = list(K_RANGE)[mejor_k_idx]
axes[1].axvline(mejor_k, color='green', linestyle='--', lw=2,
                label=f'K={mejor_k} (mejor silhouette: {max(silhouettes):.3f})')
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score\n(más alto = clusters más compactos y separados)')
axes[1].set_xticks(list(K_RANGE))
axes[1].legend()

plt.tight_layout()
plt.savefig('M4_seleccion_k.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Gráfico guardado: M4_seleccion_k.png")
print(f"\n   K óptimo sugerido por Silhouette: K={mejor_k}")
print(f"   Silhouette Score: {max(silhouettes):.3f}  (>0.50 = estructura clara)")

# Usamos K=4 por interpretación de negocio (4 segmentos claros en marketplaces LATAM)
K_FINAL = 4
print(f"\n   K seleccionado para negocio: K={K_FINAL} (4 segmentos accionables)")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — ENTRENAMIENTO KMEANS Y ASIGNACIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🤖 PASO 4: ENTRENAMIENTO KMEANS")
print("─" * 70)

km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20, max_iter=500)
df['cluster_id'] = km_final.fit_predict(X_scaled)

sil_final = silhouette_score(X_scaled, df['cluster_id'], sample_size=500)
print(f"\n  Silhouette Score final (K={K_FINAL}): {sil_final:.3f}")

# Centroides en escala original para interpretación de negocio
centroides = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=FEATURES_CLUSTER
).round(2)
centroides.index.name = 'cluster_id'

print("\nCentroides en escala original (perfil promedio de cada segmento):")
print(centroides.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — PERFILADO Y NOMENCLATURA DE SEGMENTOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("🏷️  PASO 5: PERFILADO DE SEGMENTOS")
print("─" * 70)

# Asignar nombres de negocio basados en los centroides
# Lógica: cluster con menor recency + mayor freq + mayor monto = Campeones
def asignar_nombre(row):
    """Asigna nombre de segmento según el perfil RFM del centroide."""
    r, f, m = row['recency_dias'], row['freq_90d'], row['monto_90d_usd']
    score_r = 1 / (r + 1)  # menor recency = mejor
    score_f = f
    score_m = m
    score_total = score_r * 100 + score_f * 10 + score_m * 0.5

    # Ordenar por score total: el más alto = Campeones
    return score_total

centroides['score'] = centroides.apply(asignar_nombre, axis=1)
centroides_sorted = centroides.sort_values('score', ascending=False).reset_index()

# Nombres y estrategias por ranking
nombres_segmento = {
    0: ('Campeones',        '🏆', '#2ECC71', 'Recompensar, activar como embajadores, cross-sell premium'),
    1: ('Leales Activos',   '💙', '#3498DB', 'Programa de fidelidad, upsell, encuesta NPS'),
    2: ('En Riesgo',        '⚠️',  '#F39C12', 'Campaña win-back urgente, descuento personalizado'),
    3: ('Hibernando',       '💤', '#E74C3C', 'Email de reactivación, oferta única, evaluar costo'),
}

# Mapeo cluster_id → nombre_negocio
mapa_cluster = {}
for rank, (_, row) in enumerate(centroides_sorted.iterrows()):
    cluster_id = row['cluster_id']
    mapa_cluster[cluster_id] = nombres_segmento[rank]

df['segmento']      = df['cluster_id'].map(lambda x: mapa_cluster[x][0])
df['color_segmento'] = df['cluster_id'].map(lambda x: mapa_cluster[x][2])

# Tabla resumen de segmentos
resumen_seg = df.groupby('segmento').agg(
    n_clientes         = ('cliente_id', 'count'),
    recency_prom       = ('recency_dias', 'mean'),
    freq_prom          = ('freq_90d', 'mean'),
    monto_prom_90d     = ('monto_90d_usd', 'mean'),
    ticket_prom        = ('ticket_prom_usd', 'mean'),
    nps_prom           = ('nps_ultimo', 'mean'),
    usa_descuentos_pct = ('usa_descuentos', 'mean'),
).round(2)

resumen_seg['pct_base'] = (resumen_seg['n_clientes'] / len(df) * 100).round(1)

print("\nPerfil de Segmentos (valores promedio por segmento):")
print(resumen_seg.to_string())

# Estrategias recomendadas
print("\nEstrategias por Segmento:")
for segmento, (nombre, emoji, color, estrategia) in nombres_segmento.items():
    print(f"  {emoji} {nombre:<18} → {estrategia}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

# Colores consistentes con el perfil de segmento
color_map = df['segmento'].map({v[0]: v[2] for v in nombres_segmento.values()})

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Segmentación de Clientes — Marketplace LATAM\nAnálisis RFM con KMeans',
             fontsize=15, fontweight='bold')

# Plot 1: Scatter Recency vs Frequency
ax1 = axes[0, 0]
for seg_name, group in df.groupby('segmento'):
    color = [v[2] for v in nombres_segmento.values() if v[0] == seg_name][0]
    ax1.scatter(group['recency_dias'], group['freq_90d'],
                c=color, alpha=0.5, s=30, label=seg_name, edgecolors='none')
ax1.set_xlabel('Recency (días desde último pedido)')
ax1.set_ylabel('Frecuencia (pedidos/90 días)')
ax1.set_title('Recency vs Frecuencia por Segmento')
ax1.legend(fontsize=9)

# Plot 2: Scatter Frequency vs Monto
ax2 = axes[0, 1]
for seg_name, group in df.groupby('segmento'):
    color = [v[2] for v in nombres_segmento.values() if v[0] == seg_name][0]
    ax2.scatter(group['freq_90d'], group['monto_90d_usd'],
                c=color, alpha=0.5, s=30, label=seg_name, edgecolors='none')
ax2.set_xlabel('Frecuencia (pedidos/90 días)')
ax2.set_ylabel('Gasto en 90 días (USD)')
ax2.set_title('Frecuencia vs Gasto por Segmento')
ax2.legend(fontsize=9)

# Plot 3: Composición de la base
ax3 = axes[1, 0]
seg_counts = resumen_seg['n_clientes'].sort_values(ascending=True)
seg_colors = [df[df['segmento'] == s]['color_segmento'].iloc[0] for s in seg_counts.index]
bars = ax3.barh(seg_counts.index, seg_counts.values, color=seg_colors, edgecolor='white', height=0.6)
ax3.set_xlabel('Número de Clientes')
ax3.set_title(f'Distribución de la Base de Clientes (Total: {len(df):,})')
for bar, val in zip(bars, seg_counts.values):
    pct = val / len(df) * 100
    ax3.text(val + 5, bar.get_y() + bar.get_height()/2,
             f'{val:,} ({pct:.0f}%)', va='center', fontsize=9)

# Plot 4: Revenue por segmento
ax4 = axes[1, 1]
revenue_seg = df.groupby('segmento')['monto_90d_usd'].sum().sort_values(ascending=True)
rev_colors = [df[df['segmento'] == s]['color_segmento'].iloc[0] for s in revenue_seg.index]
bars4 = ax4.barh(revenue_seg.index, revenue_seg.values / 1000,
                 color=rev_colors, edgecolor='white', height=0.6)
ax4.set_xlabel('Revenue Total 90 días (USD miles)')
ax4.set_title('Revenue por Segmento (últimos 90 días)')
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K')
                               if 'mticker' in dir() else plt.FuncFormatter(lambda x, _: f'${x:.0f}K'))
total_rev = revenue_seg.sum()
for bar, val in zip(bars4, revenue_seg.values):
    ax4.text(val/1000 + total_rev/1000*0.005, bar.get_y() + bar.get_height()/2,
             f'${val/1000:.1f}K ({val/total_rev:.0%})', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('M4_segmentacion_kmeans.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Dashboard guardado: M4_segmentacion_kmeans.png")


# Radar Chart: perfil multi-dimensional por segmento
from matplotlib.patches import FancyBboxPatch

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Perfil de Segmentos — Métricas Clave Normalizadas', fontsize=14, fontweight='bold')

# Heatmap de perfiles normalizados (mejor para presentaciones a inversores)
metricas_radar = ['recency_prom', 'freq_prom', 'monto_prom_90d', 'ticket_prom', 'nps_prom']
nombres_radar = ['Recency (inv.)', 'Frecuencia', 'Gasto 90d', 'Ticket Prom.', 'NPS']

heatmap_df = resumen_seg[metricas_radar].copy()
# Invertir recency (menor es mejor)
heatmap_df['recency_prom'] = 1 / (heatmap_df['recency_prom'] + 1)
# Normalizar entre 0 y 1 para comparación visual
for col in heatmap_df.columns:
    rng = heatmap_df[col].max() - heatmap_df[col].min()
    if rng > 0:
        heatmap_df[col] = (heatmap_df[col] - heatmap_df[col].min()) / rng

heatmap_df.columns = nombres_radar
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes2[0], linewidths=1, vmin=0, vmax=1,
            cbar_kws={'label': 'Score normalizado (0=peor, 1=mejor)'})
axes2[0].set_title('Perfil de Cada Segmento\n(verde = fuerte, rojo = débil)')

# Tabla de estrategias
axes2[1].axis('off')
estrategias_texto = "\n\n".join([
    f"{v[1]} {v[0]}\n   {v[3]}"
    for v in nombres_segmento.values()
])
axes2[1].text(0.02, 0.95, "ESTRATEGIAS POR SEGMENTO\n\n" + estrategias_texto,
              transform=axes2[1].transAxes, fontsize=10,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.9))

plt.tight_layout()
plt.savefig('M4_perfiles_segmentos.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Perfiles guardados: M4_perfiles_segmentos.png")

# Exportar lista de clientes con segmento asignado para el CRM
df_export = df[['cliente_id', 'pais', 'ciudad', 'segmento',
                'recency_dias', 'freq_90d', 'monto_90d_usd', 'ticket_prom_usd']].copy()
df_export.to_csv('M4_clientes_segmentados.csv', index=False)
print("✅ Lista exportada: M4_clientes_segmentados.csv (lista para subir al CRM)")


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🎯 CONCLUSIONES EJECUTIVAS — SEGMENTACIÓN MARKETPLACE LATAM")
print("=" * 70)

n_camp  = resumen_seg.loc['Campeones', 'n_clientes'] if 'Campeones' in resumen_seg.index else 0
n_riesg = resumen_seg.loc['En Riesgo', 'n_clientes'] if 'En Riesgo' in resumen_seg.index else 0

print(f"""
HALLAZGOS CLAVE:
  1. Silhouette Score = {sil_final:.3f}:
     {"estructura de clusters clara (>0.50 = excelente)" if sil_final > 0.50 else "estructura aceptable (considera más features si <0.40)"}

  2. Distribución encontrada:
     {resumen_seg[['n_clientes', 'pct_base']].to_string()}

  3. Los Campeones ({n_camp} clientes) generan desproporcionadamente más revenue.
     Invertir en retenerlos y convertirlos en embajadores tiene el mayor ROI.

  4. Los 'En Riesgo' ({n_riesg} clientes) son recuperables con acción inmediata.
     Un descuento del 15-20% en su próximo pedido suele reactivar el 25-35%.

  5. Los 'Hibernando' requieren evaluación costo-beneficio: si su CAC fue alto,
     vale intentar reactivación; si llegaron orgánicos, el costo es bajo.

ACCIONES INMEDIATAS:
  → Subir M4_clientes_segmentados.csv a tu CRM (Hubspot, Salesforce, Braze).
  → Crear campañas de email/push específicas por segmento esta semana.
  → Repetir el análisis cada 30 días para detectar migraciones entre segmentos.

PRÓXIMO MÓDULO RECOMENDADO:
  LTV por segmento → aplica M3 (LTV/CAC) filtrando por cada segmento.
  Proyección de demanda por zona geográfica → M5 (Prophet Forecasting).
""")

print("─" * 70)
print("📁 GUARDAR ESTE SCRIPT:")
print("   /Code Colabs/M4_KMeans_Segmentacion_LATAM.py")
print("─" * 70)

# Import faltante para el formatter del plot
import matplotlib.ticker as mticker
