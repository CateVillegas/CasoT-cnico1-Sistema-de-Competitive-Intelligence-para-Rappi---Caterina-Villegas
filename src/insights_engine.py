import pandas as pd
import numpy as np
import json
import requests
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .data_loader import (load_data, WEEK_COLS_METRICS, WEEK_LABELS,
                           COUNTRY_NAMES, METRIC_DESCRIPTIONS, format_metric_value)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_URL = os.environ.get('GEMINI_URL',
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")

BG_DARK  = '#FFFFFF'   # PDF chart background → white
BG_MID   = '#F8F9FA'   # PDF subplot background
GRID_COL = '#E5E7EB'   # PDF grid lines
TEXT_COL = '#1F2937'   # PDF text
RED      = '#E03000'
ORANGE   = '#FF6B35'
GREEN    = '#16A34A'
YELLOW   = '#D97706'
BLUE     = '#2563EB'

# Dark mode palette (for web display)
D_BG    = '#0F0F1A'
D_MID   = '#1C1C2E'
D_GRID  = '#2A2A3E'
D_TEXT  = '#E8E8F0'
D_RED   = '#FF441F'
D_ORANGE= '#FF6B35'
D_GREEN = '#22C55E'
D_YELLOW= '#F59E0B'


def _colors(dark=False):
    """Return (bg, mid, grid, text, red, orange, green, yellow) for dark or light mode."""
    if dark:
        return D_BG, D_MID, D_GRID, D_TEXT, D_RED, D_ORANGE, D_GREEN, D_YELLOW
    return BG_DARK, BG_MID, GRID_COL, TEXT_COL, RED, ORANGE, GREEN, YELLOW


def _fig_to_b64(fig, dark=False):
    bg = D_BG if dark else BG_DARK
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=bg, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

ALL_KEY_METRICS = [
    'Perfect Orders', 'Lead Penetration', 'Gross Profit UE',
    'Non-Pro PTC > OP', 'Restaurants SS > ATC CVR',
    '% PRO Users Who Breakeven', 'MLTV Top Verticals Adoption',
    'Turbo Adoption', '% Restaurants Sessions With Optimal Assortment',
]

# ──────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────

def _pct_change_safe(new_val, old_val):
    if pd.isna(new_val) or pd.isna(old_val):
        return None
    if abs(old_val) < 0.001:
        return None
    return (new_val - old_val) / abs(old_val)


def detect_anomalies(df, threshold=0.10, abs_threshold_numeric=0.5, orders_df=None):
    from .data_loader import NUMERIC_METRICS
    df = df.dropna(subset=['L0W_ROLL', 'L1W_ROLL']).copy()
    df = df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE', 'METRIC'])
    # Build orders lookup for severity weighting
    orders_lookup = {}
    if orders_df is not None:
        for _, orow in orders_df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE']).iterrows():
            orders_lookup[(orow['COUNTRY'], orow['CITY'], orow['ZONE'])] = orow.get('L0W_ROLL', 0)
    max_orders = max(orders_lookup.values()) if orders_lookup else 1
    results = []
    for _, row in df.iterrows():
        metric = row['METRIC']
        cur, prev = row['L0W_ROLL'], row['L1W_ROLL']
        abs_change = cur - prev
        if metric in NUMERIC_METRICS:
            if abs(abs_change) < abs_threshold_numeric:
                continue
            pct = _pct_change_safe(cur, prev)
            if pct is None:
                pct = abs_change
            display = f"{abs_change:+.2f}"
        else:
            pct = _pct_change_safe(cur, prev)
            if pct is None or abs(pct) < threshold:
                continue
            display = f"{pct*100:+.1f}%"
        zone_orders = orders_lookup.get((row['COUNTRY'], row['CITY'], row['ZONE']), 0)
        if pd.isna(zone_orders):
            zone_orders = 0
        # Severity = magnitude of change * volume weight (0-1)
        vol_weight = (zone_orders / max_orders) if max_orders > 0 else 0
        severity = abs(pct) * (0.4 + 0.6 * vol_weight)  # min 40% base, up to 100% for top volume
        results.append({
            'zone': row['ZONE'], 'city': row['CITY'],
            'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
            'country_code': row['COUNTRY'],
            'zone_type': row.get('ZONE_TYPE', ''),
            'metric': metric,
            'current': cur, 'previous': prev,
            'abs_change': abs_change,
            'change_pct': pct,
            'change_display': display,
            'direction': 'mejora' if pct > 0 else 'deterioro',
            'orders': int(zone_orders),
            'severity_score': round(severity, 4),
        })
    results.sort(key=lambda x: x['severity_score'], reverse=True)
    return results[:25]


def detect_consistent_trends(df, n_weeks=3):
    week_cols = WEEK_COLS_METRICS[-4:]
    df = df.dropna(subset=week_cols).copy()
    df = df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE', 'METRIC'])
    declining, improving = [], []
    for _, row in df.iterrows():
        vals = [row[c] for c in week_cols]
        if not all(v is not None and not pd.isna(v) for v in vals):
            continue
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        base = abs(vals[0]) if abs(vals[0]) > 0.001 else 0.001
        total_change = max(min((vals[-1] - vals[0]) / base, 5.0), -5.0)
        entry = {
            'zone': row['ZONE'], 'city': row['CITY'],
            'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
            'zone_type': row.get('ZONE_TYPE', ''),
            'metric': row['METRIC'],
            'values': [round(v, 4) for v in vals],
            'total_change_pct': total_change,
            'weeks': n_weeks,
            'labels': WEEK_LABELS[-4:],
        }
        if all(d < 0 for d in diffs[-n_weeks:]):
            # Add diagnostic context based on metric type and rate of decline
            metric_name = row['METRIC']
            rate = abs(total_change) * 100
            if 'Gross Profit' in metric_name:
                diag = 'Posible causa: incremento en costos logísticos, descuentos excesivos o caída en ticket promedio.'
            elif 'Perfect Orders' in metric_name:
                diag = 'Posible causa: problemas de fulfillment, cancelaciones de merchants o demoras en delivery.'
            elif 'Lead Penetration' in metric_name:
                diag = 'Posible causa: merchants abandonando la plataforma o estancamiento en activación de nuevos leads.'
            elif 'Conversion' in metric_name or 'PTC' in metric_name or 'CVR' in metric_name:
                diag = 'Posible causa: fricción en UX, cambios en métodos de pago o pricing poco competitivo.'
            elif 'Turbo' in metric_name:
                diag = 'Posible causa: problemas de cobertura, tiempos de entrega no competitivos o falta de awareness.'
            elif 'MLTV' in metric_name or 'Verticals' in metric_name:
                diag = 'Posible causa: usuarios concentrando compras en un solo vertical — revisar cross-sell y discovery.'
            elif 'Pro' in metric_name:
                diag = 'Posible causa: valor percibido de Pro en baja — revisar beneficios, pricing de membresía o churn.'
            elif 'Markdowns' in metric_name:
                diag = 'Posible causa: incremento en campañas de descuento sin control — revisar ROI de promociones.'
            else:
                diag = 'Revisar con el equipo local para identificar cambios operacionales recientes en la zona.'
            if rate > 50:
                diag += ' ⚡ Caída acelerada — requiere intervención urgente.'
            entry['diagnostic'] = diag
            declining.append(entry)
        elif all(d > 0 for d in diffs[-n_weeks:]):
            improving.append(entry)
    declining.sort(key=lambda x: x['total_change_pct'])
    improving.sort(key=lambda x: x['total_change_pct'], reverse=True)
    return declining[:12], improving[:6]


def detect_benchmarking_gaps(df):
    gaps = []
    for metric in ALL_KEY_METRICS:
        metric_df = df[df['METRIC'] == metric].dropna(subset=['L0W_ROLL'])
        metric_df = metric_df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE'])
        for (country, zone_type), grp in metric_df.groupby(['COUNTRY', 'ZONE_TYPE']):
            if len(grp) < 4:
                continue
            q75 = grp['L0W_ROLL'].quantile(0.75)
            q25 = grp['L0W_ROLL'].quantile(0.25)
            gap = q75 - q25
            median = grp['L0W_ROLL'].median()
            rel_gap = gap / abs(median) if abs(median) > 0.001 else gap
            if rel_gap < 0.2:
                continue
            top3 = grp.nlargest(3, 'L0W_ROLL')[['ZONE', 'CITY', 'L0W_ROLL']].to_dict('records')
            bot3 = grp.nsmallest(3, 'L0W_ROLL')[['ZONE', 'CITY', 'L0W_ROLL']].to_dict('records')
            gaps.append({
                'metric': metric,
                'country': COUNTRY_NAMES.get(country, country),
                'country_code': country,
                'zone_type': zone_type,
                'gap_abs': gap,
                'gap_rel': rel_gap,
                'median': median,
                'top_zones': top3,
                'bot_zones': bot3,
                'zones_count': len(grp),
            })
    gaps.sort(key=lambda x: x['gap_rel'], reverse=True)
    return gaps[:12]


def detect_correlations(df):
    pivot = df.pivot_table(
        index=['COUNTRY', 'CITY', 'ZONE'], columns='METRIC', values='L0W_ROLL'
    )
    pivot = pivot.dropna(thresh=4)
    correlations = []
    cols = pivot.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            valid = pivot[[cols[i], cols[j]]].dropna()
            if len(valid) < 15:
                continue
            corr = valid[cols[i]].corr(valid[cols[j]])
            if abs(corr) > 0.35:
                correlations.append({
                    'metric_a': cols[i], 'metric_b': cols[j],
                    'correlation': round(corr, 3),
                    'strength': 'Fuerte' if abs(corr) > 0.65 else 'Moderada',
                    'direction': 'positiva' if corr > 0 else 'negativa',
                    'n_zones': len(valid),
                })
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return correlations[:10]


def detect_opportunities(df, orders_df):
    order_current = orders_df[['COUNTRY', 'CITY', 'ZONE', 'L0W_ROLL']].copy()
    order_current = order_current.rename(columns={'L0W_ROLL': 'orders'})
    order_current = order_current.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE'])
    threshold_pct = order_current['orders'].quantile(0.65)
    high_vol = order_current[order_current['orders'] >= threshold_pct]
    opp_metrics = ['Perfect Orders', 'Lead Penetration', 'Non-Pro PTC > OP',
                   'Gross Profit UE', 'Restaurants SS > ATC CVR']
    seen = set()
    opportunities = []
    for metric in opp_metrics:
        mdf = df[df['METRIC'] == metric][['COUNTRY', 'CITY', 'ZONE', 'ZONE_TYPE', 'L0W_ROLL']].copy()
        mdf = mdf.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE'])
        merged = high_vol.merge(mdf, on=['COUNTRY', 'CITY', 'ZONE'])
        merged = merged.dropna(subset=['L0W_ROLL'])
        overall_p50 = df[df['METRIC'] == metric]['L0W_ROLL'].median()
        overall_p25 = df[df['METRIC'] == metric]['L0W_ROLL'].quantile(0.25)
        underperf = merged[merged['L0W_ROLL'] <= overall_p25]
        underperf = underperf.sort_values('orders', ascending=False).head(4)
        for _, row in underperf.iterrows():
            key = (row['ZONE'], row['COUNTRY'], metric)
            if key in seen:
                continue
            seen.add(key)
            gap_pct = (overall_p50 - row['L0W_ROLL']) / abs(overall_p50) if abs(overall_p50) > 0.001 else 0
            # Impact estimation: if metric reaches median, estimate order/revenue impact
            potential_improvement = overall_p50 - row['L0W_ROLL']
            weekly_orders = int(row['orders'])
            # Rough impact: % improvement * orders = potential additional quality/converted orders
            estimated_impact_orders = int(abs(potential_improvement) * weekly_orders) if metric != 'Gross Profit UE' else 0
            estimated_impact_gp = round(potential_improvement * weekly_orders, 1) if metric == 'Gross Profit UE' else 0
            opportunities.append({
                'zone': row['ZONE'], 'city': row['CITY'],
                'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
                'zone_type': row.get('ZONE_TYPE', 'N/A'),
                'metric': metric,
                'metric_value': row['L0W_ROLL'],
                'metric_value_fmt': format_metric_value(metric, row['L0W_ROLL']),
                'benchmark_fmt': format_metric_value(metric, overall_p50),
                'orders': weekly_orders,
                'gap_pct': gap_pct,
                'estimated_impact_orders': estimated_impact_orders,
                'estimated_impact_gp': estimated_impact_gp,
                'potential_improvement': potential_improvement,
            })
    opportunities.sort(key=lambda x: x['orders'], reverse=True)
    return opportunities[:12]


def country_scorecards(df, orders_df):
    order_by_country = orders_df.groupby('COUNTRY')['L0W_ROLL'].sum().to_dict()
    key = ['Perfect Orders', 'Lead Penetration', 'Non-Pro PTC > OP', 'Gross Profit UE', 'Turbo Adoption']
    scorecards = []
    for country in sorted(df['COUNTRY'].unique()):
        row = {'country': COUNTRY_NAMES.get(country, country),
               'country_code': country,
               'total_orders': int(order_by_country.get(country, 0)),
               'metrics': {}}
        cdf = df[df['COUNTRY'] == country]
        for m in key:
            mdf = cdf[cdf['METRIC'] == m]['L0W_ROLL'].dropna()
            prev_mdf = cdf[cdf['METRIC'] == m]['L1W_ROLL'].dropna()
            if len(mdf):
                median_val = mdf.median()
                prev_val = prev_mdf.median() if len(prev_mdf) else None
                wow = _pct_change_safe(median_val, prev_val) if prev_val else None
                row['metrics'][m] = {
                    'value': median_val,
                    'fmt': format_metric_value(m, median_val),
                    'wow': wow,
                    'wow_fmt': f"{wow*100:+.1f}%" if wow is not None else '—',
                }
        scorecards.append(row)
    scorecards.sort(key=lambda x: x['total_orders'], reverse=True)
    return scorecards


def compile_raw_insights():
    metrics_df, orders_df, _ = load_data()
    declining_trends, improving_trends = detect_consistent_trends(metrics_df)
    return {
        'anomalies': detect_anomalies(metrics_df, orders_df=orders_df),
        'declining_trends': declining_trends,
        'improving_trends': improving_trends,
        'benchmarking_gaps': detect_benchmarking_gaps(metrics_df),
        'correlations': detect_correlations(metrics_df),
        'opportunities': detect_opportunities(metrics_df, orders_df),
        'scorecards': country_scorecards(metrics_df, orders_df),
        '_metrics_df': metrics_df,  # kept for chart generation, not serialized
    }


# ──────────────────────────────────────────────────────────────
# CHART GENERATORS
# ──────────────────────────────────────────────────────────────

def chart_anomalies_top(anomalies, dark=False):
    top = [a for a in anomalies if a['direction'] == 'deterioro'][:8]
    if not top:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    labels = [f"{a['zone'][:28]} ({a['country'][:3]})" for a in top]
    vals   = [min(abs(a['change_pct']) * 100, 300) for a in top]
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(top) * 0.52)))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    bars = ax.barh(range(len(labels)), vals, color=red, alpha=0.82, height=0.55)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=text, fontsize=8)
    ax.set_xlabel('Magnitud del cambio WoW', color=text, fontsize=8)
    ax.set_title('Top Deterioros — Semana vs Semana Anterior', color=text, fontsize=10, fontweight='bold', pad=8)
    max_v = max(vals) if vals else 1
    ax.set_xlim(0, max_v * 1.30)
    for bar, a in zip(bars, top):
        ax.text(bar.get_width() + max_v * 0.015, bar.get_y() + bar.get_height() / 2,
                a['change_display'], va='center', color=red, fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text); ax.xaxis.grid(True, color=grid, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True); plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_country_metric(scorecards, metric_name, dark=False):
    items = [(s['country'], s['metrics'][metric_name]) for s in scorecards if metric_name in s['metrics']]
    if not items:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    countries = [i[0][:12] for i in items]
    values    = [i[1]['value'] for i in items]
    fmts      = [i[1]['fmt'] for i in items]
    plot_vals = [v * 100 if (0 < abs(v) <= 1.5 and '%' in format_metric_value(metric_name, v)) else v for v in values]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    colors = [red if i % 2 == 0 else orange for i in range(len(countries))]
    x = np.arange(len(countries))
    bars = ax.bar(x, plot_vals, color=colors, alpha=0.88, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(countries, color=text, fontsize=8, rotation=30, ha='right')
    ax.set_title(f'{metric_name} — Mediana por País', color=text, fontsize=9, fontweight='bold', pad=6)
    max_v = max(abs(v) for v in plot_vals) if plot_vals else 1
    ax.set_ylim(min(min(plot_vals)*1.2, 0) if min(plot_vals) < 0 else 0, max_v * 1.25)
    for bar, fmt in zip(bars, fmts):
        ypos = bar.get_height() + max_v * 0.03 if bar.get_height() >= 0 else bar.get_height() - max_v * 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, fmt,
                ha='center', color=text, fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text); ax.yaxis.grid(True, color=grid, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True); plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_trend_sparklines(trends, title, line_color_name='red', dark=False):
    items = trends[:4]
    if not items:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    color_map = {'red': red, 'green': green, 'orange': orange}
    color = color_map.get(line_color_name, red)
    n = len(items)
    cols = min(n, 2); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    fig.patch.set_facecolor(bg)
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.array(axes).flatten())
    for i, (ax, entry) in enumerate(zip(axes_flat, items)):
        ax.set_facecolor(mid)
        vals = entry['values']
        x = np.arange(len(vals))
        ax.plot(x, vals, color=color, linewidth=2, marker='o', markersize=4,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
        ax.fill_between(x, vals, alpha=0.12, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(entry['labels'], fontsize=6.5, color=text, rotation=20, ha='right')
        zone_label = entry['zone'][:20]
        metric_label = entry['metric'][:22]
        ax.set_title(f"{zone_label}\n{entry['country']} · {metric_label}",
                     fontsize=7.5, color=text, fontweight='bold', pad=3)
        ax.tick_params(colors=text, labelsize=6.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
        ax.yaxis.grid(True, color=grid, linestyle='--', alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, color=text, fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return _fig_to_b64(fig, dark)


def chart_correlations(correlations, dark=False):
    top = correlations[:7]
    if not top:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    labels = [f"{c['metric_a'][:18]}\nvs {c['metric_b'][:18]}" for c in top]
    vals   = [c['correlation'] for c in top]
    colors = [green if v > 0 else red for v in vals]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55)
    ax.axhline(0, color=grid, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=text, fontsize=7, rotation=0, ha='center')
    ax.set_ylabel('Coef. de correlación (r)', color=text, fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title('Correlaciones entre Métricas Operacionales', color=text, fontsize=10, fontweight='bold', pad=8)
    for bar, c in zip(bars, top):
        yoff = 0.05 if bar.get_height() >= 0 else -0.10
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                f"r={c['correlation']:.2f}", ha='center', color=text, fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text); ax.yaxis.grid(True, color=grid, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    legend = [mpatches.Patch(color=green, label='Positiva'), mpatches.Patch(color=red, label='Negativa')]
    ax.legend(handles=legend, facecolor=mid, edgecolor=grid, labelcolor=text, fontsize=8, loc='upper right')
    plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_correlation_heatmap(metrics_df, dark=False):
    """Correlation heatmap matrix — shows all metric pair relationships visually."""
    if metrics_df is None or metrics_df.empty:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    pivot = metrics_df.pivot_table(
        index=['COUNTRY', 'CITY', 'ZONE'], columns='METRIC', values='L0W_ROLL'
    )
    pivot = pivot.dropna(thresh=4)
    if pivot.shape[1] < 3:
        return None
    corr_matrix = pivot.corr()
    # Shorten metric names for readability
    short_names = {
        '% PRO Users Who Breakeven': '% Pro Breakeven',
        '% Restaurants Sessions With Optimal Assortment': '% Optimal Assort.',
        'MLTV Top Verticals Adoption': 'MLTV Adoption',
        'Non-Pro PTC > OP': 'Non-Pro Conv.',
        'Pro Adoption (Last Week Status)': 'Pro Adoption',
        'Restaurants Markdowns / GMV': 'Rest. Markdowns',
        'Restaurants SS > ATC CVR': 'Rest. SS→ATC',
        'Restaurants SST > SS CVR': 'Rest. SST→SS',
        'Retail SST > SS CVR': 'Retail SST→SS',
    }
    labels = [short_names.get(c, c[:18]) for c in corr_matrix.columns]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.75), max(6, n * 0.6)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    # Custom diverging colormap: red (negative) → white → green (positive)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg', [red, '#FFFFFF', green])
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=6.5, color=text, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=6.5, color=text)
    # Annotate cells with correlation values
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                txt_color = text if abs(val) < 0.5 else 'white'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=5.5, color=txt_color, fontweight='bold' if abs(val) > 0.5 else 'normal')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors=text)
    cbar.set_label('Correlación (r)', fontsize=8, color=text)
    ax.set_title('Matriz de Correlación entre Todas las Métricas', color=text,
                 fontsize=10, fontweight='bold', pad=10)
    plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_correlation_scatter(metrics_df, metric_a, metric_b, correlation_value, dark=False):
    """Scatter plot showing the relationship between two correlated metrics."""
    if metrics_df is None or metrics_df.empty:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    pivot = metrics_df.pivot_table(
        index=['COUNTRY', 'CITY', 'ZONE'], columns='METRIC', values='L0W_ROLL'
    )
    if metric_a not in pivot.columns or metric_b not in pivot.columns:
        return None
    valid = pivot[[metric_a, metric_b]].dropna()
    if len(valid) < 10:
        return None
    x = valid[metric_a].values
    y = valid[metric_b].values
    # Scale percentage metrics
    from .data_loader import RATIO_METRICS
    x_plot = x * 100 if metric_a in RATIO_METRICS else x
    y_plot = y * 100 if metric_b in RATIO_METRICS else y
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(mid)
    color = green if correlation_value > 0 else red
    ax.scatter(x_plot, y_plot, c=color, alpha=0.45, s=20, edgecolors='white', linewidths=0.3)
    # Trend line
    z = np.polyfit(x_plot, y_plot, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
    ax.plot(x_line, p(x_line), color=orange, linewidth=2, linestyle='--', alpha=0.8)
    short_a = metric_a[:25]
    short_b = metric_b[:25]
    x_suffix = ' (%)' if metric_a in RATIO_METRICS else ''
    y_suffix = ' (%)' if metric_b in RATIO_METRICS else ''
    ax.set_xlabel(f'{short_a}{x_suffix}', color=text, fontsize=8)
    ax.set_ylabel(f'{short_b}{y_suffix}', color=text, fontsize=8)
    direction = 'positiva' if correlation_value > 0 else 'negativa'
    ax.set_title(f'{short_a} vs {short_b}\nr = {correlation_value:.3f} (correlación {direction})',
                 color=text, fontsize=9, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid)
    ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text, labelsize=7)
    ax.grid(True, color=grid, linestyle='--', alpha=0.3)
    # Add annotation box
    ax.text(0.03, 0.95, f'n = {len(valid)} zonas\nr = {correlation_value:.3f}',
            transform=ax.transAxes, fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=mid, edgecolor=grid, alpha=0.9),
            color=text)
    plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_opportunities(opportunities, dark=False):
    if not opportunities:
        return None
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    top = opportunities[:7]
    labels = [f"{o['zone'][:24]} ({o['country'][:3]})" for o in top]
    orders = [o['orders'] for o in top]
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(top) * 0.52)))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    colors = [orange if i % 2 == 0 else yellow for i in range(len(top))]
    bars = ax.barh(range(len(labels)), orders, color=colors, alpha=0.88, height=0.55)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=text, fontsize=8)
    ax.set_xlabel('Órdenes semanales', color=text, fontsize=8)
    ax.set_title('Zonas de Alto Volumen con Métricas Debajo del Benchmark',
                 color=text, fontsize=10, fontweight='bold', pad=8)
    max_o = max(orders) if orders else 1
    ax.set_xlim(0, max_o * 1.20)
    for bar, o in zip(bars, top):
        ax.text(bar.get_width() + max_o * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{o['orders']:,} órd.", va='center', color=text, fontsize=7.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text); ax.xaxis.grid(True, color=grid, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True); plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_high_priority_perf(metrics_df, dark=False):
    """Avg of key metrics by zone prioritization level."""
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    key_metrics = ['Perfect Orders', 'Non-Pro PTC > OP', 'Turbo Adoption',
                   'MLTV Top Verticals Adoption', 'Pro Adoption (Last Week Status)']
    results = {}
    for prio in ['High Priority', 'Prioritized', 'Not Prioritized']:
        sub = metrics_df[metrics_df['ZONE_PRIORITIZATION'] == prio]
        row = {}
        for m in key_metrics:
            vals = sub[sub['METRIC'] == m]['L0W_ROLL'].dropna()
            if len(vals):
                v = vals.mean()
                row[m] = v * 100 if abs(v) <= 1.5 else v
            else:
                row[m] = 0
        results[prio] = row
    short_names = ['Perfect\nOrders', 'Non-Pro\nConv.', 'Turbo\nAdop.', 'MLTV\nAdop.', 'Pro\nAdop.']
    x = np.arange(len(key_metrics))
    width = 0.25
    prio_colors = [red, orange, yellow]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for i, (prio, vals) in enumerate(results.items()):
        vals_list = [vals.get(m, 0) for m in key_metrics]
        ax.bar(x + i * width, vals_list, width, label=prio, color=prio_colors[i], alpha=0.82)
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, color=text, fontsize=8)
    ax.set_ylabel('Valor promedio (%)', color=text, fontsize=8)
    ax.set_title('Desempeño por Priorización de Zona', color=text, fontsize=10, fontweight='bold', pad=8)
    ax.legend(facecolor=mid, edgecolor=grid, labelcolor=text, fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text); ax.yaxis.grid(True, color=grid, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True); plt.tight_layout()
    return _fig_to_b64(fig, dark)


def chart_funnel_by_country(metrics_df, dark=False):
    """Show funnel conversion stages averaged by country."""
    bg, mid, grid, text, red, orange, green, yellow = _colors(dark)
    funnel_metrics = ['Restaurants SST > SS CVR', 'Restaurants SS > ATC CVR', 'Non-Pro PTC > OP']
    short = ['SST→SS\n(listing→tienda)', 'SS→ATC\n(tienda→carrito)', 'PTC→OP\n(pago→orden)']
    countries_ordered = ['CO', 'MX', 'AR', 'PE', 'BR', 'CL', 'EC', 'UY', 'CR']
    data = {}
    for country in countries_ordered:
        sub = metrics_df[metrics_df['COUNTRY'] == country]
        row = []
        for m in funnel_metrics:
            vals = sub[sub['METRIC'] == m]['L0W_ROLL'].dropna()
            row.append(vals.mean() * 100 if len(vals) and abs(vals.mean()) <= 1.5 else (vals.mean() if len(vals) else 0))
        data[country] = row
    x = np.arange(len(countries_ordered))
    width = 0.25
    funnel_colors = [red, orange, yellow]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for i, (m, s, c) in enumerate(zip(funnel_metrics, short, funnel_colors)):
        vals_list = [data[country][i] for country in countries_ordered]
        ax.bar(x + i * width, vals_list, width, label=s, color=c, alpha=0.82)
    ax.set_xticks(x + width)
    ax.set_xticklabels(countries_ordered, color=text, fontsize=9)
    ax.set_ylabel('% conversión', color=text, fontsize=8)
    ax.set_title('Embudo de Conversión en Restaurantes por País', color=text, fontsize=10, fontweight='bold', pad=8)
    ax.legend(facecolor=mid, edgecolor=grid, labelcolor=text, fontsize=7.5, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.spines['left'].set_color(grid)
    ax.tick_params(colors=text); ax.yaxis.grid(True, color=grid, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True); plt.tight_layout()
    return _fig_to_b64(fig, dark)


# ──────────────────────────────────────────────────────────────
# GEMINI — EXECUTIVE NARRATIVE ONLY
# ──────────────────────────────────────────────────────────────

def generate_executive_summary(raw_insights: dict) -> dict:
    top_anomalies = raw_insights['anomalies'][:8]
    top_opps      = raw_insights['opportunities'][:5]
    top_corr      = raw_insights['correlations'][:4]
    declining     = raw_insights['declining_trends'][:5]

    # Build compact human-readable summaries for the prompt
    anom_summary = "\n".join([
        f"- {a['zone']} ({a['country']}): {a['metric']} cambió {a['change_display']} WoW"
        for a in top_anomalies
    ])
    dec_summary = "\n".join([
        f"- {t['zone']} ({t['country']}): {t['metric']} cayó {t['total_change_pct']*100:.0f}% en 4 semanas"
        for t in declining
    ])
    opp_summary = "\n".join([
        f"- {o['zone']} ({o['country']}): {o['metric']} en {o['metric_value_fmt']} vs benchmark {o['benchmark_fmt']} — {o['orders']:,} órdenes/sem"
        for o in top_opps
    ])
    corr_summary = "\n".join([
        f"- {c['metric_a']} ↔ {c['metric_b']}: correlación {c['strength'].lower()} (r={c['correlation']:.2f})"
        for c in top_corr
    ])

    prompt = f"""Sos analista senior de Rappi. Escribí un resumen ejecutivo en español para un Director de Operaciones o VP de SP&A.

CONTEXTO: Rappi opera en 9 países (AR, BR, CL, CO, CR, EC, MX, PE, UY) con cientos de zonas.
Las métricas clave son:
- WoW: variación de una semana a la siguiente
- Perfect Orders: órdenes sin problemas / total (>85% es aceptable, >90% es bueno)
- Gross Profit UE: margen bruto por orden en moneda local (negativo = pérdida por orden)
- Lead Penetration: tiendas activas / prospectos identificados (mide cobertura de merchants)
- Non-Pro PTC→OP: conversión de checkout a orden completada (usuarios sin Pro)
- Turbo Adoption: % de usuarios que usan Turbo donde está disponible

ANOMALÍAS DETECTADAS (cambios abruptos semana vs semana):
{anom_summary}

TENDENCIAS EN DETERIORO (3+ semanas seguidas cayendo):
{dec_summary}

OPORTUNIDADES (zonas con alto volumen y métricas debajo del benchmark):
{opp_summary}

CORRELACIONES ESTADÍSTICAS ENTRE MÉTRICAS:
{corr_summary}

Escribí hallazgos que expliquen QUÉ está pasando, EN QUÉ MÉTRICA y CUÁL ES EL IMPACTO DE NEGOCIO.
Ejemplo BUENO: "GRAN_MENDOZA_GODOY (Argentina) perdió 1.70 puntos de Gross Profit UE en una semana — pasó de rentabilidad positiva a negativa, lo que indica que las órdenes en esta zona generan pérdida."
Ejemplo MALO: "Se detectaron anomalías WoW en múltiples métricas."

Devolvé EXACTAMENTE este JSON (sin markdown, sin texto extra):
{{
  "hallazgos_criticos": [
    "hallazgo 1: zona + país + métrica + qué pasó + impacto de negocio en 1-2 oraciones",
    "hallazgo 2",
    "hallazgo 3",
    "hallazgo 4",
    "hallazgo 5"
  ],
  "narrativa": "2 párrafos. Párrafo 1: explicar el patrón general de esta semana (qué métrica está más afectada y en qué países). Párrafo 2: conectar las oportunidades con las correlaciones — qué intervención concreta puede tener el mayor impacto.",
  "recomendaciones": [
    "ACCIÓN: [qué hacer concretamente] | ZONA/PAÍS: [dónde] | OWNER SUGERIDO: [equipo responsable: Ops Local, SP&A, Growth, Product, Finance] | TIMELINE: [esta semana / próximos 15 días / este mes] | IMPACTO ESPERADO: [cuantificación aproximada del beneficio]",
    "acción 2 con mismo formato",
    "acción 3", "acción 4", "acción 5", "acción 6", "acción 7", "acción 8"
  ],
  "alerta_critica": "1 frase. El hallazgo MÁS urgente con zona, métrica y magnitud concreta."
}}"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4, "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        }
    }
    try:
        resp = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                             json=payload, timeout=60)
        resp.raise_for_status()
        text = resp.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text.strip())
    except Exception as e:
        # Meaningful fallback using actual data — no generic phrases
        anoms = raw_insights.get('anomalies', [])
        corrs = raw_insights.get('correlations', [])
        dec   = raw_insights.get('declining_trends', [])
        opps  = raw_insights.get('opportunities', [])

        hallazgos = []
        if anoms:
            a = anoms[0]
            hallazgos.append(
                f"{a['zone']} ({a['country']}) registró un cambio de {a['change_display']} en {a['metric']} "
                f"semana vs semana — {'deterioro abrupto que requiere investigación urgente' if a['direction']=='deterioro' else 'mejora inesperada a investigar para replicar'}."
            )
        if len(anoms) > 1:
            gp_det = [a for a in anoms if 'Gross Profit' in a['metric'] and a['direction'] == 'deterioro']
            if gp_det:
                countries_affected = list(set(a['country'] for a in gp_det[:5]))
                hallazgos.append(
                    f"Gross Profit UE en deterioro en {len(gp_det)} zonas de {', '.join(countries_affected[:3])} — "
                    f"varias zonas operando con margen negativo esta semana."
                )
        if dec:
            d = dec[0]
            hallazgos.append(
                f"{d['zone']} ({d['country']}) acumula {abs(d['total_change_pct']*100):.0f}% de deterioro en {d['metric']} "
                f"durante 4 semanas consecutivas — señal de problema estructural, no puntual."
            )
        if opps:
            o = opps[0]
            hallazgos.append(
                f"Oportunidad prioritaria: {o['zone']} ({o['country']}) tiene {o['orders']:,} órdenes semanales "
                f"pero {o['metric']} en {o['metric_value_fmt']} vs benchmark {o['benchmark_fmt']} — "
                f"mejorar esta métrica tendría alto impacto inmediato."
            )
        if corrs:
            c = corrs[0]
            hallazgos.append(
                f"Correlación fuerte (r={c['correlation']:.2f}) entre {c['metric_a']} y {c['metric_b']}: "
                f"mejorar una arrastra a la otra — es el par de métricas con mayor palanca operacional."
            )
        while len(hallazgos) < 5:
            hallazgos.append(f"Se detectaron {len(raw_insights.get('benchmarking_gaps',[]))} brechas de benchmarking entre zonas del mismo país y tipo — revisar para identificar oportunidades de transferencia de buenas prácticas.")

        recs = []
        if anoms:
            recs.append(f"ACCIÓN: Investigar causa raíz del cambio de {anoms[0]['change_display']} en {anoms[0]['metric']} | ZONA/PAÍS: {anoms[0]['zone']} ({anoms[0]['country']}) | OWNER SUGERIDO: Ops Local | TIMELINE: Esta semana | IMPACTO ESPERADO: Prevenir propagación del deterioro a zonas adyacentes")
        if dec:
            recs.append(f"ACCIÓN: Plan de intervención urgente — {dec[0]['metric']} en caída por 4 semanas consecutivas | ZONA/PAÍS: {dec[0]['zone']} ({dec[0]['country']}) | OWNER SUGERIDO: SP&A + Ops Local | TIMELINE: Próximos 15 días | IMPACTO ESPERADO: Revertir caída de {abs(dec[0]['total_change_pct']*100):.0f}% acumulada")
        if opps:
            impact_txt = f"{opps[0].get('estimated_impact_orders', 0):,} órdenes adicionales de calidad/sem" if opps[0].get('estimated_impact_orders', 0) > 0 else f"mejora en margen de {opps[0].get('estimated_impact_gp', 0):.0f}/orden"
            recs.append(f"ACCIÓN: Priorizar mejora de {opps[0]['metric']} (actualmente {opps[0]['metric_value_fmt']} vs benchmark {opps[0]['benchmark_fmt']}) | ZONA/PAÍS: {opps[0]['zone']} ({opps[0]['country']}) — {opps[0]['orders']:,} órd/sem | OWNER SUGERIDO: Growth + Ops Local | TIMELINE: Este mes | IMPACTO ESPERADO: ~{impact_txt}")
        if corrs:
            recs.append(f"ACCIÓN: Focalizar intervención en {corrs[0]['metric_a']} — correlación fuerte (r={corrs[0]['correlation']:.2f}) con {corrs[0]['metric_b']} genera efecto cascada | OWNER SUGERIDO: SP&A | TIMELINE: Próximos 15 días | IMPACTO ESPERADO: Mejora simultánea en ambas métricas")
        recs += [
            "ACCIÓN: Documentar buenas prácticas de zonas con mejora consistente y crear playbook replicable | OWNER SUGERIDO: SP&A | TIMELINE: Este mes | IMPACTO ESPERADO: Acelerar mejora en zonas rezagadas con prácticas ya validadas",
            "ACCIÓN: Auditar estructura de costos en zonas con Gross Profit UE negativo o en caída | OWNER SUGERIDO: Finance + Ops | TIMELINE: Próximos 15 días | IMPACTO ESPERADO: Identificar y corregir fuentes de pérdida por orden",
            "ACCIÓN: Revisar pricing y descuentos en zonas con Restaurants Markdowns/GMV alto y Gross Profit bajo | OWNER SUGERIDO: Growth + Finance | TIMELINE: Este mes | IMPACTO ESPERADO: Reducir subsidios ineficientes sin impactar conversión",
            "ACCIÓN: Activar campaña de Turbo en zonas con baja adopción pero alta densidad de tiendas Turbo disponibles | OWNER SUGERIDO: Growth + Marketing | TIMELINE: Próximas 2 semanas | IMPACTO ESPERADO: Incrementar ticket promedio y diferenciación competitiva",
        ]

        alert = f"{anoms[0]['zone']} ({anoms[0]['country']}) — {anoms[0]['metric']}: {anoms[0]['change_display']} WoW." if anoms else "Múltiples zonas con Gross Profit UE negativo — revisar urgente."

        return {
            "hallazgos_criticos": hallazgos[:5],
            "narrativa": (
                f"Esta semana, la métrica con mayor presencia en las alertas es Gross Profit UE, "
                f"con deterioros abruptos en {', '.join(list(set(a['country'] for a in anoms[:5] if 'Gross Profit' in a['metric']))[:3])}. "
                f"Esto sugiere un problema sistémico en la estructura de costos o política de descuentos "
                f"que requiere revisión coordinada entre los equipos de operaciones y finanzas.\n\n"
                f"En cuanto a oportunidades, las zonas de alto volumen con métricas debajo del benchmark "
                f"representan el mayor potencial de impacto inmediato. La correlación detectada entre "
                f"{corrs[0]['metric_a'] if corrs else 'MLTV'} y {corrs[0]['metric_b'] if corrs else 'Turbo Adoption'} "
                f"sugiere que una intervención focalizada en una sola métrica puede tener efecto cascada positivo."
            ),
            "recomendaciones": recs[:6],
            "alerta_critica": alert,
        }


# ──────────────────────────────────────────────────────────────
# HTML REPORT BUILDER
# ──────────────────────────────────────────────────────────────

def _img_tag(b64, alt=''):
    if not b64:
        return '<div style="padding:20px;text-align:center;color:#444;font-size:12px">Sin datos suficientes</div>'
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:100%;border-radius:10px;display:block">'


def _wow_badge(direction, display):
    color = GREEN if direction == 'mejora' else RED
    bg    = 'rgba(34,197,94,0.12)' if direction == 'mejora' else 'rgba(255,68,31,0.12)'
    arrow = '▲' if direction == 'mejora' else '▼'
    return (f'<span style="background:{bg};color:{color};padding:2px 8px;'
            f'border-radius:100px;font-size:12px;font-weight:700">{arrow} {display}</span>')


def generate_html_report(raw_insights: dict, executive_summary: dict) -> str:
    now         = datetime.now().strftime('%d/%m/%Y %H:%M')
    scorecards  = raw_insights.get('scorecards', [])
    anomalies   = raw_insights.get('anomalies', [])
    declining   = raw_insights.get('declining_trends', [])
    improving   = raw_insights.get('improving_trends', [])
    gaps        = raw_insights.get('benchmarking_gaps', [])
    correlations= raw_insights.get('correlations', [])
    opps        = raw_insights.get('opportunities', [])
    hallazgos   = executive_summary.get('hallazgos_criticos', [])
    narrativa   = executive_summary.get('narrativa', '')
    recomendaciones = executive_summary.get('recomendaciones', [])
    alerta      = executive_summary.get('alerta_critica', '')

    # Charts
    ch_anom    = chart_anomalies_top(anomalies)
    ch_po      = chart_country_metric(scorecards, 'Perfect Orders')
    ch_lp      = chart_country_metric(scorecards, 'Lead Penetration')
    ch_gp      = chart_country_metric(scorecards, 'Gross Profit UE')
    ch_conv    = chart_country_metric(scorecards, 'Non-Pro PTC > OP')
    ch_turbo   = chart_country_metric(scorecards, 'Turbo Adoption')
    ch_dec     = chart_trend_sparklines(declining, 'Zonas en Deterioro Consistente (3+ semanas)', 'red')
    ch_imp     = chart_trend_sparklines(improving, 'Zonas con Mejora Consistente (3+ semanas)', 'green')
    ch_corr    = chart_correlations(correlations)
    ch_opp     = chart_opportunities(opps)

    # ── Scorecard table ──
    key_sc = ['Perfect Orders', 'Lead Penetration', 'Non-Pro PTC > OP', 'Gross Profit UE', 'Turbo Adoption']
    sc_headers = ''.join(f'<th>{m}</th>' for m in key_sc)
    sc_rows = ''
    for s in scorecards:
        cells = (f'<td style="font-weight:700;color:#eee;white-space:nowrap">{s["country"]}</td>'
                 f'<td style="color:#9999BB;text-align:center">{s["total_orders"]:,}</td>')
        for m in key_sc:
            if m in s['metrics']:
                d = s['metrics'][m]
                wc = GREEN if d['wow'] and d['wow'] > 0 else (RED if d['wow'] and d['wow'] < 0 else '#9999BB')
                cells += (f'<td style="text-align:center">'
                          f'<strong style="color:#eee">{d["fmt"]}</strong>'
                          f'<br><span style="font-size:10px;color:{wc}">{d["wow_fmt"]}</span></td>')
            else:
                cells += '<td style="text-align:center;color:#444">—</td>'
        sc_rows += f'<tr>{cells}</tr>'

    # ── Anomalies table ──
    anom_scores = [a.get('severity_score', 0) for a in anomalies[:15] if a.get('severity_score', 0) > 0]
    anom_p75 = sorted(anom_scores)[int(len(anom_scores)*0.75)] if len(anom_scores) > 2 else 10
    anom_p25 = sorted(anom_scores)[int(len(anom_scores)*0.25)] if len(anom_scores) > 2 else 3
    anom_rows = ''
    for a in anomalies[:15]:
        cur_fmt = format_metric_value(a['metric'], a['current'])
        pre_fmt = format_metric_value(a['metric'], a['previous'])
        sev = a.get('severity_score', 0)
        sev_label = 'CRÍTICA' if sev >= anom_p75 else ('ALTA' if sev >= anom_p25 else 'MEDIA')
        sev_color = RED if sev >= anom_p75 else (YELLOW if sev >= anom_p25 else '#9999BB')
        orders_val = a.get('orders', 0)
        anom_rows += (
            f'<tr><td><strong style="color:#eee">{a["zone"]}</strong>'
            f'<br><span style="font-size:11px;color:#9999BB">{a["city"]} · {a["country"]}</span></td>'
            f'<td style="font-size:11px;color:#9999BB">{a["zone_type"]}</td>'
            f'<td style="font-size:12px">{a["metric"]}</td>'
            f'<td style="text-align:center">{_wow_badge(a["direction"], a["change_display"])}</td>'
            f'<td style="text-align:center;font-family:monospace;font-size:12px">'
            f'{pre_fmt} → <strong style="color:#eee">{cur_fmt}</strong></td>'
            f'<td style="text-align:center;font-size:11px;color:#9999BB">{orders_val:,}</td>'
            f'<td style="text-align:center"><span style="color:{sev_color};font-weight:700;font-size:11px">{sev_label}</span></td></tr>'
        )

    # ── Trends table ──
    trend_rows = ''
    for t in declining[:10]:
        vals_html = ' → '.join(
            f'<span style="color:#9999BB;font-size:10px">{format_metric_value(t["metric"], v)}</span>'
            for v in t['values']
        )
        diag_html = t.get('diagnostic', '')
        trend_rows += (
            f'<tr><td><strong style="color:#eee">{t["zone"]}</strong>'
            f'<br><span style="font-size:11px;color:#9999BB">{t["city"]} · {t["country"]}</span></td>'
            f'<td style="font-size:12px">{t["metric"]}</td>'
            f'<td style="text-align:center"><span style="color:{RED};font-weight:700">'
            f'{t["total_change_pct"]*100:.1f}%</span></td>'
            f'<td style="font-size:11px">{vals_html}</td>'
            f'<td style="font-size:11px;color:#9999BB;max-width:250px">{diag_html}</td></tr>'
        )

    # ── Improving table ──
    impr_rows = ''
    for t in improving[:6]:
        impr_rows += (
            f'<tr><td><strong style="color:#eee">{t["zone"]}</strong>'
            f'<br><span style="font-size:11px;color:#9999BB">{t["city"]} · {t["country"]}</span></td>'
            f'<td style="font-size:12px">{t["metric"]}</td>'
            f'<td style="text-align:center"><span style="color:{GREEN};font-weight:700">'
            f'+{abs(t["total_change_pct"]*100):.1f}%</span></td></tr>'
        )

    # ── Gaps table ──
    gaps_rows = ''
    for g in gaps[:10]:
        top_z = g['top_zones'][0] if g['top_zones'] else {}
        bot_z = g['bot_zones'][0]  if g['bot_zones']  else {}
        top_fmt = format_metric_value(g['metric'], top_z.get('L0W_ROLL', 0))
        bot_fmt = format_metric_value(g['metric'], bot_z.get('L0W_ROLL', 0))
        gaps_rows += (
            f'<tr><td style="font-weight:600;color:#eee">{g["country"]}</td>'
            f'<td style="font-size:12px;color:#9999BB">{g["zone_type"]}</td>'
            f'<td style="font-size:12px">{g["metric"]}</td>'
            f'<td><span style="color:{GREEN};font-weight:700">{top_fmt}</span>'
            f'<br><span style="font-size:10px;color:#9999BB">{top_z.get("ZONE","")[:28]}</span></td>'
            f'<td><span style="color:{RED};font-weight:700">{bot_fmt}</span>'
            f'<br><span style="font-size:10px;color:#9999BB">{bot_z.get("ZONE","")[:28]}</span></td>'
            f'<td style="text-align:center;color:{YELLOW}">{g["zones_count"]}</td></tr>'
        )

    # ── Correlations table ──
    corr_rows = ''
    for c in correlations:
        bar_w = int(abs(c['correlation']) * 100)
        bc    = GREEN if c['correlation'] > 0 else RED
        bg_tag= 'rgba(34,197,94,0.1)' if c['direction'] == 'positiva' else 'rgba(255,68,31,0.1)'
        corr_rows += (
            f'<tr><td style="font-size:12px">{c["metric_a"]}</td>'
            f'<td style="font-size:12px">{c["metric_b"]}</td>'
            f'<td style="text-align:center">'
            f'<div style="background:#1C1C2E;border-radius:100px;height:7px;overflow:hidden;margin-bottom:3px">'
            f'<div style="width:{bar_w}%;background:{bc};height:100%;border-radius:100px"></div></div>'
            f'<span style="font-size:11px;color:{bc};font-weight:700">r = {c["correlation"]:.3f}</span></td>'
            f'<td style="text-align:center"><span style="background:{bg_tag};color:{bc};'
            f'padding:2px 10px;border-radius:100px;font-size:11px;font-weight:600">'
            f'{c["strength"]} · {c["direction"]}</span></td>'
            f'<td style="text-align:center;font-size:11px;color:#9999BB">{c["n_zones"]}</td></tr>'
        )

    # ── Opportunities table ──
    opp_rows = ''
    for o in opps:
        gap_pct = abs(o['gap_pct']) * 100
        imp_orders = o.get('estimated_impact_orders', 0)
        imp_gp = o.get('estimated_impact_gp', 0)
        if imp_orders > 0:
            impact_html = f'<span style="color:{ORANGE};font-weight:700;font-size:11px">~{imp_orders:,} órd. calidad/sem</span>'
        elif imp_gp != 0:
            impact_html = f'<span style="color:{ORANGE};font-weight:700;font-size:11px">~{imp_gp:+,.0f} margen/sem</span>'
        else:
            impact_html = f'<span style="color:{ORANGE};font-size:11px">{gap_pct:.0f}% gap</span>'
        opp_rows += (
            f'<tr><td><strong style="color:#eee">{o["zone"]}</strong>'
            f'<br><span style="font-size:11px;color:#9999BB">{o["city"]} · {o["country"]}</span></td>'
            f'<td style="font-size:12px;color:#9999BB">{o["zone_type"]}</td>'
            f'<td style="font-size:12px">{o["metric"]}</td>'
            f'<td style="text-align:center;font-family:monospace">'
            f'<span style="color:{RED};font-weight:700">{o["metric_value_fmt"]}</span>'
            f'<span style="color:#555;font-size:10px"> vs </span>'
            f'<span style="color:{GREEN};font-weight:700">{o["benchmark_fmt"]}</span></td>'
            f'<td style="text-align:center;font-weight:700;color:#eee">{o["orders"]:,}</td>'
            f'<td style="min-width:100px">'
            f'<div style="background:#1C1C2E;border-radius:100px;height:7px;overflow:hidden;margin-bottom:3px">'
            f'<div style="width:{min(gap_pct,100):.0f}%;background:{ORANGE};height:100%;border-radius:100px"></div></div>'
            f'{impact_html}</td></tr>'
        )

    # ── Hallazgos & recs ──
    hallazgos_html = ''.join(
        f'<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:10px">'
        f'<span style="color:{RED};font-size:16px;flex-shrink:0;margin-top:2px">◆</span>'
        f'<span style="color:#D0D0E8;font-size:13.5px;line-height:1.6">{h}</span></div>'
        for h in hallazgos
    )
    narrativa_html = ''.join(
        f'<p style="margin-bottom:12px;line-height:1.75;color:#B8B8D4;font-size:13.5px">{p.strip()}</p>'
        for p in narrativa.split('\n') if p.strip()
    )
    rec_html = ''.join(
        f'<li style="margin-bottom:8px;padding:11px 14px;background:#1C1C2E;border-radius:8px;'
        f'border-left:3px solid {RED};font-size:13px;color:#D8D8F0;list-style:none">'
        f'<span style="color:{ORANGE};font-weight:700;margin-right:8px">→</span>{rec}</li>'
        for rec in recomendaciones
    )

    # ── HTML OUTPUT ──
    return f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reporte Ejecutivo Rappi · {now}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"DM Sans",sans-serif;background:#0B0B14;color:#EEEEF5;min-height:100vh}}
.page{{max-width:1120px;margin:0 auto;padding:40px 24px 80px}}
.report-header{{background:linear-gradient(135deg,#13131F,#1C1C2E);border:1px solid #2E2E45;
  border-radius:16px;padding:32px 36px;margin-bottom:24px;position:relative;overflow:hidden}}
.report-header::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,{RED},{ORANGE})}}
.report-title{{font-size:27px;font-weight:700;letter-spacing:-0.5px;margin-bottom:6px}}
.report-title span{{color:{RED}}}
.report-meta{{font-size:13px;color:#6666AA;margin-bottom:10px}}
.report-badge{{display:inline-block;background:rgba(255,68,31,0.1);color:{ORANGE};
  padding:4px 14px;border-radius:100px;font-size:12px;font-weight:600;
  border:1px solid rgba(255,68,31,0.2)}}
.alert-banner{{background:rgba(255,68,31,0.07);border:1px solid rgba(255,68,31,0.3);
  border-left:4px solid {RED};border-radius:10px;padding:14px 18px;
  margin-bottom:24px;display:flex;align-items:center;gap:12px}}
.alert-text{{font-size:13.5px;color:#FFB0A0;font-weight:500}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin-bottom:24px}}
.kpi-card{{background:#13131F;border:1px solid #2E2E45;border-radius:12px;
  padding:18px 16px;text-align:center}}
.kpi-value{{font-size:28px;font-weight:700;line-height:1}}
.kpi-label{{font-size:10px;color:#6666AA;margin-top:6px;text-transform:uppercase;letter-spacing:0.5px}}
.section{{background:#13131F;border:1px solid #2E2E45;border-radius:14px;
  padding:26px 28px;margin-bottom:22px}}
.section-title{{font-size:17px;font-weight:700;margin-bottom:4px}}
.section-sub{{font-size:12px;color:#6666AA;margin-bottom:18px}}
.divider{{height:1px;background:#2E2E45;margin:18px 0}}
.chart-grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}
.chart-grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:14px 0}}
.chart-box{{background:#1C1C2E;border:1px solid #2E2E45;border-radius:10px;overflow:hidden;padding:4px}}
.chart-box.full{{grid-column:1/-1}}
.dt{{width:100%;border-collapse:collapse;font-size:13px}}
.dt th{{background:#1C1C2E;color:#9999BB;padding:9px 11px;text-align:left;
  font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.4px;
  border-bottom:1px solid #2E2E45}}
.dt td{{padding:9px 11px;border-bottom:1px solid #1A1A2A;vertical-align:middle}}
.dt tr:hover td{{background:rgba(255,255,255,0.02)}}
.dt tr:last-child td{{border-bottom:none}}
.sc{{width:100%;border-collapse:collapse;font-size:12px}}
.sc th{{background:rgba(255,68,31,0.08);color:{ORANGE};padding:9px 11px;text-align:center;
  font-weight:600;font-size:11px;border:1px solid #2E2E45;
  text-transform:uppercase;letter-spacing:0.3px}}
.sc th:first-child{{text-align:left}}
.sc td{{padding:9px 11px;border:1px solid #1A1A2A;text-align:center}}
.sc tr:hover td{{background:rgba(255,255,255,0.02)}}
.info-box{{background:#1C1C2E;border-left:3px solid {BLUE};border-radius:8px;
  padding:13px 16px;margin-top:14px;font-size:13px;color:#9999BB}}
.footer{{text-align:center;color:#3A3A55;font-size:12px;margin-top:40px;
  padding-top:20px;border-top:1px solid #1A1A2A}}
</style>
</head>
<body>
<div class="page">

<!-- HEADER -->
<div class="report-header">
  <div class="report-title">🦊 Reporte Ejecutivo de Operaciones — <span>Rappi</span></div>
  <div class="report-meta">Generado automáticamente · {now} · Análisis de {len(scorecards)} países, {len(anomalies)+len(declining)} alertas detectadas</div>
  <div class="report-badge">⚡ SP&amp;A Intelligence Suite · Gemini 2.5 Flash</div>
</div>

<!-- ALERT -->
{"" if not alerta else f'<div class="alert-banner"><div style="font-size:22px;flex-shrink:0">🚨</div><div class="alert-text"><strong>Alerta Crítica:</strong> {alerta}</div></div>'}

<!-- KPI STRIP -->
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-value" style="color:{RED}">{len(anomalies)}</div><div class="kpi-label">Anomalías WoW</div></div>
  <div class="kpi-card"><div class="kpi-value" style="color:{YELLOW}">{len(declining)}</div><div class="kpi-label">Tendencias Negativas</div></div>
  <div class="kpi-card"><div class="kpi-value" style="color:{GREEN}">{len(improving)}</div><div class="kpi-label">Tendencias Positivas</div></div>
  <div class="kpi-card"><div class="kpi-value" style="color:{ORANGE}">{len(opps)}</div><div class="kpi-label">Oportunidades</div></div>
  <div class="kpi-card"><div class="kpi-value" style="color:{BLUE}">{len(gaps)}</div><div class="kpi-label">Brechas Benchmark</div></div>
  <div class="kpi-card"><div class="kpi-value" style="color:#9999BB">{len(scorecards)}</div><div class="kpi-label">Países</div></div>
</div>

<!-- EXECUTIVE SUMMARY -->
<div class="section">
  <div class="section-title">📋 Resumen Ejecutivo</div>
  <div class="section-sub">Análisis generado por IA · Para VP Operations &amp; SP&amp;A Leadership</div>
  <div style="margin-bottom:18px">{hallazgos_html}</div>
  <div class="divider"></div>
  {narrativa_html}
</div>

<!-- COUNTRY SCORECARDS -->
<div class="section">
  <div class="section-title">🌎 Scorecard por País — Métricas Clave</div>
  <div class="section-sub">Mediana por país · WoW = variación vs semana anterior · Ordenado por volumen de órdenes</div>
  <div style="overflow-x:auto">
    <table class="sc">
      <thead><tr><th style="text-align:left">País</th><th>Órdenes</th>{sc_headers}</tr></thead>
      <tbody>{sc_rows}</tbody>
    </table>
  </div>
  <div class="chart-grid-2" style="margin-top:18px">
    <div class="chart-box">{_img_tag(ch_po, 'Perfect Orders')}</div>
    <div class="chart-box">{_img_tag(ch_lp, 'Lead Penetration')}</div>
    <div class="chart-box">{_img_tag(ch_gp, 'Gross Profit UE')}</div>
    <div class="chart-box">{_img_tag(ch_conv, 'Conversión No-Pro')}</div>
  </div>
</div>

<!-- ANOMALIES -->
<div class="section">
  <div class="section-title">⚠️ Anomalías — Cambios Drásticos Semana vs Semana</div>
  <div class="section-sub">Variación &gt;10% WoW en métricas de ratio · O cambio absoluto significativo en métricas numéricas</div>
  <div class="chart-box full" style="margin-bottom:16px">{_img_tag(ch_anom, 'Top deterioros')}</div>
  <div style="overflow-x:auto">
    <table class="dt">
      <thead><tr><th>Zona</th><th>Tipo</th><th>Métrica</th><th>Cambio WoW</th><th>Anterior → Actual</th><th>Órd/sem</th><th>Severidad</th></tr></thead>
      <tbody>{anom_rows}</tbody>
    </table>
  </div>
</div>

<!-- TRENDS -->
<div class="section">
  <div class="section-title">📉 Tendencias Preocupantes — Deterioro 3+ Semanas</div>
  <div class="section-sub">Zonas con declive sostenido — señal de problemas estructurales que requieren intervención</div>
  {_img_tag(ch_dec, 'Tendencias deterioro')}
  <div style="overflow-x:auto;margin-top:14px">
    <table class="dt">
      <thead><tr><th>Zona</th><th>Métrica</th><th>Cambio Total</th><th>Evolución (últimas 4 semanas)</th><th>Diagnóstico Probable</th></tr></thead>
      <tbody>{trend_rows}</tbody>
    </table>
  </div>
  {f'''<div class="divider"></div>
  <div class="section-title" style="font-size:15px;color:{GREEN};margin-bottom:12px">📈 Zonas con Mejora Consistente — Casos a Replicar</div>
  {_img_tag(ch_imp, "Tendencias mejora")}
  <div style="overflow-x:auto;margin-top:12px">
    <table class="dt"><thead><tr><th>Zona</th><th>Métrica</th><th>Mejora Total</th></tr></thead>
    <tbody>{impr_rows}</tbody></table>
  </div>''' if improving else ''}
</div>

<!-- BENCHMARKING -->
<div class="section">
  <div class="section-title">📊 Benchmarking — Brechas entre Zonas del Mismo País y Tipo</div>
  <div class="section-sub">Mismo contexto geográfico y socioeconómico, resultados distintos — indica oportunidad de mejora</div>
  <div style="overflow-x:auto">
    <table class="dt">
      <thead><tr><th>País</th><th>Tipo de Zona</th><th>Métrica</th><th>Zona Top ▲</th><th>Zona Débil ▼</th><th>Zonas</th></tr></thead>
      <tbody>{gaps_rows}</tbody>
    </table>
  </div>
</div>

<!-- CORRELATIONS -->
<div class="section">
  <div class="section-title">🔗 Correlaciones entre Métricas</div>
  <div class="section-sub">Relaciones estadísticas — útiles para priorizar intervenciones con efecto cascada</div>
  {_img_tag(ch_corr, 'Correlaciones')}
  <div style="overflow-x:auto;margin-top:14px">
    <table class="dt">
      <thead><tr><th>Métrica A</th><th>Métrica B</th><th>Intensidad</th><th>Característica</th><th>N° Zonas</th></tr></thead>
      <tbody>{corr_rows}</tbody>
    </table>
  </div>
  <div class="info-box">
    💡 <strong style="color:#eee">¿Cómo usarlo?</strong>
    Una correlación positiva fuerte (r &gt; 0.65) significa que ambas métricas suben juntas.
    Mejorar una puede arrastrar a la otra. Priorizá intervenciones en el par de métricas con mayor correlación
    y menor nivel actual para maximizar el impacto operacional.
  </div>
</div>

<!-- OPPORTUNITIES -->
<div class="section">
  <div class="section-title">💎 Oportunidades de Mayor Impacto</div>
  <div class="section-sub">Zonas con alto volumen de órdenes y métricas por debajo del benchmark — mayor potencial de impacto inmediato</div>
  {_img_tag(ch_opp, 'Oportunidades')}
  <div style="overflow-x:auto;margin-top:14px">
    <table class="dt">
      <thead><tr><th>Zona</th><th>Tipo</th><th>Métrica con Gap</th><th>Actual vs Benchmark</th><th>Órdenes/sem</th><th>Impacto Estimado</th></tr></thead>
      <tbody>{opp_rows}</tbody>
    </table>
  </div>
</div>

<!-- RECOMMENDATIONS -->
<div class="section">
  <div class="section-title">✅ Recomendaciones Accionables para Esta Semana</div>
  <div class="section-sub">Priorizadas por potencial de impacto en negocio</div>
  <ul style="list-style:none;padding:0">{rec_html}</ul>
</div>

<div class="footer">
  Rappi Analytics Intelligence Suite · Reporte generado el {now}<br>
  Datos indicativos — validar con equipos locales antes de tomar decisiones.
</div>
</div>
</body>
</html>'''


def generate_report_with_gemini(raw_insights: dict) -> str:
    executive_summary = generate_executive_summary(raw_insights)
    raw_insights['executive_summary'] = executive_summary
    return generate_html_report(raw_insights, executive_summary)


def generate_pdf_report(raw_insights: dict, output_path: str) -> str:
    """Professional PDF report — ReportLab only, no browser needed."""
    import re, base64 as b64lib
    from io import BytesIO
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, PageBreak, KeepTogether
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    # ── Palette ───────────────────────────────────────────────────
    CR  = HexColor('#E03000')   # Rappi red
    CO  = HexColor('#FF6B35')   # orange
    CG  = HexColor('#16A34A')   # green
    CY  = HexColor('#D97706')   # amber
    CB  = HexColor('#2563EB')   # blue
    CSF = HexColor('#F8F9FA')   # section bg
    CCA = HexColor('#F1F3F5')   # card/table header
    CBR = HexColor('#DEE2E6')   # border
    CTX = HexColor('#1F2937')   # main text
    CMT = HexColor('#6B7280')   # muted text
    CWH = HexColor('#FFFFFF')   # white

    W, H = A4
    LM = RM = 18*mm
    CW = W - LM - RM   # content width ~159mm

    # ── Style factory ─────────────────────────────────────────────
    def ps(name, font='Helvetica', size=9, color=CTX, **kw):
        return ParagraphStyle(name, fontName=font, fontSize=size, textColor=color, **kw)

    sT1  = ps('T1','Helvetica-Bold',20,CR, spaceAfter=2, leading=24)
    sT2  = ps('T2','Helvetica-Bold',12,CO, spaceAfter=6, leading=15)
    sMt  = ps('Mt','Helvetica',8,CMT, spaceAfter=8)
    sH2  = ps('H2','Helvetica-Bold',12,CTX, spaceBefore=4, spaceAfter=3, leading=15)
    sH3  = ps('H3','Helvetica-Bold',10,CR,  spaceBefore=6, spaceAfter=2)
    sH3g = ps('H3g','Helvetica-Bold',10,CG, spaceBefore=6, spaceAfter=2)
    sBod = ps('Bd','Helvetica',9,CMT,  leading=13, spaceAfter=4)
    sBul = ps('Bl','Helvetica',9,CTX,  leading=13, leftIndent=10, spaceAfter=3)
    sInt = ps('It','Helvetica-Oblique',8.5,CMT, leading=12, spaceAfter=3,
               leftIndent=6, borderPad=4)
    sTH  = ps('TH','Helvetica-Bold',7.5,CO,  alignment=TA_LEFT)
    sTD  = ps('TD','Helvetica',8,CTX,  leading=11)
    sTDm = ps('TDm','Helvetica',7.5,CMT, leading=10)
    sTDg = ps('TDg','Helvetica-Bold',8,CG)
    sTDr = ps('TDr','Helvetica-Bold',8,CR)
    sCtr = ps('Ct','Helvetica',8,CTX,  alignment=TA_CENTER, leading=11)
    sIdx = ps('Ix','Helvetica',9,CB,   leading=13, spaceAfter=2)
    sGlo = ps('Gl','Helvetica',8.5,CTX, leading=13, spaceAfter=4)
    sGlB = ps('GlB','Helvetica-Bold',8.5,CTX, spaceAfter=1)
    sFt  = ps('Ft','Helvetica',7,CMT,  alignment=TA_CENTER, leading=9)

    def hr(c=CBR, t=0.5, sb=4, sa=4):
        return HRFlowable(width='100%', thickness=t, color=c, spaceBefore=sb, spaceAfter=sa)

    BASE_TS = [
        ('BACKGROUND',   (0,0),(-1,0),  CCA),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[CWH, CSF]),
        ('GRID',         (0,0),(-1,-1), 0.4, CBR),
        ('TOPPADDING',   (0,0),(-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ('LEFTPADDING',  (0,0),(-1,-1), 5),
        ('RIGHTPADDING', (0,0),(-1,-1), 5),
        ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
    ]
    def tbl(rows, cw, extra=None):
        t = Table(rows, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle(BASE_TS + (extra or [])))
        return t

    def img_flow(b64_str, width=CW, max_h=170):
        if not b64_str:
            return None
        try:
            data = b64lib.b64decode(b64_str)
            buf  = BytesIO(data)
            im   = RLImage(buf, width=width)
            if im.imageHeight and im.imageWidth:
                ratio = im.imageHeight / im.imageWidth
                h = width * ratio
                if h > max_h:
                    nw = max_h / ratio
                    im = RLImage(BytesIO(data), width=nw, height=max_h)
            im.hAlign = 'CENTER'
            return im
        except Exception:
            return None

    def interpretation(text):
        """Grey italic interpretation box under charts/tables."""
        return Paragraph(f'📌 {text}', sInt)

    # ── Pull data ─────────────────────────────────────────────────
    anomalies  = raw_insights.get('anomalies', [])
    declining  = raw_insights.get('declining_trends', [])
    improving  = raw_insights.get('improving_trends', [])
    gaps       = raw_insights.get('benchmarking_gaps', [])
    corrs      = raw_insights.get('correlations', [])
    opps       = raw_insights.get('opportunities', [])
    scorecards = raw_insights.get('scorecards', [])
    ex         = raw_insights.get('executive_summary', {})
    hallazgos  = ex.get('hallazgos_criticos', ex.get('hallazgos', []))
    recs       = ex.get('recomendaciones', [])
    narrativa  = ex.get('narrativa', '')

    # ── Generate charts ───────────────────────────────────────────
    metrics_df = raw_insights.get('_metrics_df', None)
    ch_anom = chart_anomalies_top(anomalies)
    ch_dec  = chart_trend_sparklines(declining, 'Tendencias de Deterioro', 'red')
    ch_imp  = chart_trend_sparklines(improving, 'Zonas con Mejora Consistente', 'green')
    ch_corr = chart_correlations(corrs)
    ch_opp  = chart_opportunities(opps)
    ch_po   = chart_country_metric(scorecards, 'Perfect Orders')
    ch_lp   = chart_country_metric(scorecards, 'Lead Penetration')
    ch_gp   = chart_country_metric(scorecards, 'Gross Profit UE')
    ch_conv = chart_country_metric(scorecards, 'Non-Pro PTC > OP')
    ch_prio = chart_high_priority_perf(metrics_df) if metrics_df is not None else None
    ch_funnel = chart_funnel_by_country(metrics_df) if metrics_df is not None else None
    ch_heatmap = chart_correlation_heatmap(metrics_df) if metrics_df is not None else None
    # Scatter plots for top 2 strongest correlations
    ch_scatter_1 = None
    ch_scatter_2 = None
    if corrs and metrics_df is not None:
        c1 = corrs[0]
        ch_scatter_1 = chart_correlation_scatter(metrics_df, c1['metric_a'], c1['metric_b'], c1['correlation'])
        if len(corrs) > 1:
            c2 = corrs[1]
            ch_scatter_2 = chart_correlation_scatter(metrics_df, c2['metric_a'], c2['metric_b'], c2['correlation'])

    now = datetime.now().strftime('%d/%m/%Y %H:%M')

    story = []

    # ════════════════════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════════════════════
    story.append(Spacer(1, 12*mm))
    story.append(Paragraph('Reporte Ejecutivo de Operaciones', sT1))
    story.append(Paragraph('Rappi Analytics — SP&A Intelligence Suite', sT2))
    story.append(Paragraph(f'Generado automáticamente · {now} · Powered by Gemini 2.5 Flash', sMt))
    story.append(hr(CR, 1.5))
    story.append(Spacer(1, 6))

    # KPI strip
    kpi_d = [
        [Paragraph('ANOMALÍAS WoW', ps('kl','Helvetica-Bold',7,CO,alignment=TA_CENTER)),
         Paragraph('TEND. NEGATIVAS', ps('kl','Helvetica-Bold',7,CO,alignment=TA_CENTER)),
         Paragraph('TEND. POSITIVAS', ps('kl','Helvetica-Bold',7,CO,alignment=TA_CENTER)),
         Paragraph('OPORTUNIDADES', ps('kl','Helvetica-Bold',7,CO,alignment=TA_CENTER)),
         Paragraph('BRECHAS', ps('kl','Helvetica-Bold',7,CO,alignment=TA_CENTER)),
         Paragraph('PAÍSES', ps('kl','Helvetica-Bold',7,CO,alignment=TA_CENTER))],
        [Paragraph(str(len(anomalies)), ps('kv','Helvetica-Bold',22,CR,alignment=TA_CENTER)),
         Paragraph(str(len(declining)), ps('kv','Helvetica-Bold',22,CY,alignment=TA_CENTER)),
         Paragraph(str(len(improving)), ps('kv','Helvetica-Bold',22,CG,alignment=TA_CENTER)),
         Paragraph(str(len(opps)),      ps('kv','Helvetica-Bold',22,CO,alignment=TA_CENTER)),
         Paragraph(str(len(gaps)),      ps('kv','Helvetica-Bold',22,CB,alignment=TA_CENTER)),
         Paragraph(str(len(scorecards)),ps('kv','Helvetica-Bold',22,CMT,alignment=TA_CENTER))],
    ]
    kpi_t = Table(kpi_d, colWidths=[CW/6]*6)
    kpi_t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),CCA),
        ('GRID',(0,0),(-1,-1),0.5,CBR),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEABOVE',(0,0),(-1,0),2,CR),
    ]))
    story.append(kpi_t)
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════
    # INDEX
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('Índice del Reporte', sH2))
    story.append(hr())
    sections = [
        ('1.', 'Resumen Ejecutivo — Principales Hallazgos y Recomendaciones'),
        ('2.', 'Scorecard por País — Métricas Clave de la Semana'),
        ('3.', 'Anomalías Detectadas — Cambios Drásticos WoW'),
        ('4.', 'Tendencias Preocupantes — Deterioro Sostenido 3+ Semanas'),
        ('5.', 'Benchmarking — Brechas entre Zonas Similares'),
        ('6.', 'Correlaciones entre Métricas Operacionales'),
        ('7.', 'Oportunidades de Alto Impacto'),
        ('8.', 'Desempeño por Priorización de Zona'),
        ('9.', 'Embudo de Conversión en Restaurantes por País'),
        ('10.', 'Recomendaciones Accionables para Esta Semana'),
        ('11.', 'Glosario de Métricas'),
    ]
    for num, title in sections:
        story.append(Paragraph(f'<font color="#E03000"><b>{num}</b></font>  {title}', sIdx))
    story.append(Spacer(1, 6))

    # ════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('1.  Resumen Ejecutivo', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Este reporte consolida el análisis automático de todas las zonas operacionales de Rappi '
        'en los 9 países de operación. Fue generado por IA a partir de los datos de la última semana '
        'y tiene como objetivo principal identificar alertas, oportunidades y acciones concretas '
        'para los equipos de SP&A y Operaciones.', sBod))
    story.append(Spacer(1, 4))

    story.append(Paragraph('Principales Hallazgos de la Semana', sH3))
    story.append(Paragraph(
        'Cada hallazgo incluye contexto de negocio para facilitar la comprensión y toma de decisión:', sBod))

    HALLAZGO_CONTEXT = {
        'Gross Profit UE': 'es el margen bruto por orden — una caída indica que las órdenes están siendo menos rentables (mayor costo operativo, descuentos excesivos o bajo ticket promedio).',
        'Perfect Orders': 'mide la tasa de órdenes sin problemas (sin cancelaciones, demoras ni defectos) — valores bajos afectan directamente la experiencia del usuario y la retención.',
        'Lead Penetration': 'indica qué porcentaje de los comercios potenciales ya están activos en Rappi — un valor bajo sugiere oportunidad de expansión de la red de merchants.',
        'Turbo Adoption': 'mide adopción del servicio de entrega ultra-rápida — clave para diferenciación y mayor ticket promedio.',
        'MLTV Top Verticals': 'mide si los usuarios compran en múltiples verticales (restaurantes, supermercados, farmacias) — usuarios multi-vertical tienen mayor lifetime value.',
        'Non-Pro PTC > OP': 'es la tasa de conversión de usuarios no-Pro desde "ir a pagar" hasta "orden confirmada" — caídas indican fricción en el checkout.',
    }
    for h in hallazgos[:6]:
        clean = re.sub(r'<[^>]+>', '', str(h))
        # Find relevant context
        ctx = ''
        for key, explanation in HALLAZGO_CONTEXT.items():
            if key.lower() in clean.lower():
                ctx = f' ({explanation})'
                break
        story.append(Paragraph(f'▸  {clean}{ctx}', sBul))
    story.append(Spacer(1, 6))

    if narrativa:
        story.append(Paragraph('Análisis General', sH3))
        for para in narrativa.split('\n'):
            p = re.sub(r'<[^>]+>', '', para.strip())
            if p:
                story.append(Paragraph(p, sBod))
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════
    # 2. COUNTRY SCORECARDS
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('2.  Scorecard por País — Métricas Clave de la Semana', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Resumen de métricas principales por país. Los valores son la mediana de todas las zonas de ese país. '
        '"WoW" (Week over Week) indica si la métrica mejoró o empeoró respecto a la semana anterior. '
        'Un ↑ verde es mejora, un ↓ rojo es deterioro. Los países están ordenados por volumen total de órdenes.', sBod))
    story.append(Spacer(1, 5))

    if scorecards:
        sc_metrics = ['Perfect Orders', 'Lead Penetration', 'Gross Profit UE', 'Non-Pro PTC > OP', 'Turbo Adoption']
        sc_hdr = [Paragraph('País', sTH), Paragraph('Órdenes\n(sem.)', sTH)]
        for m in sc_metrics:
            sc_hdr.append(Paragraph(m[:16], sTH))
        sc_rows = [sc_hdr]
        for sc in scorecards:
            row = [
                Paragraph(f"{sc['country']}", ps('cc','Helvetica-Bold',8,CTX)),
                Paragraph(f"{sc['total_orders']:,}", sCtr),
            ]
            for m in sc_metrics:
                md = sc['metrics'].get(m)
                if md:
                    wow = md.get('wow')
                    arrow = '↑' if wow and wow > 0 else ('↓' if wow and wow < 0 else '→')
                    col = CG if wow and wow > 0 else (CR if wow and wow < 0 else CMT)
                    row.append(Paragraph(
                        f"{md['fmt']}\n<font color='{'#16A34A' if wow and wow > 0 else '#E03000' if wow and wow < 0 else '#6B7280'}'>{arrow} {md.get('wow_fmt','—')}</font>",
                        ps('sv','Helvetica',7.5,CTX, alignment=TA_CENTER, leading=10)
                    ))
                else:
                    row.append(Paragraph('—', sCtr))
            sc_rows.append(row)

        sc_cw = [CW*0.13, CW*0.10] + [CW*0.154]*5
        story.append(tbl(sc_rows, sc_cw))
        story.append(Spacer(1, 5))
        story.append(interpretation(
            '¿Qué buscar? Países con múltiples ↓ en Perfect Orders y Gross Profit UE al mismo tiempo '
            'indican problemas operacionales sistémicos, no puntuales. '
            'Uruguay y Perú suelen liderar en Perfect Orders — sus prácticas son replicables.'
        ))

    # Country charts - full width, one per row to avoid distortion
    story.append(Spacer(1, 8))
    for ch, title in [(ch_po, 'Perfect Orders'), (ch_lp, 'Lead Penetration'),
                      (ch_gp, 'Gross Profit UE'), (ch_conv, 'Non-Pro PTC > OP')]:
        img = img_flow(ch, CW, 140)
        if img:
            story.append(img)
            story.append(Spacer(1, 6))

    # ════════════════════════════════════════════════════════════
    # 3. ANOMALIES
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('3.  Anomalías Detectadas — Cambios Drásticos WoW', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Qué es una anomalía: Un salto brusco (>10%) en una métrica de una semana a la siguiente. '
        'No implica tendencia — puede ser un evento puntual (campaña, falla técnica, cambio de pricing). '
        'Tanto las caídas como las mejoras inesperadas merecen investigación.', sBod))
    story.append(Spacer(1, 5))

    ch_anom_img = img_flow(ch_anom, CW, 160)
    if ch_anom_img:
        story.append(ch_anom_img)
        story.append(Spacer(1, 4))
        story.append(interpretation(
            'Cómo leer el gráfico: La magnitud de la barra = tamaño del cambio WoW. '
            'Las zonas con mayor magnitud son las que cambiaron más bruscamente esta semana. '
            'Para priorizar cuál investigar primero, cruzar con el volumen de órdenes (sección 9).'
        ))

    story.append(Spacer(1, 6))
    if anomalies:
        det = [a for a in anomalies if a['direction'] == 'deterioro'][:10]
        mej = [a for a in anomalies if a['direction'] == 'mejora'][:5]

        story.append(Paragraph('Deterioros más significativos (ordenados por severidad = magnitud × volumen)', sH3))
        rows = [[Paragraph('Zona', sTH), Paragraph('País', sTH), Paragraph('Métrica', sTH),
                 Paragraph('Cambio WoW', sTH), Paragraph('Ant. → Actual', sTH),
                 Paragraph('Órd/sem', sTH), Paragraph('Severidad', sTH)]]
        # Compute severity thresholds from data
        det_scores = [a.get('severity_score', 0) for a in det if a.get('severity_score', 0) > 0]
        sev_p75 = sorted(det_scores)[int(len(det_scores)*0.75)] if len(det_scores) > 2 else 10
        sev_p25 = sorted(det_scores)[int(len(det_scores)*0.25)] if len(det_scores) > 2 else 3
        for a in det:
            sev = a.get('severity_score', 0)
            sev_label = 'CRÍTICA' if sev >= sev_p75 else ('ALTA' if sev >= sev_p25 else 'MEDIA')
            sev_color = CR if sev >= sev_p75 else (CY if sev >= sev_p25 else CMT)
            rows.append([
                Paragraph(a['zone'][:28], sTD),
                Paragraph(a['country'], sTDm),
                Paragraph(a['metric'][:24], sTDm),
                Paragraph(a['change_display'], ps('cd','Helvetica-Bold',9,CR, alignment=TA_CENTER)),
                Paragraph(f"{format_metric_value(a['metric'], a['previous'])} → "
                          f"{format_metric_value(a['metric'], a['current'])}", sTDm),
                Paragraph(f"{a.get('orders', 0):,}", sCtr),
                Paragraph(sev_label, ps('sv','Helvetica-Bold',8,sev_color, alignment=TA_CENTER)),
            ])
        story.append(tbl(rows, [CW*0.22, CW*0.10, CW*0.20, CW*0.12, CW*0.18, CW*0.09, CW*0.09]))

        if mej:
            story.append(Spacer(1, 8))
            story.append(Paragraph('Mejoras inesperadas — investigar causa para replicar', sH3g))
            rows2 = [[Paragraph('Zona', sTH), Paragraph('País', sTH), Paragraph('Métrica', sTH),
                      Paragraph('Cambio WoW', sTH)]]
            for a in mej:
                rows2.append([
                    Paragraph(a['zone'][:32], sTD),
                    Paragraph(a['country'], sTDm),
                    Paragraph(a['metric'][:28], sTDm),
                    Paragraph(a['change_display'], ps('cd','Helvetica-Bold',9,CG, alignment=TA_CENTER)),
                ])
            story.append(tbl(rows2, [CW*0.35, CW*0.15, CW*0.32, CW*0.18]))

    # ════════════════════════════════════════════════════════════
    # 4. TRENDS
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('4.  Tendencias Preocupantes — Deterioro 3+ Semanas Consecutivas', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Diferencia con anomalías: Una tendencia es un deterioro gradual durante 3+ semanas seguidas. '
        'Es más preocupante que una anomalía porque indica un problema estructural, no un evento puntual. '
        'Estas zonas necesitan un plan de intervención, no solo monitoreo.', sBod))
    story.append(Spacer(1, 5))

    if ch_dec:
        ch_dec_img = img_flow(ch_dec, CW, 200)
        if ch_dec_img:
            story.append(ch_dec_img)
            story.append(Spacer(1, 4))
            story.append(interpretation(
                'Cómo leer los mini-gráficos: Cada cuadro muestra la evolución de una zona en las últimas '
                '4 semanas. Una línea con pendiente negativa consistente = problema estructural. '
                'Compará el valor inicial (Sem -3) con el actual para dimensionar el impacto acumulado.'
            ))

    if declining:
        story.append(Spacer(1, 6))
        rows = [[Paragraph('Zona', sTH), Paragraph('País', sTH), Paragraph('Métrica', sTH),
                 Paragraph('Caída (4 sem.)', sTH), Paragraph('Urgencia', sTH),
                 Paragraph('Diagnóstico Probable', sTH)]]
        for t in declining[:10]:
            pct = t['total_change_pct'] * 100
            urgencia = ('Monitorear' if abs(pct) < 20 else
                       'Intervenir' if abs(pct) < 50 else
                       'Urgente')
            diag = t.get('diagnostic', 'Revisar con equipo local.')
            rows.append([
                Paragraph(t['zone'][:26], sTD),
                Paragraph(t['country'], sTDm),
                Paragraph(t['metric'][:22], sTDm),
                Paragraph(f"{pct:+.1f}%", ps('tc','Helvetica-Bold',9,CR,alignment=TA_CENTER)),
                Paragraph(urgencia, ps('ur','Helvetica-Bold',8,CR if abs(pct)>=50 else CY,alignment=TA_CENTER)),
                Paragraph(diag[:80], ps('dg','Helvetica',7,CMT, leading=9)),
            ])
        story.append(tbl(rows, [CW*0.18, CW*0.09, CW*0.16, CW*0.11, CW*0.10, CW*0.36]))

    if improving:
        story.append(Spacer(1, 10))
        story.append(Paragraph('Zonas con Mejora Consistente — Casos a Replicar', sH3g))
        story.append(Paragraph(
            'Estas zonas mejoraron durante 3+ semanas seguidas. Investigá qué cambió '
            '(nuevo equipo, campaña local, ajuste operacional) para replicar la práctica.', sBod))
        if ch_imp:
            ch_imp_img = img_flow(ch_imp, CW, 180)
            if ch_imp_img:
                story.append(ch_imp_img)

    # ════════════════════════════════════════════════════════════
    # 5. BENCHMARKING
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('5.  Benchmarking — Brechas entre Zonas Similares', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'El benchmarking compara zonas del mismo país y mismo tipo (Wealthy / Non Wealthy). '
        'Si dos zonas tienen el mismo contexto socioeconómico y geográfico pero resultados muy '
        'distintos, la zona débil tiene una oportunidad de mejora concreta: puede alcanzar el '
        'nivel de la zona top adoptando sus prácticas. Una brecha grande es una oportunidad, no solo un problema.', sBod))
    story.append(Spacer(1, 5))

    if gaps:
        rows = [[Paragraph('País', sTH), Paragraph('Tipo de Zona', sTH),
                 Paragraph('Métrica', sTH), Paragraph('Zona Referente (Top)', sTH),
                 Paragraph('Zona con Oportunidad', sTH), Paragraph('N° Zonas', sTH)]]
        for g in gaps[:10]:
            top_z = g['top_zones'][0] if g['top_zones'] else {}
            bot_z = g['bot_zones'][0]  if g['bot_zones']  else {}
            top_fmt = format_metric_value(g['metric'], top_z.get('L0W_ROLL', 0))
            bot_fmt = format_metric_value(g['metric'], bot_z.get('L0W_ROLL', 0))
            rows.append([
                Paragraph(g['country'], sTD),
                Paragraph(g['zone_type'], sTDm),
                Paragraph(g['metric'][:24], sTDm),
                Paragraph(f"{top_fmt}\n{top_z.get('ZONE','')[:26]}",
                          ps('gt','Helvetica',7.5,CG, leading=10)),
                Paragraph(f"{bot_fmt}\n{bot_z.get('ZONE','')[:26]}",
                          ps('gr','Helvetica',7.5,CR, leading=10)),
                Paragraph(str(g['zones_count']), sCtr),
            ])
        story.append(tbl(rows, [CW*0.12, CW*0.13, CW*0.19, CW*0.23, CW*0.23, CW*0.10]))
        story.append(Spacer(1, 5))
        story.append(interpretation(
            'La columna "Zona Referente" muestra el benchmark ideal dentro del mismo mercado. '
            'La columna "Zona con Oportunidad" es donde se puede aplicar una intervención concreta. '
            'Priorizá las brechas en métricas de alta visibilidad para el usuario como Perfect Orders.'
        ))

    # ════════════════════════════════════════════════════════════
    # 6. CORRELATIONS
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('6.  Correlaciones entre Métricas Operacionales', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Una correlación mide qué tan relacionadas están dos métricas entre sí, en todas las zonas. '
        'Un valor cercano a +1 significa que cuando una sube, la otra también sube. '
        'Cercano a -1 significa que van en sentido opuesto. '
        'Esto es útil para priorizar intervenciones: si dos métricas tienen correlación fuerte, '
        'mejorar una puede arrastrar a la otra sin costo adicional.', sBod))
    story.append(Spacer(1, 5))

    if ch_corr:
        ch_corr_img = img_flow(ch_corr, CW, 170)
        if ch_corr_img:
            story.append(ch_corr_img)
            story.append(Spacer(1, 4))

    if corrs:
        rows = [[Paragraph('Métrica A', sTH), Paragraph('Métrica B', sTH),
                 Paragraph('r', sTH), Paragraph('Intensidad', sTH),
                 Paragraph('Qué significa en la práctica', sTH)]]
        CORR_MEANING = {
            ('MLTV Top Verticals Adoption','Turbo Adoption'):
                'Usuarios multi-vertical también usan Turbo: activar Turbo impulsa retención global.',
            ('Non-Pro PTC > OP','Restaurants SS > ATC CVR'):
                'El checkout y la conversión en restaurantes están vinculados: mejorar UX en uno mejora el otro.',
            ('MLTV Top Verticals Adoption','Pro Adoption'):
                'Los usuarios Pro adoptan más verticales: la suscripción impulsa el uso cruzado.',
        }
        for c in corrs[:8]:
            key = (c['metric_a'], c['metric_b'])
            meaning = CORR_MEANING.get(key, CORR_MEANING.get((c['metric_b'], c['metric_a']),
                      'Relación estadística significativa — explorar causa-efecto con el equipo local.'))
            col = CG if c['correlation'] > 0 else CR
            rows.append([
                Paragraph(c['metric_a'][:26], sTDm),
                Paragraph(c['metric_b'][:26], sTDm),
                Paragraph(f"r={c['correlation']:.2f}",
                          ps('rv','Helvetica-Bold',9,col, alignment=TA_CENTER)),
                Paragraph(c['strength'], sCtr),
                Paragraph(meaning, sTDm),
            ])
        story.append(Spacer(1, 5))
        story.append(tbl(rows, [CW*0.20, CW*0.20, CW*0.10, CW*0.12, CW*0.38]))

    # Correlation heatmap
    if ch_heatmap:
        story.append(Spacer(1, 10))
        story.append(Paragraph('Matriz Completa de Correlaciones', sH3))
        story.append(Paragraph(
            'Esta matriz muestra todas las correlaciones entre métricas. '
            'Los colores verdes indican correlación positiva (suben juntas), '
            'los rojos correlación negativa (una sube cuando la otra baja). '
            'La intensidad del color indica la fuerza de la relación.', sBod))
        hm_img = img_flow(ch_heatmap, CW, 220)
        if hm_img:
            story.append(hm_img)
            story.append(Spacer(1, 4))
            story.append(interpretation(
                'Buscar celdas con colores intensos (r > 0.5 o r < -0.5). '
                'Estos pares de métricas son los que tienen mayor potencial de "efecto cascada": '
                'una intervención en una métrica impacta automáticamente la otra.'
            ))

    # Scatter plots for top correlations
    if ch_scatter_1 or ch_scatter_2:
        story.append(Spacer(1, 10))
        story.append(Paragraph('Detalle Visual — Correlaciones Más Fuertes', sH3))
        story.append(Paragraph(
            'Cada punto representa una zona. La línea punteada muestra la tendencia. '
            'Si los puntos siguen la línea de cerca, la relación es consistente en todo el mercado.', sBod))
        story.append(Spacer(1, 4))
        if ch_scatter_1:
            sc1_img = img_flow(ch_scatter_1, CW * 0.85, 180)
            if sc1_img:
                story.append(sc1_img)
                story.append(Spacer(1, 6))
        if ch_scatter_2:
            sc2_img = img_flow(ch_scatter_2, CW * 0.85, 180)
            if sc2_img:
                story.append(sc2_img)
                story.append(Spacer(1, 4))
        story.append(interpretation(
            'Una nube de puntos apretada alrededor de la línea = relación robusta y predecible. '
            'Puntos dispersos = la correlación existe pero con mucha variabilidad entre zonas. '
            'Las zonas alejadas de la tendencia son outliers que merecen investigación individual.'
        ))

    # ════════════════════════════════════════════════════════════
    # 7. OPPORTUNITIES (moved before priority zones)
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('7.  Oportunidades de Alto Impacto', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Estas zonas tienen alto volumen de órdenes semanales pero una o más métricas por debajo '
        'del benchmark del mercado. Son las oportunidades de mayor impacto potencial: '
        'el volumen ya está, solo hay que mejorar la eficiencia. '
        'Priorizá estas zonas antes de buscar crecimiento en nuevas áreas.', sBod))
    story.append(Spacer(1, 5))

    if ch_opp:
        ch_opp_img = img_flow(ch_opp, CW, 160)
        if ch_opp_img:
            story.append(ch_opp_img)
            story.append(Spacer(1, 4))
            story.append(interpretation(
                'El gráfico muestra el volumen de órdenes de cada zona. '
                'A mayor barra, mayor impacto potencial de la mejora. '
                'La tabla debajo detalla exactamente cuánto está por debajo del benchmark cada zona.'
            ))

    if opps:
        story.append(Spacer(1, 6))
        rows = [[Paragraph('Zona', sTH), Paragraph('País / Tipo', sTH),
                 Paragraph('Métrica con Gap', sTH), Paragraph('Actual vs Benchmark', sTH),
                 Paragraph('Órd/sem', sTH), Paragraph('Impacto Estimado', sTH)]]
        for o in opps:
            # Build impact string
            imp_orders = o.get('estimated_impact_orders', 0)
            imp_gp = o.get('estimated_impact_gp', 0)
            if imp_orders > 0:
                impact_str = f"~{imp_orders:,} órd. de\ncalidad adicionales"
            elif imp_gp != 0:
                impact_str = f"~{imp_gp:+,.0f} margen\nadicional/sem"
            else:
                impact_str = f"{abs(o['gap_pct']*100):.0f}% gap\nvs benchmark"
            rows.append([
                Paragraph(f"{o['zone'][:24]}", sTD),
                Paragraph(f"{o['country']}\n{o.get('zone_type','')}", sTDm),
                Paragraph(o['metric'][:22], sTDm),
                Paragraph(f"<font color='#E03000'>{o['metric_value_fmt']}</font> vs "
                          f"<font color='#16A34A'>{o['benchmark_fmt']}</font>",
                          ps('ab','Helvetica',8,CTX, alignment=TA_CENTER, leading=11)),
                Paragraph(f"{o['orders']:,}", sCtr),
                Paragraph(impact_str, ps('im','Helvetica-Bold',7.5,CO, alignment=TA_CENTER, leading=10)),
            ])
        story.append(tbl(rows, [CW*0.20, CW*0.13, CW*0.18, CW*0.19, CW*0.10, CW*0.20]))

    # ════════════════════════════════════════════════════════════
    # 8. HIGH PRIORITY ZONES
    # ════════════════════════════════════════════════════════════
    if ch_prio:
        story.append(PageBreak())
        story.append(Paragraph('8.  Desempeño por Priorización de Zona', sH2))
        story.append(hr(CR, 0.8))
        story.append(Paragraph(
            'Qué muestra: Compara el promedio de métricas clave entre zonas clasificadas como '
            'High Priority, Prioritized y Not Prioritized. '
            'Idealmente, las zonas High Priority deberían tener mejor desempeño — si no es así, '
            'hay una señal de que la priorización no se está traduciendo en resultados.', sBod))
        story.append(Spacer(1, 5))
        img = img_flow(ch_prio, CW, 160)
        if img:
            story.append(img)
            story.append(Spacer(1, 4))
            story.append(interpretation(
                '¿Qué buscar? Si las barras de "High Priority" y "Not Prioritized" son similares, '
                'significa que la estrategia de priorización no está impactando el desempeño operacional. '
                'Las brechas positivas en favor de High Priority son la señal de que el foco está funcionando.'
            ))

    # ════════════════════════════════════════════════════════════
    # 9. FUNNEL
    # ════════════════════════════════════════════════════════════
    if ch_funnel:
        story.append(PageBreak())
        story.append(Paragraph('9.  Embudo de Conversión en Restaurantes por País', sH2))
        story.append(hr(CR, 0.8))
        story.append(Paragraph(
            'Las tres etapas del funnel: '
            '1) SST→SS: Del listado de restaurantes, el usuario selecciona uno. '
            '2) SS→ATC: Dentro del restaurante, el usuario agrega algo al carrito. '
            '3) PTC→OP: El usuario completa el checkout y confirma la orden. '
            'Un país con alta conversión en etapa 1 pero baja en etapa 2 indica problema de assortment '
            '(menú poco atractivo, precios altos). Un problema en etapa 3 indica fricción en el pago.', sBod))
        story.append(Spacer(1, 5))
        img = img_flow(ch_funnel, CW, 160)
        if img:
            story.append(img)
            story.append(Spacer(1, 4))
            story.append(interpretation(
                'Comparar entre países: Si un país tiene una etapa específica del funnel '
                'consistentemente más baja, esa es la palanca de mejora. '
                'Una caída entre etapas 2 y 3 puede indicar problemas con los métodos de pago o la UX del checkout local.'
            ))

    # ════════════════════════════════════════════════════════════
    # 10. RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('10.  Recomendaciones Accionables para Esta Semana', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Derivadas directamente de los hallazgos anteriores — priorizadas por potencial de impacto inmediato.', sBod))
    story.append(Spacer(1, 6))

    if recs:
        for i, rec in enumerate(recs, 1):
            clean = re.sub(r'<[^>]+>', '', str(rec))
            rec_data = [[
                Paragraph(str(i), ps('rn','Helvetica-Bold',14,CR, alignment=TA_CENTER)),
                Paragraph(f'→  {clean}', ps('rb','Helvetica',9,CTX, leading=14))
            ]]
            rt = Table(rec_data, colWidths=[10*mm, CW - 10*mm])
            rt.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,-1),CSF),
                ('LINEBELOW',(0,0),(-1,-1),0.4,CBR),
                ('LINEAFTER',(0,0),(0,-1),2.5,CR),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('TOPPADDING',(0,0),(-1,-1),8),
                ('BOTTOMPADDING',(0,0),(-1,-1),8),
                ('LEFTPADDING',(0,0),(-1,-1),8),
                ('RIGHTPADDING',(0,0),(-1,-1),8),
            ]))
            story.append(rt)
            story.append(Spacer(1, 3))

    # ════════════════════════════════════════════════════════════
    # 11. GLOSSARY
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('11.  Glosario de Métricas', sH2))
    story.append(hr(CR, 0.8))
    story.append(Paragraph(
        'Referencia rápida de todas las métricas usadas en este reporte, '
        'con su definición y cómo interpretarlas en contexto operacional.', sBod))
    story.append(Spacer(1, 6))

    GLOSARIO = [
        ('Perfect Orders',
         'Órdenes sin cancelaciones, demoras ni defectos / Total de órdenes.',
         'Métrica de calidad de servicio. Un valor bajo (<85%) indica problemas operacionales que afectan la experiencia del usuario y aumentan los costos de soporte. Meta ideal: >90%.'),
        ('Lead Penetration',
         'Tiendas habilitadas en Rappi / (Tiendas habilitadas + prospectos identificados + tiendas que salieron).',
         'Mide el avance en la captación de merchants. Valores bajos (<15%) indican oportunidad de expansión de la red. No confundir con market share total.'),
        ('Gross Profit UE',
         'Margen bruto de ganancia / Total de órdenes (valor absoluto en moneda local).',
         'Rentabilidad por orden. Puede ser negativo si los subsidios o costos superan los ingresos. Valores negativos o cerca de cero requieren revisión de pricing y costos logísticos.'),
        ('Non-Pro PTC > OP',
         'Usuarios No-Pro que completaron una orden / Usuarios No-Pro que iniciaron el checkout.',
         'Tasa de conversión final del funnel de compra para usuarios sin suscripción Pro. Caídas indican fricción en el pago, bugs en la app o problemas con métodos de pago.'),
        ('% PRO Users Who Breakeven',
         'Usuarios Pro cuyo valor generado cubre el costo de su membresía / Total usuarios Pro.',
         'Mide la rentabilidad de la base Pro. Un valor bajo significa que Rappi está subsidiando membresías que no se usan suficientemente. Meta: maximizar esta tasa.'),
        ('MLTV Top Verticals Adoption',
         'Usuarios con órdenes en múltiples verticales (restaurantes, super, pharma, liquors) / Total usuarios.',
         'Indicador de retención y lifetime value. Usuarios multi-vertical tienen mayor frecuencia de compra y menor churn. Correlaciona fuerte con Pro Adoption.'),
        ('Pro Adoption',
         'Usuarios activos con suscripción Pro / Total usuarios de Rappi.',
         'Penetración del producto de suscripción. La suscripción Pro aumenta la frecuencia de compra y reduce el costo de adquisición a largo plazo.'),
        ('Restaurants Markdowns / GMV',
         'Descuentos totales en órdenes de restaurantes / Gross Merchandise Value de restaurantes.',
         'Mide el nivel de subsidio vía descuentos. Un ratio alto puede estar afectando el Gross Profit. Ideal: mantenerlo lo más bajo posible sin afectar la conversión.'),
        ('Restaurants SS > ATC CVR',
         'Sesiones que agregaron algo al carrito después de entrar a una tienda de restaurantes.',
         'Indica la relevancia del assortment de la tienda. Baja conversión sugiere que los precios, fotos o menú no son atractivos.'),
        ('Restaurants SST > SS CVR',
         '% de usuarios que seleccionan una tienda de la lista de restaurantes.',
         'Mide si el listing de restaurantes es atractivo. Baja conversión puede indicar pocas opciones, precios altos o malas fotos de portada.'),
        ('Retail SST > SS CVR',
         '% de usuarios que seleccionan una tienda de la lista de supermercados/retail.',
         'Mismo concepto que Restaurants SST>SS CVR pero para el vertical de supermercados. Baja conversión puede indicar problemas de assortment o cobertura geográfica.'),
        ('Turbo Adoption',
         'Usuarios que compran en Turbo / Usuarios con tiendas Turbo disponibles en su zona.',
         'Mide la penetración del servicio de entrega ultra-rápida entre quienes lo tienen disponible. Aumentar esta métrica puede incrementar el ticket promedio y diferenciación competitiva.'),
        ('WoW (Week over Week)',
         'Variación de una métrica respecto a la semana anterior.',
         'Ejemplo: un WoW de -5% en Perfect Orders significa que esta semana hubo 5 puntos porcentuales menos de órdenes perfectas que la semana pasada.'),
        ('Zona Wealthy / Non Wealthy',
         'Segmentación de zonas por nivel socioeconómico predominante.',
         'Permite comparar zonas con contextos similares. No mezclar en benchmarking: una zona Non Wealthy no debe compararse con una Wealthy para KPIs de ticket o conversión.'),
    ]

    for name, definition, interpretation_text in GLOSARIO:
        story.append(KeepTogether([
            Paragraph(name, sGlB),
            Paragraph(f'<b>Definición:</b> {definition}', sGlo),
            Paragraph(f'<b>Cómo interpretarlo:</b> {interpretation_text}', sGlo),
            Spacer(1, 3),
            hr(CBR, 0.3, 0, 4),
        ]))

    # ════════════════════════════════════════════════════════════
    # PAGE HEADER / FOOTER
    # ════════════════════════════════════════════════════════════
    def on_page(canvas, doc):
        canvas.saveState()
        # Top bar
        canvas.setFillColor(HexColor('#F8F9FA'))
        canvas.rect(0, H-20, W, 20, fill=1, stroke=0)
        canvas.setFillColor(CR)
        canvas.rect(0, H-21, W, 1.5, fill=1, stroke=0)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(HexColor('#E03000'))
        canvas.drawString(LM, H-14, 'Rappi Analytics — Reporte Ejecutivo de Operaciones')
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(HexColor('#6B7280'))
        canvas.drawRightString(W-RM, H-14, now)
        # Bottom bar
        canvas.setFillColor(HexColor('#F8F9FA'))
        canvas.rect(0, 0, W, 14, fill=1, stroke=0)
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(HexColor('#9CA3AF'))
        canvas.drawString(LM, 4, 'Datos indicativos — validar con equipos locales antes de tomar decisiones.')
        canvas.drawRightString(W-RM, 4, f'Página {doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=20*mm, bottomMargin=14*mm,
        title='Reporte Ejecutivo Rappi',
        author='Rappi Analytics SP&A',
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output_path

