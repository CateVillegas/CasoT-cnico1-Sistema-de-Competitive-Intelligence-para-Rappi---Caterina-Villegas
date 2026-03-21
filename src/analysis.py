import pandas as pd
import numpy as np
import json
from .data_loader import (load_data, WEEK_COLS_METRICS, WEEK_LABELS,
                           COUNTRY_NAMES, METRIC_DESCRIPTIONS,
                           format_metric_value, is_percentage_metric)


def safe_pct(metric, val):
    """Format value using metric-aware formatting."""
    return format_metric_value(metric, val)


def _filter_df(df, country=None, city=None, zone=None, zone_type=None, metric=None):
    mask = pd.Series([True] * len(df), index=df.index)
    if country:
        c_upper = country.upper()
        mask &= df['COUNTRY'].str.upper() == c_upper
    if city:
        mask &= df['CITY'].str.lower().str.contains(city.lower(), na=False)
    if zone:
        mask &= df['ZONE'].str.lower().str.contains(zone.lower(), na=False)
    if zone_type:
        mask &= df['ZONE_TYPE'].str.lower().str.contains(zone_type.lower(), na=False)
    if metric:
        mask &= df['METRIC'].str.lower().str.contains(metric.lower(), na=False)
    return df[mask]


def top_zones_by_metric(metric, n=5, ascending=False, country=None, city=None, zone_type=None):
    metrics_df, orders_df, _ = load_data()
    if metric.lower() == 'orders':
        df = orders_df.copy()
    else:
        df = _filter_df(metrics_df, country=country, city=city, zone_type=zone_type, metric=metric)
    
    if df.empty:
        return None, "No se encontraron datos para el filtro especificado."
    
    df = df.copy()
    df['current_value'] = df['L0W_ROLL']
    df = df.dropna(subset=['current_value'])
    df = df.sort_values('current_value', ascending=ascending).head(n)
    
    results = []
    for _, row in df.iterrows():
        results.append({
            'zone': row['ZONE'],
            'city': row['CITY'],
            'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
            'zone_type': row['ZONE_TYPE'],
            'value': row['current_value'],
            'value_fmt': format_metric_value(metric, row['current_value']),
        })
    
    meta = {
        'metric': metric,
        'metric_desc': METRIC_DESCRIPTIONS.get(metric, metric),
        'direction': 'mayor' if not ascending else 'menor',
        'n': n,
        'type': 'top_zones',
        'data': results
    }
    return meta, None


def compare_zones_by_type(metric, zone_type_a='Wealthy', zone_type_b='Non Wealthy', country=None):
    metrics_df, _, _ = load_data()
    df = _filter_df(metrics_df, country=country, metric=metric)
    
    if df.empty:
        return None, "No se encontraron datos."

    results = {}
    for zt in [zone_type_a, zone_type_b]:
        sub = df[df['ZONE_TYPE'] == zt]
        if not sub.empty:
            vals = sub['L0W_ROLL'].dropna()
            prev_vals = sub['L1W_ROLL'].dropna()
            results[zt] = {
                'mean': vals.mean(),
                'mean_fmt': format_metric_value(metric, vals.mean()),
                'median': vals.median(),
                'median_fmt': format_metric_value(metric, vals.median()),
                'count': len(vals),
                'prev_mean': prev_vals.mean(),
                'prev_mean_fmt': format_metric_value(metric, prev_vals.mean()),
                'zones_sample': sub.nlargest(3, 'L0W_ROLL')[['ZONE', 'CITY', 'L0W_ROLL']].to_dict('records')
            }

    meta = {
        'metric': metric,
        'metric_desc': METRIC_DESCRIPTIONS.get(metric, metric),
        'country': COUNTRY_NAMES.get(country, country) if country else 'Todos los países',
        'type': 'comparison',
        'data': results
    }
    return meta, None


def zone_trend(zone_name, metric, weeks=8):
    metrics_df, orders_df, _ = load_data()
    
    if metric.lower() == 'orders':
        df = orders_df.copy()
    else:
        df = metrics_df.copy()
    
    df_zone = df[df['ZONE'].str.lower().str.contains(zone_name.lower(), na=False)]
    if metric.lower() != 'orders':
        df_zone = df_zone[df_zone['METRIC'].str.lower().str.contains(metric.lower(), na=False)]
    
    if df_zone.empty:
        return None, f"No se encontró la zona '{zone_name}' con la métrica '{metric}'."
    
    row = df_zone.iloc[0]
    week_cols = WEEK_COLS_METRICS[-weeks:]
    labels = WEEK_LABELS[-weeks:]
    
    values = [row[c] if not pd.isna(row[c]) else None for c in week_cols]
    
    meta = {
        'zone': row['ZONE'],
        'city': row['CITY'],
        'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
        'metric': metric,
        'metric_desc': METRIC_DESCRIPTIONS.get(metric, metric),
        'type': 'trend',
        'labels': labels,
        'values': values,
        'data': list(zip(labels, [format_metric_value(metric, v) if v is not None else 'N/A' for v in values]))
    }
    return meta, None


def avg_by_country(metric):
    metrics_df, orders_df, _ = load_data()
    if metric.lower() == 'orders':
        df = orders_df.copy()
    else:
        df = _filter_df(metrics_df, metric=metric)
    
    if df.empty:
        return None, "No se encontraron datos."
    
    results = []
    for country, grp in df.groupby('COUNTRY'):
        vals = grp['L0W_ROLL'].dropna()
        if len(vals) > 0:
            results.append({
                'country_code': country,
                'country': COUNTRY_NAMES.get(country, country),
                'avg': vals.mean(),
                'avg_fmt': format_metric_value(metric, vals.mean()),
                'zones_count': len(vals),
                'median': vals.median(),
                'median_fmt': format_metric_value(metric, vals.median()),
            })
    results.sort(key=lambda x: x['avg'], reverse=True)
    
    meta = {
        'metric': metric,
        'metric_desc': METRIC_DESCRIPTIONS.get(metric, metric),
        'type': 'avg_by_country',
        'data': results
    }
    return meta, None


def multivariable_analysis(metric_high, metric_low, high_threshold_pct=0.6, low_threshold_pct=0.4, country=None):
    metrics_df, _, _ = load_data()
    
    df_high = _filter_df(metrics_df, country=country, metric=metric_high)[['COUNTRY', 'CITY', 'ZONE', 'ZONE_TYPE', 'L0W_ROLL']].copy()
    df_low = _filter_df(metrics_df, country=country, metric=metric_low)[['COUNTRY', 'CITY', 'ZONE', 'ZONE_TYPE', 'L0W_ROLL']].copy()
    
    df_high = df_high.rename(columns={'L0W_ROLL': 'val_high'})
    df_low = df_low.rename(columns={'L0W_ROLL': 'val_low'})
    
    merged = df_high.merge(df_low, on=['COUNTRY', 'CITY', 'ZONE', 'ZONE_TYPE'])
    merged = merged.dropna(subset=['val_high', 'val_low'])
    
    if merged.empty:
        return None, "No se pudo cruzar las métricas especificadas."
    
    high_thresh = merged['val_high'].quantile(high_threshold_pct)
    low_thresh = merged['val_low'].quantile(low_threshold_pct)
    
    filtered = merged[(merged['val_high'] >= high_thresh) & (merged['val_low'] <= low_thresh)]
    filtered = filtered.sort_values('val_high', ascending=False).head(10)
    
    results = []
    for _, row in filtered.iterrows():
        results.append({
            'zone': row['ZONE'],
            'city': row['CITY'],
            'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
            'zone_type': row['ZONE_TYPE'],
            'val_high': row['val_high'],
            'val_high_fmt': format_metric_value(metric_high, row['val_high']),
            'val_low': row['val_low'],
            'val_low_fmt': format_metric_value(metric_low, row['val_low']),
        })
    
    meta = {
        'metric_high': metric_high,
        'metric_low': metric_low,
        'type': 'multivariable',
        'threshold_high': format_metric_value(metric_high, high_thresh),
        'threshold_low': format_metric_value(metric_low, low_thresh),
        'data': results
    }
    return meta, None


def growth_leaders(n=5, weeks=5):
    _, orders_df, _ = load_data()
    
    week_cols = WEEK_COLS_METRICS[-weeks:]
    df = orders_df.dropna(subset=['L0W_ROLL', week_cols[0]])
    
    df = df.copy()
    df['growth_abs'] = df['L0W_ROLL'] - df[week_cols[0]]
    df['growth_pct'] = (df['L0W_ROLL'] - df[week_cols[0]]) / df[week_cols[0]].abs()
    df = df.dropna(subset=['growth_pct'])
    df = df.sort_values('growth_pct', ascending=False).head(n)
    
    results = []
    for _, row in df.iterrows():
        weekly = [row[c] for c in week_cols]
        results.append({
            'zone': row['ZONE'],
            'city': row['CITY'],
            'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
            'growth_pct': row['growth_pct'],
            'growth_pct_fmt': f"+{row['growth_pct']*100:.1f}%",
            'start_orders': int(row[week_cols[0]]) if not pd.isna(row[week_cols[0]]) else 0,
            'end_orders': int(row['L0W_ROLL']) if not pd.isna(row['L0W_ROLL']) else 0,
            'weekly': [int(v) if not pd.isna(v) else 0 for v in weekly],
        })
    
    meta = {
        'type': 'growth_leaders',
        'weeks': weeks,
        'labels': WEEK_LABELS[-weeks:],
        'data': results
    }
    return meta, None


def run_analysis_query(query_type, params):
    """Dispatch analysis based on query type."""
    if query_type == 'top_zones':
        return top_zones_by_metric(**params)
    elif query_type == 'comparison':
        return compare_zones_by_type(**params)
    elif query_type == 'trend':
        return zone_trend(**params)
    elif query_type == 'avg_by_country':
        return avg_by_country(**params)
    elif query_type == 'multivariable':
        return multivariable_analysis(**params)
    elif query_type == 'growth_leaders':
        return growth_leaders(**params)
    else:
        return None, "Tipo de análisis no reconocido."
