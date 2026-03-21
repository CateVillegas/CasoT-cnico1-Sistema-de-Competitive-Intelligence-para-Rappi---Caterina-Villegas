import pandas as pd
import numpy as np
import json
import requests
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from datetime import datetime
from .data_loader import load_data, WEEK_COLS_METRICS, WEEK_LABELS, COUNTRY_NAMES, METRIC_DESCRIPTIONS

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_URL = os.environ.get('GEMINI_URL', "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")


def detect_anomalies(df, threshold=0.10):
    """Detect zones with drastic week-over-week changes."""
    anomalies = []
    df = df.dropna(subset=['L0W_ROLL', 'L1W_ROLL']).copy()
    # Drop duplicate zone+metric rows, keep first
    df = df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE', 'METRIC'])
    
    # For metrics where value can cross zero, use absolute change instead
    df['wow_change_abs'] = df['L0W_ROLL'] - df['L1W_ROLL']
    
    # Use pct change only when previous value is meaningfully non-zero
    df = df[df['L1W_ROLL'].abs() > 0.001].copy()
    df['wow_change'] = df['wow_change_abs'] / df['L1W_ROLL'].abs()
    
    # For Gross Profit UE use absolute change threshold (not %)
    from .data_loader import NUMERIC_METRICS
    result_rows = []
    for _, row in df.iterrows():
        if row['METRIC'] in NUMERIC_METRICS:
            # Use absolute change > 1.0 as threshold for numeric metrics
            if abs(row['wow_change_abs']) > 1.0:
                result_rows.append(row)
        else:
            if abs(row['wow_change']) >= threshold:
                result_rows.append(row)
    
    for row in result_rows:
        if row['METRIC'] in NUMERIC_METRICS:
            change_display = f"{row['wow_change_abs']:+.2f}"
            change_pct_val = row['wow_change_abs'] / max(abs(row['L1W_ROLL']), 0.01)
        else:
            change_display = f"{row['wow_change']*100:+.1f}%"
            change_pct_val = row['wow_change']
        
        anomalies.append({
            'zone': row['ZONE'],
            'city': row['CITY'],
            'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
            'metric': row['METRIC'],
            'current': row['L0W_ROLL'],
            'previous': row['L1W_ROLL'],
            'change_pct': change_pct_val,
            'change_display': change_display,
            'direction': 'mejora' if change_pct_val > 0 else 'deterioro',
        })
    
    anomalies.sort(key=lambda x: abs(x['change_pct']), reverse=True)
    return anomalies[:20]


def detect_consistent_trends(df, n_weeks=3):
    """Detect metrics declining consistently for 3+ weeks."""
    trending = []
    week_cols = WEEK_COLS_METRICS[-4:]  # Last 4 weeks
    df = df.dropna(subset=week_cols).copy()
    df = df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE', 'METRIC'])
    
    for _, row in df.iterrows():
        vals = [row[c] for c in week_cols]
        if all(v is not None and not pd.isna(v) for v in vals):
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            if all(d < 0 for d in diffs[-n_weeks:]):
                base = abs(vals[0]) if vals[0] != 0 else 0.001
                total_change = (vals[-1] - vals[0]) / base
                # Cap extreme changes for display
                total_change = max(min(total_change, 5.0), -5.0)
                trending.append({
                    'zone': row['ZONE'],
                    'city': row['CITY'],
                    'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
                    'metric': row['METRIC'],
                    'values': [round(v, 4) for v in vals],
                    'total_change_pct': total_change,
                    'weeks_declining': n_weeks,
                })
    
    trending.sort(key=lambda x: x['total_change_pct'])
    return trending[:15]


def detect_benchmarking_gaps(df):
    """Compare zones within same country/type with divergent performance."""
    gaps = []
    key_metrics = ['Perfect Orders', 'Lead Penetration', 'Gross Profit UE']
    
    for metric in key_metrics:
        metric_df = df[df['METRIC'] == metric].dropna(subset=['L0W_ROLL'])
        
        for (country, zone_type), grp in metric_df.groupby(['COUNTRY', 'ZONE_TYPE']):
            if len(grp) < 3:
                continue
            q75 = grp['L0W_ROLL'].quantile(0.75)
            q25 = grp['L0W_ROLL'].quantile(0.25)
            gap = q75 - q25
            
            if gap > 0.1:
                top_zone = grp.nlargest(1, 'L0W_ROLL').iloc[0]
                bot_zone = grp.nsmallest(1, 'L0W_ROLL').iloc[0]
                gaps.append({
                    'metric': metric,
                    'country': COUNTRY_NAMES.get(country, country),
                    'zone_type': zone_type,
                    'gap': gap,
                    'top_zone': top_zone['ZONE'],
                    'top_value': top_zone['L0W_ROLL'],
                    'bottom_zone': bot_zone['ZONE'],
                    'bottom_value': bot_zone['L0W_ROLL'],
                    'zones_count': len(grp),
                })
    
    gaps.sort(key=lambda x: x['gap'], reverse=True)
    return gaps[:10]


def detect_correlations(df):
    """Find metrics that tend to move together."""
    pivot = df.pivot_table(index=['COUNTRY', 'CITY', 'ZONE'], columns='METRIC', values='L0W_ROLL')
    pivot = pivot.dropna(thresh=4)
    
    correlations = []
    cols = pivot.columns.tolist()
    
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            valid = pivot[[cols[i], cols[j]]].dropna()
            if len(valid) > 10:
                corr = valid[cols[i]].corr(valid[cols[j]])
                if abs(corr) > 0.4:
                    correlations.append({
                        'metric_a': cols[i],
                        'metric_b': cols[j],
                        'correlation': corr,
                        'strength': 'fuerte' if abs(corr) > 0.7 else 'moderada',
                        'direction': 'positiva' if corr > 0 else 'negativa',
                        'n_zones': len(valid),
                    })
    
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return correlations[:8]


def detect_opportunities(df, orders_df):
    """Find zones with high orders but underperforming metrics."""
    opportunities = []
    
    order_current = orders_df[['COUNTRY', 'CITY', 'ZONE', 'L0W_ROLL']].copy()
    order_current = order_current.rename(columns={'L0W_ROLL': 'orders'})
    order_current = order_current.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE'])
    order_percentile = order_current['orders'].quantile(0.7)
    high_order_zones = order_current[order_current['orders'] >= order_percentile]
    
    key_metrics = ['Perfect Orders', 'Lead Penetration', 'Gross Profit UE']
    seen = set()
    
    for metric in key_metrics:
        metric_df = df[df['METRIC'] == metric][['COUNTRY', 'CITY', 'ZONE', 'ZONE_TYPE', 'L0W_ROLL']].copy()
        metric_df = metric_df.drop_duplicates(subset=['COUNTRY', 'CITY', 'ZONE'])
        merged = high_order_zones.merge(metric_df, on=['COUNTRY', 'CITY', 'ZONE'])
        merged = merged.dropna(subset=['L0W_ROLL'])
        
        overall_median = df[df['METRIC'] == metric]['L0W_ROLL'].median()
        
        underperforming = merged[merged['L0W_ROLL'] < overall_median * 0.9]
        underperforming = underperforming.sort_values('orders', ascending=False).head(5)
        
        for _, row in underperforming.iterrows():
            key = (row['ZONE'], row['COUNTRY'], metric)
            if key in seen:
                continue
            seen.add(key)
            opportunities.append({
                'zone': row['ZONE'],
                'city': row['CITY'],
                'country': COUNTRY_NAMES.get(row['COUNTRY'], row['COUNTRY']),
                'zone_type': row.get('ZONE_TYPE', 'N/A'),
                'metric': metric,
                'metric_value': row['L0W_ROLL'],
                'orders': row['orders'],
                'median_benchmark': overall_median,
                'gap': overall_median - row['L0W_ROLL'],
            })
    
    opportunities.sort(key=lambda x: x['orders'], reverse=True)
    return opportunities[:10]


def compile_raw_insights():
    metrics_df, orders_df, _ = load_data()
    
    return {
        'anomalies': detect_anomalies(metrics_df),
        'consistent_trends': detect_consistent_trends(metrics_df),
        'benchmarking_gaps': detect_benchmarking_gaps(metrics_df),
        'correlations': detect_correlations(metrics_df),
        'opportunities': detect_opportunities(metrics_df, orders_df),
    }


def generate_report_with_gemini(raw_insights: dict) -> str:
    """Use Gemini to generate the executive report in markdown."""
    
    summary = json.dumps(raw_insights, ensure_ascii=False, default=str, indent=2)
    
    prompt = f"""Eres el analista senior de Rappi. Con base en los insights automáticos detectados en los datos operacionales, genera un REPORTE EJECUTIVO COMPLETO en español.

DATOS DE INSIGHTS DETECTADOS:
{summary[:8000]}

ESTRUCTURA DEL REPORTE (usa markdown):
# 🚀 Reporte Ejecutivo de Operaciones — Rappi
*Generado automáticamente | {datetime.now().strftime('%d/%m/%Y %H:%M')}*

---
## 📋 Resumen Ejecutivo
[3-5 hallazgos críticos en bullets con impacto en el negocio]

---
## ⚠️ Anomalías Detectadas
[Top anomalías: cambios bruscos semana a semana. Para cada una: zona, métrica, cambio %, qué podría significar para el negocio]

---
## 📉 Tendencias Preocupantes
[Métricas en deterioro consistente 3+ semanas. Menciona zonas específicas y la magnitud del deterioro]

---
## 📊 Benchmarking: Brechas entre Zonas Similares
[Zonas del mismo país/tipo con performance muy diferente. Identifica las más llamativas]

---
## 🔗 Correlaciones entre Métricas
[Relaciones estadísticas encontradas. Explica el impacto operacional de cada correlación]

---
## 💎 Oportunidades de Mejora
[Zonas con alto volumen de órdenes pero métricas debajo del benchmark. Estas tienen el mayor potencial de impacto]

---
## ✅ Recomendaciones Accionables
[5-7 acciones concretas y específicas que el equipo de Ops/SP&A puede tomar esta semana]

---
*Nota: Este reporte fue generado automáticamente. Los datos son indicativos y deben validarse con los equipos locales.*

IMPORTANTE:
- Sé específico con nombres de zonas, países y métricas
- Cuantifica cuando sea posible (ej: "deterioro del 23%")
- Las recomendaciones deben ser concretas y accionables para un equipo de operaciones
- Usa el contexto de negocio de Rappi (marketplace, merchants, conversión, etc.)
- Escribe para un Director de Operaciones o VP de SP&A
"""
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        }
    }
    
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error generando reporte: {str(e)}"


def generate_pdf_report(markdown_content: str, output_path: str) -> str:
    """Convert markdown report to PDF using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    import re
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.5*cm, rightMargin=2.5*cm
    )
    
    RAPPI_RED = HexColor('#FF441F')
    RAPPI_DARK = HexColor('#1A1A2E')
    LIGHT_GRAY = HexColor('#F5F5F5')
    MID_GRAY = HexColor('#888888')
    
    styles = getSampleStyleSheet()
    
    style_h1 = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=22,
                               textColor=RAPPI_RED, spaceAfter=6, spaceBefore=0)
    style_h2 = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=14,
                               textColor=RAPPI_DARK, spaceAfter=4, spaceBefore=14,
                               borderPad=4)
    style_h3 = ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=11,
                               textColor=RAPPI_DARK, spaceAfter=3, spaceBefore=8)
    style_body = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                                 leading=14, textColor=HexColor('#333333'), spaceAfter=4)
    style_bullet = ParagraphStyle('bullet', fontName='Helvetica', fontSize=9.5,
                                   leading=14, textColor=HexColor('#333333'),
                                   leftIndent=16, spaceAfter=3)
    style_italic = ParagraphStyle('italic', fontName='Helvetica-Oblique', fontSize=8.5,
                                   textColor=MID_GRAY, spaceAfter=8)
    
    story = []
    
    lines = markdown_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        
        # Clean markdown bold/italic for PDF
        line_clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        line_clean = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line_clean)
        line_clean = re.sub(r'`(.*?)`', r'\1', line_clean)
        line_clean = line_clean.replace('&', '&amp;').replace('<b>', '<b>').replace('</b>', '</b>')
        
        if line.startswith('# '):
            text = line[2:].replace('&', '&amp;')
            story.append(Paragraph(text, style_h1))
            story.append(HRFlowable(width="100%", thickness=2, color=RAPPI_RED, spaceAfter=8))
        elif line.startswith('## '):
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line[3:]).replace('&', '&amp;')
            story.append(Paragraph(text, style_h2))
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#DDDDDD'), spaceAfter=4))
        elif line.startswith('### '):
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line[4:]).replace('&', '&amp;')
            story.append(Paragraph(text, style_h3))
        elif line.startswith('- ') or line.startswith('* '):
            text = '• ' + re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line[2:]).replace('&', '&amp;')
            story.append(Paragraph(text, style_bullet))
        elif line.startswith('---'):
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#EEEEEE'), spaceAfter=6, spaceBefore=6))
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            text = line.strip('*').replace('&', '&amp;')
            story.append(Paragraph(text, style_italic))
        elif line.startswith('|'):
            continue  # Skip markdown tables (handled separately or left as text)
        else:
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line).replace('&', '&amp;')
            if text:
                story.append(Paragraph(text, style_body))
    
    doc.build(story)
    return output_path
