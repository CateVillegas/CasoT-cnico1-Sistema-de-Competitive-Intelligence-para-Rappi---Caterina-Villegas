import os
import sys
import json
import base64
import io
import csv
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.data_loader import load_data, get_context_summary, WEEK_LABELS, is_percentage_metric
from src.analysis import run_analysis_query
from src.gemini_client import parse_query_to_analysis, generate_response, generate_chart_data
from src.insights_engine import compile_raw_insights, generate_report_with_gemini, generate_pdf_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Check API key
if not os.environ.get('GEMINI_API_KEY'):
    print("⚠️  ADVERTENCIA: Variable GEMINI_API_KEY no configurada.")
    print("    Configura: export GEMINI_API_KEY='tu-api-key'")


def render_chart_to_base64(chart_config: dict) -> str | None:
    """Render chart with matplotlib and return base64 PNG."""
    if not chart_config:
        return None
    
    try:
        RAPPI_RED = '#FF441F'
        RAPPI_ORANGE = '#FF6B35'
        BG = '#0F0F1A'
        GRID = '#2A2A3E'
        TEXT = '#E8E8F0'
        
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        
        labels = chart_config['labels']
        values = chart_config['values']
        value_labels = chart_config.get('value_labels', [])
        title = chart_config.get('title', '')
        ylabel = chart_config.get('ylabel', 'Valor')
        color = chart_config.get('color', RAPPI_RED)
        chart_type = chart_config.get('type', 'bar')
        
        # Replace None with 0 for plotting
        clean_values = [v if v is not None else 0 for v in values]
        max_val = max(abs(v) for v in clean_values) if clean_values else 1
        
        if chart_type in ['bar', 'bar_horizontal']:
            n = len(labels)
            x = np.arange(n)
            colors = [RAPPI_RED if i % 2 == 0 else RAPPI_ORANGE for i in range(n)]
            
            if chart_type == 'bar_horizontal':
                bars = ax.barh(x, clean_values, color=colors, alpha=0.9, height=0.6)
                ax.set_yticks(x)
                ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
                ax.set_xlabel(ylabel, color=TEXT, fontsize=9)
                for i, (bar, val) in enumerate(zip(bars, clean_values)):
                    lbl = value_labels[i] if i < len(value_labels) else f"{val:.1f}"
                    ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                            lbl, va='center', color=TEXT, fontsize=8.5)
            else:
                bars = ax.bar(x, clean_values, color=colors, alpha=0.9, width=0.65)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha='right', color=TEXT, fontsize=8)
                ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
                for i, (bar, val) in enumerate(zip(bars, clean_values)):
                    lbl = value_labels[i] if i < len(value_labels) else f"{val:.1f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.01,
                            lbl, ha='center', color=TEXT, fontsize=8.5, fontweight='bold')
        
        elif chart_type == 'line':
            valid_idx = [i for i, v in enumerate(values) if v is not None]
            vl = [labels[i] for i in valid_idx]
            vv = [clean_values[i] for i in valid_idx]
            vvl = [value_labels[i] for i in valid_idx] if value_labels else []
            
            x = np.arange(len(vl))
            ax.plot(x, vv, color=RAPPI_RED, linewidth=2.5, marker='o',
                    markersize=6, markerfacecolor='white', markeredgecolor=RAPPI_RED, markeredgewidth=2)
            ax.fill_between(x, vv, alpha=0.12, color=RAPPI_RED)
            ax.set_xticks(x)
            ax.set_xticklabels(vl, color=TEXT, fontsize=9)
            ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
            
            for xi, val, lbl in zip(x, vv, vvl or [str(v) for v in vv]):
                ax.annotate(lbl, (xi, val), textcoords="offset points",
                            xytext=(0, 10), ha='center', color=TEXT, fontsize=8)
        
        # Common styling
        ax.set_title(title, color=TEXT, fontsize=10.5, fontweight='bold', pad=12)
        ax.tick_params(colors=TEXT)
        ax.spines['bottom'].set_color(GRID)
        ax.spines['left'].set_color(GRID)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                    facecolor=BG, edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    except Exception as e:
        print(f"Chart error: {e}")
        import traceback; traceback.print_exc()
        return None


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    history = data.get('history', [])
    
    if not user_message:
        return jsonify({'error': 'Mensaje vacío'}), 400
    
    # Step 1: Parse the natural language query
    parsed = parse_query_to_analysis(user_message, history)
    
    if 'error' in parsed and not parsed.get('query_type'):
        return jsonify({
            'response': f"Lo siento, no pude interpretar tu pregunta. Error: {parsed['error']}\n\nIntentá ser más específico, por ejemplo: '¿Cuáles son las 5 zonas con mayor Lead Penetration?'",
            'chart': None,
            'csv_data': None
        })
    
    query_type = parsed.get('query_type')
    params = parsed.get('params', {})
    
    if not query_type:
        return jsonify({
            'response': "No pude identificar el tipo de análisis que necesitás. ¿Podés reformular la pregunta? Por ejemplo: '¿Top 5 zonas con mayor Perfect Orders en Colombia?'",
            'chart': None,
        })
    
    # Step 2: Execute analysis
    result, error = run_analysis_query(query_type, params)
    
    if error:
        return jsonify({
            'response': f"❌ No se pudo ejecutar el análisis: {error}\n\nVerificá que el nombre de la zona o métrica sea correcto.",
            'chart': None,
        })
    
    # Step 3: Generate chart data
    chart_config = generate_chart_data(result, query_type)
    chart_b64 = render_chart_to_base64(chart_config) if chart_config else None
    
    # Step 4: Generate natural language response
    response_text = generate_response(user_message, result, query_type, history)

    # Sanitize: if the assistant accidentally included a raw CSV/data dump
    # detect blocks that look like CSV (multiple lines with commas) and remove them.
    try:
        parts = response_text.split('\n\n')
        filtered_parts = []
        for p in parts:
            lines = [l for l in p.strip().splitlines() if l.strip()]
            if len(lines) >= 2:
                comma_lines = sum(1 for l in lines if ',' in l)
                # If most lines contain commas, treat as CSV-like and skip it
                if comma_lines / len(lines) > 0.6:
                    continue
            filtered_parts.append(p)
        response_text = '\n\n'.join(filtered_parts).strip()
    except Exception:
        pass

    # For top_zones responses, return a concise markdown table (title + table)
    # and omit the verbose AI analysis to avoid duplication.
    try:
        if query_type == 'top_zones' and result and isinstance(result.get('data'), list):
            rows = result['data']
            if rows:
                metric = result.get('metric', '')
                n = result.get('n', len(rows))
                header = f"**Top {n} Zonas — {metric} (Semana Actual)**"
                table_lines = ["| Zona | Ciudad | País | Valor |", "| --- | --- | ---: | ---: |"]
                for r in rows:
                    zone = r.get('zone', '')
                    city = r.get('city', '')
                    country = r.get('country', '')
                    val = r.get('value_fmt', str(r.get('value', '')))
                    table_lines.append(f"| {zone} | {city} | {country} | {val} |")

                table_md = header + "\n\n" + "\n".join(table_lines) + "\n"
                response_text = table_md
    except Exception:
        pass

    # For multivariable, show a concise table with both metrics and omit verbose AI text
    try:
        if query_type == 'multivariable' and result and isinstance(result.get('data'), list):
            rows = result['data']
            if rows:
                mh = result.get('metric_high', 'Metric A')
                ml = result.get('metric_low', 'Metric B')
                header = f"**Zonas con alto {mh} pero bajo {ml}**"
                table_lines = [f"| Zona | Ciudad | País | Tipo de Zona | {mh} | {ml} |", "| --- | --- | ---: | --- | ---: | ---: |"]
                for r in rows:
                    zone = r.get('zone', '')
                    city = r.get('city', '')
                    country = r.get('country', '')
                    zt = r.get('zone_type', '')
                    vh = r.get('val_high_fmt', str(r.get('val_high', '')))
                    vl = r.get('val_low_fmt', str(r.get('val_low', '')))
                    table_lines.append(f"| {zone} | {city} | {country} | {zt} | {vh} | {vl} |")

                table_md = header + "\n\n" + "\n".join(table_lines) + "\n"
                response_text = table_md
    except Exception:
        pass

    # For comparison queries, produce a short, clear textual summary (plus chart)
    try:
        if query_type == 'comparison' and result and isinstance(result.get('data'), dict):
            comp = result['data']
            metric = result.get('metric', '')
            labels = list(comp.keys())
            if len(labels) >= 2:
                a, b = labels[0], labels[1]
                ma = comp[a].get('mean')
                mb = comp[b].get('mean')
                ma_fmt = comp[a].get('mean_fmt', str(ma))
                mb_fmt = comp[b].get('mean_fmt', str(mb))
                count_a = comp[a].get('count', 0)
                count_b = comp[b].get('count', 0)

                # format difference
                diff = None
                diff_display = ''
                try:
                    if ma is not None and mb is not None:
                        diff = ma - mb
                        if is_percentage_metric(metric, ma) or is_percentage_metric(metric, mb):
                            diff_display = f"{diff * 100:+.1f} p.p."
                            threshold = 0.02
                        else:
                            diff_display = f"{diff:+.2f}"
                            threshold = 0.10
                except Exception:
                    diff_display = ''
                    threshold = 0.10

                country_txt = f" en {result.get('country')}" if result.get('country') else ''
                summary_lines = []
                summary_lines.append(f"**Comparación: {metric}**")
                summary_lines.append("")
                summary_lines.append(f"En{country_txt}, `{a}` tiene {ma_fmt} ({count_a} zonas) y `{b}` tiene {mb_fmt} ({count_b} zonas).")
                if diff_display:
                    summary_lines[-1] += f" Diferencia: {diff_display}."

                # interpretation: give concise, actionable explanation + next steps
                if diff is not None:
                    # direction and magnitude
                    if diff > 0:
                        direction = f"`{a}` muestra {diff_display} más que `{b}`"
                    elif diff < 0:
                        direction = f"`{b}` muestra {diff_display} más que `{a}`"
                    else:
                        direction = "No hay diferencia entre los grupos."

                    # practical impact (qualitative)
                    if abs(diff) >= threshold:
                        impact = "Impacto: diferencial operativo potencial — conviene investigar causas y oportunidades de mejora."
                    else:
                        impact = "Impacto: diferencia pequeña, probablemente no requiera acción inmediata salvo monitoreo."

                    # suggested next steps
                    steps = (
                        "Sugerencias: 1) Revisar las métricas de cumplimiento y tiempos de entrega en las zonas con peor Perfect Orders; "
                        "2) Inspeccionar mix y calidad de merchants en esos grupos; 3) Revisar promociones/operaciones locales que afecten cancelaciones."
                    )

                    summary_lines.append(direction + ".")
                    summary_lines.append(impact)
                    summary_lines.append(steps)

                # examples (top zones) if available
                try:
                    a_examples = comp[a].get('zones_sample', [])[:3]
                    b_examples = comp[b].get('zones_sample', [])[:3]
                    if a_examples:
                        names = ', '.join([z.get('ZONE') or z.get('zone') or str(z) for z in a_examples])
                        summary_lines.append(f"Ejemplos {a}: {names}.")
                    if b_examples:
                        names = ', '.join([z.get('ZONE') or z.get('zone') or str(z) for z in b_examples])
                        summary_lines.append(f"Ejemplos {b}: {names}.")
                except Exception:
                    pass

                response_text = '\n'.join(summary_lines)
    except Exception:
        pass
    
    # Step 5: Prepare CSV export data
    csv_data = None
    if result and result.get('data'):
        try:
            import csv as csv_module
            output = io.StringIO()
            rows = result['data']
            if rows and isinstance(rows[0], dict):
                writer = csv_module.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                csv_data = output.getvalue()
        except Exception:
            pass
    
    return jsonify({
        'response': response_text,
        'chart': chart_b64,
        'csv_data': csv_data,
        'query_type': query_type,
        'analysis_explanation': parsed.get('explanation', ''),
    })


@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    data = request.json
    csv_data = data.get('csv_data', '')
    filename = data.get('filename', 'rappi_analisis.csv')
    
    buf = io.BytesIO(csv_data.encode('utf-8'))
    return send_file(buf, mimetype='text/csv',
                    as_attachment=True, download_name=filename)


_report_status = {'status': 'idle', 'progress': '', 'path': None, 'content': None}

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    def _run():
        _report_status['status'] = 'running'
        _report_status['progress'] = 'Analizando datos...'
        
        try:
            raw = compile_raw_insights()
            _report_status['progress'] = 'Generando reporte con IA...'
            
            md_content = generate_report_with_gemini(raw)
            _report_status['content'] = md_content
            _report_status['progress'] = 'Generando PDF...'
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = os.path.join(REPORTS_DIR, f'rappi_insights_{timestamp}.pdf')
            generate_pdf_report(md_content, pdf_path)
            
            _report_status['status'] = 'done'
            _report_status['path'] = pdf_path
            _report_status['progress'] = 'Listo'
        except Exception as e:
            _report_status['status'] = 'error'
            _report_status['progress'] = str(e)
    
    if _report_status['status'] == 'running':
        return jsonify({'status': 'running', 'progress': _report_status['progress']})
    
    _report_status['status'] = 'idle'
    t = threading.Thread(target=_run)
    t.daemon = True
    t.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/report-status', methods=['GET'])
def report_status():
    return jsonify({
        'status': _report_status['status'],
        'progress': _report_status['progress'],
        'has_content': _report_status['content'] is not None
    })


@app.route('/api/download-report', methods=['GET'])
def download_report():
    if _report_status.get('path') and os.path.exists(_report_status['path']):
        return send_file(_report_status['path'], as_attachment=True,
                        download_name='Rappi_Reporte_Ejecutivo.pdf')
    return jsonify({'error': 'Reporte no disponible'}), 404


@app.route('/api/report-preview', methods=['GET'])
def report_preview():
    if _report_status.get('content'):
        return jsonify({'content': _report_status['content']})
    return jsonify({'error': 'No hay reporte generado'}), 404


@app.route('/api/context', methods=['GET'])
def context():
    return jsonify(get_context_summary())


@app.route('/api/health', methods=['GET'])
def health():
    api_key_set = bool(os.environ.get('GEMINI_API_KEY'))
    _, orders_df, _ = load_data()
    return jsonify({
        'status': 'ok',
        'api_key_configured': api_key_set,
        'data_loaded': True,
    })


if __name__ == '__main__':
    print("🚀 Rappi Analytics — Sistema de Análisis Inteligente")
    print("=" * 50)
    if not os.environ.get('GEMINI_API_KEY'):
        print("⚠️  Configura tu API key: export GEMINI_API_KEY='tu-key'")
    print("🌐 Iniciando en http://localhost:5001")
    app.run(debug=False, host='0.0.0.0', port=5001)
