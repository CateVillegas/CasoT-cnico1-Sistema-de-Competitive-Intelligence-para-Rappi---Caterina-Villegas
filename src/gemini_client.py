import os
import json
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, fall back to environment variables only
    pass
from .data_loader import is_percentage_metric

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
# Allow overriding the full Gemini endpoint via env (useful if API surface changes)
GEMINI_URL = os.environ.get('GEMINI_URL', "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")

SYSTEM_CONTEXT = """Eres un asistente analítico para los equipos de Strategy, Planning & Analytics (SP&A) y Operations de Rappi.

Tu trabajo es dos cosas:
1. INTERPRETAR la pregunta del usuario y devolver un JSON estructurado indicando qué análisis ejecutar sobre los datos.
2. Cuando recibes resultados de datos, GENERAR una respuesta clara, ejecutiva y accionable.

=== MÉTRICAS DISPONIBLES (nombres exactos) ===
- "Lead Penetration": Tiendas habilitadas en Rappi / total prospectos
- "Perfect Orders": Órdenes sin defectos ni demoras / total órdenes  
- "Gross Profit UE": Margen bruto por orden
- "% PRO Users Who Breakeven": Usuarios Pro rentables / total Pro
- "% Restaurants Sessions With Optimal Assortment": Sesiones con ≥40 restaurantes
- "MLTV Top Verticals Adoption": Adopción de múltiples verticales
- "Non-Pro PTC > OP": Conversión usuarios No-Pro a orden
- "Pro Adoption (Last Week Status)": % usuarios con suscripción Pro
- "Restaurants Markdowns / GMV": Descuentos restaurantes / GMV
- "Restaurants SS > ATC CVR": Conversión Select Store → Add to Cart en restaurantes
- "Restaurants SST > SS CVR": Conversión ver lista → seleccionar tienda en restaurantes
- "Retail SST > SS CVR": Conversión ver lista → seleccionar tienda en retail/super
- "Turbo Adoption": Usuarios usando Turbo / total con Turbo disponible
- "Orders": Volumen de órdenes (dataset separado)

=== PAÍSES (códigos) ===
AR=Argentina, BR=Brasil, CL=Chile, CO=Colombia, CR=Costa Rica, EC=Ecuador, MX=México, PE=Perú, UY=Uruguay

=== TIPOS DE ANÁLISIS QUE PUEDES RETORNAR ===

1. top_zones: zonas con mayor/menor valor de una métrica
   params: metric (str), n (int, default 5), ascending (bool, default false), country (str opcional, código 2 letras), city (str opcional), zone_type (str opcional: "Wealthy" o "Non Wealthy")

2. comparison: comparar métricas entre tipos de zona (Wealthy vs Non Wealthy)
   params: metric (str), zone_type_a (str), zone_type_b (str), country (str opcional)

3. trend: evolución temporal de una métrica en una zona específica
   params: zone_name (str), metric (str), weeks (int, default 8)

4. avg_by_country: promedio de una métrica por país
   params: metric (str)

5. multivariable: zonas con alto valor en métrica A y bajo en métrica B
   params: metric_high (str), metric_low (str), country (str opcional)

6. growth_leaders: zonas con mayor crecimiento en órdenes
   params: n (int, default 5), weeks (int, default 5)

=== REGLAS ===
- Siempre retorna JSON válido con estructura: {"query_type": "...", "params": {...}, "explanation": "..."}
- "explanation" es una frase de 1 línea que explica qué se va a buscar (en español)
- Si la pregunta menciona "zonas problemáticas", infiere métricas deterioradas (Perfect Orders bajo, Lead Penetration bajo)
- Si hay ambigüedad en el nombre de métrica, elige la más probable por contexto
- Para nombres de ciudades/zonas, usa el texto tal como lo mencionó el usuario
- Cuando el usuario pide "últimas N semanas" en una tendencia, usa ese N en el parámetro weeks
"""

def parse_query_to_analysis(user_message: str, conversation_history: list) -> dict:
    """Use Gemini to parse natural language query into analysis parameters."""
    
    messages = []
    for msg in conversation_history[-6:]:
        role = "user" if msg['role'] == 'user' else "model"
        messages.append({
            "role": role,
            "parts": [{"text": msg['content']}]
        })
    
    messages.append({
        "role": "user",
        "parts": [{"text": f"INSTRUCCIÓN: Analiza esta pregunta y retorna SOLO un JSON válido con query_type, params y explanation. No incluyas markdown ni texto adicional.\n\nPREGUNTA: {user_message}"}]
    })
    
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_CONTEXT}]},
        "contents": messages,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json"
        }
    }
    
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        text = data['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text.strip())
    except Exception as e:
        return {"error": str(e), "query_type": None}


RESPONSE_SYSTEM = """Eres un analista experto de Rappi que comunica insights de datos a equipos de Strategy, Planning & Analytics (SP&A) y Operations.

Tu audiencia son gerentes y directores de operaciones, no técnicos de datos. Comunicas con claridad ejecutiva.

REGLAS DE COMUNICACIÓN:
- Respuestas claras, directas y accionables
- Usa contexto de negocio (Lead Penetration baja = riesgo de perder merchants, Perfect Orders baja = mala experiencia de usuario)
- Destaca los hallazgos más importantes primero
- Cuando hay tablas de datos, interprétalos: ¿qué significa para el negocio?
- Usa formato markdown para estructurar bien (encabezados ##, **negrita**, tablas con |)
- Cierra SIEMPRE con una sección "💡 **Sugerencia de siguiente análisis:**" con 1-2 preguntas de seguimiento relevantes basadas en lo que se acaba de analizar
- Sé conciso: máximo 300 palabras en el cuerpo, más la tabla de datos si aplica
- Si la métrica es un porcentaje (0-100%), interpreta: <30% es crítico, 30-50% es bajo, 50-70% es medio, >70% es bueno (adapta según contexto de negocio de Rappi)
- Cuando hay variación semana a semana >10%, destácalo como alerta

FORMATO DE TABLA MARKDOWN:
| Zona | Ciudad | País | Valor |
|------|--------|------|-------|
| ... | ... | ... | ... |
"""

def generate_response(user_message: str, analysis_result: dict, analysis_type: str, conversation_history: list) -> str:
    """Use Gemini to generate a natural language response from analysis data."""
    
    data_summary = json.dumps(analysis_result, ensure_ascii=False, indent=2)
    
    prompt = f"""El usuario preguntó: "{user_message}"

Se ejecutó el análisis de tipo "{analysis_type}" y estos son los resultados:

{data_summary}

Genera una respuesta ejecutiva clara para el equipo de SP&A y Operations de Rappi. 
Incluye una tabla formateada con los datos si hay múltiples zonas/países.
Termina con la sección de sugerencia de siguiente análisis."""
    # Important: request the model NOT to dump raw data tables or CSV output.
    # We provide the data via the UI Export CSV button; the assistant should only
    # include concise summaries, highlights, and at most a 3-row example table.
    prompt += "\n\nINSTRUCCIÓN IMPORTANTE: No incluyas ni pegues las filas de datos completas ni formatos CSV en la respuesta. Resume hallazgos, destaca anomalías y, si acaso, muestra como máximo 3 filas de ejemplo en una tabla resumida. Para obtener los datos completos, usa la opción 'Exportar CSV'."

    messages = [{"role": "user", "parts": [{"text": prompt}]}]
    
    payload = {
        "system_instruction": {"parts": [{"text": RESPONSE_SYSTEM}]},
        "contents": messages,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error generando respuesta: {str(e)}"


def _chart_val(v, metric):
    """Scale value for chart display: ratios → ×100 for %, others → raw."""
    from .data_loader import is_percentage_metric
    if v is None:
        return None
    if is_percentage_metric(metric, v):
        return v * 100
    return v


def generate_chart_data(analysis_result: dict, analysis_type: str) -> dict | None:
    """Return chart configuration based on analysis type."""
    
    if analysis_type == 'top_zones':
        data = analysis_result.get('data', [])
        if not data:
            return None
        metric = analysis_result.get('metric', '')
        return {
            'type': 'bar',
            'labels': [f"{d['zone'][:25]}\n({d['country']})" for d in data],
            'values': [_chart_val(d['value'], metric) for d in data],
            'value_labels': [d['value_fmt'] for d in data],
            'title': f"Top Zonas — {metric}",
            'ylabel': '%' if any(is_percentage_metric(metric, d['value']) for d in data) else 'Valor',
            'color': '#FF441F',
        }
    
    elif analysis_type == 'trend':
        labels = analysis_result.get('labels', [])
        values = analysis_result.get('values', [])
        metric = analysis_result.get('metric', '')
        if not values:
            return None
        return {
            'type': 'line',
            'labels': labels,
            'values': [_chart_val(v, metric) for v in values],
            'value_labels': [analysis_result['data'][i][1] if i < len(analysis_result.get('data', [])) else '' for i in range(len(values))],
            'title': f"Tendencia: {metric}\n{analysis_result.get('zone', '')} · {analysis_result.get('city', '')}",
            'ylabel': '%' if any(is_percentage_metric(metric, v) for v in values if v is not None) else 'Valor',
            'color': '#FF441F',
        }
    
    elif analysis_type == 'avg_by_country':
        data = analysis_result.get('data', [])
        if not data:
            return None
        metric = analysis_result.get('metric', '')
        return {
            'type': 'bar_horizontal',
            'labels': [d['country'] for d in data],
            'values': [_chart_val(d['avg'], metric) for d in data],
            'value_labels': [d['avg_fmt'] for d in data],
            'title': f"Promedio por País — {metric}",
            'ylabel': '%' if any(is_percentage_metric(metric, d['avg']) for d in data) else 'Valor',
            'color': '#FF441F',
        }
    
    elif analysis_type == 'comparison':
        comp_data = analysis_result.get('data', {})
        if not comp_data:
            return None
        metric = analysis_result.get('metric', '')
        labels = list(comp_data.keys())
        values = [_chart_val(comp_data[k]['mean'], metric) for k in labels]
        value_labels = [comp_data[k].get('mean_fmt', str(comp_data[k]['mean'])) for k in labels]
        return {
            'type': 'bar',
            'labels': labels,
            'values': values,
            'value_labels': value_labels,
            'title': f"Comparación: {metric}",
            'ylabel': '%' if any(is_percentage_metric(metric, comp_data[k]['mean']) for k in labels) else 'Valor',
            'color': '#FF441F',
        }

    elif analysis_type == 'multivariable':
        data = analysis_result.get('data', [])
        if not data:
            return None
        metric_high = analysis_result.get('metric_high', '')
        metric_low = analysis_result.get('metric_low', '')
        return {
            'type': 'bar',
            'labels': [f"{d['zone'][:25]}\n({d['country']})" for d in data],
            'values': [_chart_val(d.get('val_high'), metric_high) for d in data],
            'value_labels': [f"{d.get('val_high_fmt','') } / {d.get('val_low_fmt','')}" for d in data],
            'title': f"Zonas: {metric_high} vs {metric_low}",
            'ylabel': '%' if any(is_percentage_metric(metric_high, d.get('val_high')) for d in data) else 'Valor',
            'color': '#FF441F',
        }
    
    elif analysis_type == 'growth_leaders':
        data = analysis_result.get('data', [])
        if not data:
            return None
        return {
            'type': 'bar',
            'labels': [f"{d['zone'][:22]}\n({d['country']})" for d in data],
            'values': [d['growth_pct'] * 100 for d in data],
            'value_labels': [d['growth_pct_fmt'] for d in data],
            'title': "Top Zonas por Crecimiento en Órdenes",
            'ylabel': '% crecimiento',
            'color': '#22c55e',
        }
    
    return None
