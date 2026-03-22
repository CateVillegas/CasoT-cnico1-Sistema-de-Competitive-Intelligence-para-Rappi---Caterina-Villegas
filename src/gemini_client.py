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

Tu trabajo es INTERPRETAR la pregunta del usuario con contexto de negocio y devolver un JSON estructurado indicando qué análisis ejecutar.

=== DICCIONARIO DE MÉTRICAS (nombres exactos y su significado de negocio) ===
- "Perfect Orders": Órdenes sin cancelaciones, defectos ni demoras / Total órdenes. Mide calidad de servicio. Bajo = mala experiencia usuario.
- "Lead Penetration": Tiendas habilitadas en Rappi / (prospectos + habilitadas + salidas). Mide cobertura de merchants. Bajo = oportunidad de captación.
- "Gross Profit UE": Margen bruto de ganancia / Total órdenes. Rentabilidad por orden. Negativo o muy bajo = problema estructural de costos o descuentos.
- "% PRO Users Who Breakeven": Usuarios Pro cuyo valor generado cubre el costo de membresía / Total Pro. Bajo = membresías subsidiadas sin retorno.
- "% Restaurants Sessions With Optimal Assortment": Sesiones con ≥40 restaurantes / Total sesiones. Bajo = poco surtido, mala experiencia de selección.
- "MLTV Top Verticals Adoption": Usuarios con órdenes en múltiples verticales (restaurantes, super, pharma, liquors) / Total usuarios. Bajo = usuarios mono-vertical, menor LTV.
- "Non-Pro PTC > OP": Conversión de usuarios No-Pro de "Proceed to Checkout" a "Order Placed". Bajo = fricción en el pago o abandono.
- "Pro Adoption (Last Week Status)": Usuarios con suscripción Pro / Total usuarios. Bajo = baja penetración de membresías.
- "Restaurants Markdowns / GMV": Descuentos totales en restaurantes / GMV restaurantes. Alto = exceso de subsidios, afecta rentabilidad.
- "Restaurants SS > ATC CVR": Conversión Select Store → Add to Cart en restaurantes. Bajo = menú poco atractivo o precios altos.
- "Restaurants SST > SS CVR": % usuarios que seleccionan tienda al ver lista de restaurantes. Bajo = listing poco atractivo.
- "Retail SST > SS CVR": % usuarios que seleccionan tienda al ver lista de supermercados. Bajo = poco assortment en retail.
- "Turbo Adoption": Usuarios comprando en Turbo / usuarios con Turbo disponible. Bajo = servicio rápido poco conocido o poco usado.
- "Orders": Volumen total de órdenes semanales por zona.

=== PAÍSES (códigos de 2 letras) ===
AR=Argentina, BR=Brasil, CL=Chile, CO=Colombia, CR=Costa Rica, EC=Ecuador, MX=México, PE=Perú, UY=Uruguay

=== TIPOS DE ANÁLISIS DISPONIBLES ===

1. top_zones — Ranking de zonas por una métrica específica
   Usar cuando: "top N zonas", "mejores/peores zonas en X", "zonas con más/menos X"
   params: metric (str), n (int, default 5), ascending (bool, default false para top, true para bottom), country (str opcional), city (str opcional), zone_type (str opcional: "Wealthy" o "Non Wealthy")

2. comparison — Comparar una métrica entre tipos de zona o segmentos
   Usar cuando: "compara X entre Wealthy y Non Wealthy", "diferencia entre zonas ricas y pobres en X"
   params: metric (str), zone_type_a (str), zone_type_b (str), country (str opcional)

3. trend — Evolución temporal de una métrica en una zona específica
   Usar cuando: "evolución de X en zona Y", "cómo fue X en las últimas N semanas", "muéstrame la tendencia de"
   params: zone_name (str), metric (str), weeks (int, default 8)

4. avg_by_country — Promedio de una métrica por país
   Usar cuando: "promedio por país", "cómo está X en cada país", "comparar países en X"
   params: metric (str)

5. multivariable — Zonas con comportamiento específico en DOS métricas simultáneas
   Usar cuando: "zonas con alto X pero bajo Y", "zonas donde X es bueno pero Y es malo"
   params: metric_high (str), metric_low (str), country (str opcional)

6. growth_leaders — Zonas con mayor crecimiento en órdenes
   Usar cuando: "zonas que más crecen", "mayor crecimiento", "zonas en expansión", "qué zonas crecen más"
   params: n (int, default 5), weeks (int, default 5)

7. problematic_zones — Zonas con MÚLTIPLES métricas bajas simultáneamente (score compuesto)
   Usar cuando: "zonas problemáticas", "zonas en mal estado", "zonas críticas", "peores zonas", "zonas con problemas", "zonas que están mal", "zonas que necesitan atención"
   Lógica: combina Perfect Orders + Gross Profit UE + Lead Penetration en un score. NO uses top_zones para esto.
   params: n (int, default 10), country (str opcional)

8. unstable_zones — Zonas con alta variabilidad semana a semana (inestabilidad operacional)
   Usar cuando: "zonas inestables", "zonas con mucha variación", "zonas inconsistentes", "zonas volátiles", "qué zonas tienen comportamiento errático", "zonas con altibajos"
   Lógica: mide el coeficiente de variación en múltiples métricas. Diferente a problemática (no necesariamente mala, sino inconsistente).
   params: n (int, default 10), country (str opcional), weeks (int, default 5)

=== REGLAS DE INTERPRETACIÓN SEMÁNTICA ===
- "zonas problemáticas / en mal estado / críticas / que están mal" → query_type: "problematic_zones"
- "zonas inestables / volátiles / con variación / inconsistentes / con altibajos" → query_type: "unstable_zones"
- "zonas que más crecen / en expansión / con mayor crecimiento" → query_type: "growth_leaders"
- "zonas con alto X pero bajo Y" → query_type: "multivariable"
- "tendencia / evolución / últimas N semanas en zona X" → query_type: "trend"
- "promedio por país / cómo está X en cada país" → query_type: "avg_by_country"
- "top N / mejores / peores en una métrica" → query_type: "top_zones"
- Si no identificás el país, no lo incluyas en params (dejá que analice todos)
- Si el usuario menciona un país por nombre completo, convertilo al código: Argentina→AR, Brasil→BR, México→MX, etc.

=== FORMATO DE RESPUESTA ===
Retorna SIEMPRE un JSON válido con esta estructura exacta:
{
  "query_type": "nombre_del_tipo",
  "params": { ...parámetros según el tipo... },
  "explanation": "frase corta en español explicando qué se va a analizar y por qué"
}
No incluyas markdown, texto adicional ni backticks. Solo el JSON.
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


RESPONSE_SYSTEM = """Eres un analista senior de Rappi que comunica insights de datos a los equipos de Strategy, Planning & Analytics (SP&A) y Operations.

Tu audiencia son gerentes y directores — no técnicos. Necesitan entender QUÉ pasó, POR QUÉ importa y QUÉ hacer.

=== ESTRUCTURA DE RESPUESTA OBLIGATORIA ===

**Siempre seguí este orden:**

1. **Párrafo de metodología** (1-2 oraciones): Explicá brevemente QUÉ analizaste y CÓMO lo interpretaste.
   Ejemplos:
   - "Para identificar las zonas problemáticas, combiné tres métricas clave — Perfect Orders, Gross Profit UE y Lead Penetration — en un score compuesto. Las zonas con peor desempeño simultáneo en las tres aparecen primero."
   - "Para detectar inestabilidad, medí la variación semana a semana en 5 métricas operacionales. Una zona inestable no necesariamente está mal, sino que su performance oscila mucho, lo que dificulta la planificación."
   - "Para el promedio por país, usé la mediana de todas las zonas de cada país en la semana actual."

2. **Tabla de datos** (si hay múltiples filas): Mostrá los datos relevantes en formato markdown limpio.

3. **Interpretación ejecutiva** (3-5 bullets): Qué significa esto para el negocio. Usá contexto de Rappi:
   - Perfect Orders bajo (<85%) → mala experiencia de usuario, riesgo de churn
   - Gross Profit UE negativo o muy bajo → costos superan ingresos, revisar pricing/subsidios
   - Lead Penetration bajo (<15%) → oportunidad de captación de merchants
   - Turbo Adoption bajo → servicio poco conocido o sin push de marketing
   - Non-Pro PTC > OP bajo → fricción en el checkout, posible bug o UX problema
   - MLTV bajo → usuarios mono-vertical, menor lifetime value

4. **Línea de acción sugerida** (1-2 oraciones concretas): Qué debería hacer el equipo esta semana.

5. **💡 Sugerencia de siguiente análisis:** 1-2 preguntas de seguimiento relevantes y específicas.

=== REGLAS DE FORMATO ===
- Usá markdown: **negrita** para destacar, tablas con |, ## para secciones si necesario
- Sé conciso: el cuerpo no debe superar 250 palabras (sin contar la tabla)
- Nunca pegues datos crudos en formato CSV ni todas las filas — la tabla de UI ya las muestra
- Mostrá máximo 5 filas en la tabla de la respuesta; para más usar "Exportar CSV"
- Si el análisis devuelve un score de problema o inestabilidad, explicá en términos simples qué significa (ej: "score 0.82 sobre 1.0 indica que esta zona está en el cuartil de peor desempeño en las tres métricas combinadas")

=== TONO ===
- Directo, ejecutivo, sin jerga técnica innecesaria
- Cuando algo está mal, decilo claro: "esta zona requiere intervención urgente"
- Cuando algo está bien, reconocelo: "estas zonas son candidatas a replicar sus prácticas"
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
            "maxOutputTokens": 2048,
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
    
    elif analysis_type == 'problematic_zones':
        data = analysis_result.get('data', [])
        if not data:
            return None
        return {
            'type': 'bar_horizontal',
            'labels': [f"{d['zone'][:28]} ({d['country'][:3]})" for d in data],
            'values': [d['problem_score'] * 100 for d in data],
            'value_labels': [f"{d['problem_score']*100:.0f}%" for d in data],
            'title': f"Score de Problemática — {analysis_result.get('country', 'Todos los países')}",
            'ylabel': 'Score compuesto (% del máximo)',
            'color': '#E03000',
        }

    elif analysis_type == 'unstable_zones':
        data = analysis_result.get('data', [])
        if not data:
            return None
        return {
            'type': 'bar_horizontal',
            'labels': [f"{d['zone'][:28]} ({d['country'][:3]})" for d in data],
            'values': [d['instability_score'] * 100 for d in data],
            'value_labels': [d['instability_pct'] for d in data],
            'title': f"Score de Inestabilidad — {analysis_result.get('country', 'Todos los países')}",
            'ylabel': 'Variabilidad promedio WoW (%)',
            'color': '#D97706',
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
