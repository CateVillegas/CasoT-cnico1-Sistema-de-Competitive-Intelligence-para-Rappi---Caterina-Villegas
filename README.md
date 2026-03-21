# 🦊 Rappi Analytics — Sistema de Análisis Inteligente

Sistema de IA para democratizar el acceso a datos operacionales de Rappi, diseñado para equipos de **Strategy, Planning & Analytics (SP&A)** y **Operations**.

## ¿Qué hace este sistema?

### 🤖 Bot Conversacional (70%)
- Responde preguntas en **lenguaje natural** sobre métricas operacionales
- Soporta: rankings, comparaciones, tendencias, promedios, análisis multivariable, crecimiento
- Genera **gráficos automáticos** (barras, líneas) según el tipo de pregunta
- **Exportación CSV** de cualquier resultado
- **Sugerencias proactivas** de análisis relacionados
- Memoria conversacional (contexto de la sesión)
- Input fijo en la parte inferior — UX tipo ChatGPT

### 📊 Reporte Ejecutivo Automático (30%)
- Detecta automáticamente: anomalías, tendencias preocupantes, brechas de benchmarking, correlaciones y oportunidades
- Genera reporte ejecutivo en **PDF** con Gemini 2.5 Flash
- Descarga directa desde la interfaz

---

## 🚀 Cómo ejecutar

### 1. Requisitos
- Python 3.8+
- API Key de Gemini 2.5 Flash

### 2. Instalación de dependencias
```bash
pip install flask pandas openpyxl reportlab matplotlib
```

### 3. Configurar API Key
```bash
export GEMINI_API_KEY="tu-api-key-aqui"
```

### 4. Ejecutar
```bash
cd rappi_analytics
python app.py
```

### 5. Abrir en el browser
```
http://localhost:5001
```

---

## 📁 Estructura del Proyecto

```
rappi_analytics/
├── app.py                  # Flask app principal, rutas API
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Carga y cache del Excel, constantes
│   ├── analysis.py         # Motor de análisis (top_zones, comparisons, trends, etc.)
│   ├── gemini_client.py    # Integración Gemini: parsing NL → análisis, generación respuesta
│   └── insights_engine.py  # Detección automática de insights + generador PDF
├── templates/
│   └── index.html          # Frontend completo (single file)
├── data/
│   └── data.xlsx           # Dataset proporcionado
├── reports/                # PDFs generados (auto-creado)
└── README.md
```

---

## 💬 Ejemplos de Preguntas Soportadas

| Tipo | Ejemplo |
|------|---------|
| Ranking | "¿Cuáles son las 5 zonas con mayor Lead Penetration?" |
| Comparación | "Compara Perfect Order entre Wealthy y Non Wealthy en México" |
| Tendencia | "Muestra la evolución de Gross Profit UE en Chapinero últimas 8 semanas" |
| Promedio | "¿Cuál es el promedio de Lead Penetration por país?" |
| Multivariable | "Zonas con alto Lead Penetration pero bajo Perfect Order" |
| Crecimiento | "¿Qué zonas crecen más en órdenes en las últimas 5 semanas?" |
| Inferencia | "¿Cuáles son las zonas problemáticas?" |

---

## 💰 Costo estimado de API
- ~$0.02–0.05 por pregunta (Gemini 2.5 Flash input + output tokens)
- ~$0.10–0.30 por reporte ejecutivo completo
- Costo estimado por sesión de 10 preguntas: **~$0.50**

---

## 🏗️ Arquitectura Técnica

```
Usuario (lenguaje natural)
        ↓
   Flask API (/api/chat)
        ↓
   Gemini 2.5 Flash ← parse_query_to_analysis()
   (JSON con query_type + params)
        ↓
   Analysis Engine (pandas)
   (executa el análisis sobre el Excel)
        ↓
   Gemini 2.5 Flash ← generate_response()
   (respuesta ejecutiva en español)
        ↓
   Matplotlib → gráfico PNG base64
        ↓
   Frontend (respuesta + gráfico + CSV)
```

### Decisiones técnicas
- **Gemini 2.5 Flash**: Mejor balance costo/calidad para NLP + structured output JSON
- **Flask**: Simple, liviano, ideal para demos y prototipado rápido
- **pandas**: Análisis de datos eficiente sin overhead de DBs
- **Matplotlib**: Disponible offline, sin dependencias adicionales
- **ReportLab**: Generación PDF nativa sin servicios externos

---

## ⚠️ Limitaciones conocidas
- El contexto conversacional se pierde al recargar la página (no hay persistencia)
- Nombres de zonas con caracteres especiales pueden necesitar variantes
- Análisis multivariable requiere que ambas métricas existan en las mismas zonas
- El reporte PDF no incluye tablas markdown (se renderiza como texto)

## 🔮 Próximos pasos
- Persistencia de conversaciones (SQLite/Postgres)
- Alertas automáticas por email (SMTP + cron)
- Deployment en Render/Railway con autenticación
- Soporte para upload de nuevos datasets
- Dashboard de métricas en tiempo real
