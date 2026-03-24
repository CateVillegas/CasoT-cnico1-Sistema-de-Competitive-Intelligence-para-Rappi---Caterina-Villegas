# 🦊 Rappi Analytics — Sistema de Análisis Inteligente

Sistema de IA para democratizar el acceso a datos operacionales de Rappi, diseñado para equipos de **Strategy, Planning & Analytics (SP&A)** y **Operations**.

## ✨ Capacidades clave

- **Asistente conversacional**: entiende lenguaje natural, ejecuta consultas estructuradas sobre el Excel operativo y responde con tablas Markdown, gráficos Matplotlib y exportación CSV.
- **Motor de insights automáticos**: recorre todas las zonas/países, detecta anomalías, tendencias, correlaciones y oportunidades; genera un resumen ejecutivo listo para “copy-paste”.
- **Reportes ejecutivos en PDF**: usa ReportLab + Gemini para compilar hallazgos, scorecards por país y visualizaciones internas; cada corrida crea un archivo `reports/rappi_insights_YYYYMMDD_HHMMSS.pdf`.
- **Experiencia UI dual**: la misma SPA ofrece la vista de chat para preguntas puntuales y la vista "Insights automáticos" para correr el deep dive completo y descargar el PDF.
- **Instrumentación adicional**: generación de CSV, descarga de gráficos, badges de cobertura (9 países, 13 métricas) y chips de consulta rápida.

---

## 🚀 Puesta en marcha rápida

### 1. Requisitos
- Python 3.8+
- API Key de Gemini 2.5 Flash (se puede leer desde `.env` vía `python-dotenv`)

### 2. Crear entorno e instalar dependencias
```bash
cd rappi_analytics
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar la API Key
```bash
# macOS / Linux
export GEMINI_API_KEY="tu-api-key"

# Windows PowerShell
$env:GEMINI_API_KEY = "tu-api-key"

# Windows CMD
set GEMINI_API_KEY=tu-api-key
```

### 4. Levantar el servidor Flask
```bash
python app.py
```

### 5. Abrir la aplicación
```
http://localhost:5001
```

---

## 🧭 Flujos dentro de la interfaz

### Chat “Asistente de Operaciones”
1. Ingresá a la vista **Asistente** (por defecto).
2. Realizá preguntas tipo “Top 5 zonas con peor Perfect Orders en MX”.
3. El backend envía la consulta a Gemini → `analysis.py` → genera tabla/gráfico y respuesta ejecutiva.
4. Opcional: descargá el CSV o la imagen del gráfico con los botones del mensaje.

### Insights automáticos
1. Cambiá a la pestaña **Insights**.
2. Clic en **Generar Insights** para lanzar `insights_engine.py` + Gemini (tarda 20–40s).
3. Se renderiza el resumen ejecutivo, KPIs, anomalías, oportunidades y gráficos especiales.
4. Habilitá **Descargar PDF** para crear el reporte (también se guarda en `reports/`).

> Cada ejecución mantiene el historial de PDFs en la carpeta `reports/`, por lo que podés versionar reportes sin sobrescribir.

---

## 🔌 API y automatizaciones

| Endpoint | Método | Uso principal |
|----------|--------|---------------|
| `/api/chat` | POST | Procesa mensajes del chat (LLM + análisis + chart base64 + CSV).
| `/api/export-csv` | POST | Devuelve un CSV arbitrario generado en el chat.
| `/api/insights-data` | POST | Ejecuta el motor de insights completos y devuelve JSON con hallazgos + gráficos (base64).
| `/api/generate-report` | POST | Lanza la generación de insights + PDF en background (con estado interno).
| `/api/report-status` | GET | Consulta el estado del job (`idle/running/done/error`).
| `/api/download-report` | GET | Descarga el PDF más reciente (`reports/...`).
| `/api/generate-pdf` | POST | Regenera únicamente el PDF usando el último resultado de insights cargado en memoria.

### Ejemplo rápido con `curl`
```bash
# Lanzar reporte completo
curl -X POST http://localhost:5001/api/generate-report

# Consultar progreso
curl http://localhost:5001/api/report-status

# Descargar PDF cuando el estado sea "done"
curl -OJ http://localhost:5001/api/download-report

# Regenerar solo PDF tras usar la vista de Insights
curl -X POST http://localhost:5001/api/generate-pdf -o Rappi_Reporte_Ejecutivo.pdf
```

---

## 🧠 Arquitectura técnica

```
Usuario → Frontend (index.html)
     → Flask (/api/chat | /api/insights-data)
     → Gemini 2.5 Flash (parse_query_to_analysis / generate_response)
     → Módulos de análisis (data_loader.py + analysis.py)
     → Matplotlib / ReportLab (gráficos base64 + PDF)
     → Respuesta UI + descargas (CSV, PNG, PDF)
```

- `src/data_loader.py`: cachea el Excel, formatea métricas y expone helpers para semanas, países y ratios.
- `src/analysis.py`: implementa las consultas estructuradas (top_zones, trends, comparisons, multivariable, growth, problematic_zones, etc.).
- `src/gemini_client.py`: maneja prompts, parsing JSON y respuesta natural con Gemini 2.5 Flash.
- `src/insights_engine.py`: identifica anomalías, tendencias, oportunidades, correlaciones y arma scorecards + visualizaciones reutilizadas tanto en la UI como en el PDF.
- `app.py`: orquesta endpoints, renderiza gráficos (tema dark Rappi) y coordina generación de PDFs.

---

## 📁 Estructura del Proyecto

```
rappi_analytics/
├── app.py                  # Flask app principal, rutas API
├── Informe final.pdf
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

## 💰 Costo estimado de API
- ~$0.02–0.05 por pregunta (Gemini 2.5 Flash input + output tokens)
- ~$0.10–0.30 por reporte ejecutivo completo
- Costo estimado por sesión de 10 preguntas: **~$0.50**

---

## ⚠️ Limitaciones conocidas
- El contexto conversacional vive solo en memoria del navegador (no hay persistencia multiusuario).
- Los nombres de zonas con caracteres especiales deben ingresarse tal como figuran en el Excel para evitar falsos negativos.
- Las consultas multivariables requieren que ambas métricas existan en las mismas zonas, de lo contrario se filtran.
- El generador de insights depende del último dataset cargado; si se actualiza `data.xlsx` es necesario reiniciar el servidor.
- El PDF exporta gráficos como imágenes raster y tablas como texto enriquecido, no como tablas editables.

## 🔮 Próximos pasos
- Persistencia de conversaciones e insights en SQLite/Postgres para auditoría y handoff entre analistas.
- Alertas automáticas por email/Slack cuando se detecten anomalías críticas.
- Deployment administrado (Render/Railway) con autenticación básica y rotación de API Keys.
- Carga de nuevos datasets vía UI + validaciones (en vez de reemplazar `data.xlsx`).
- Dashboard comparativo en tiempo real (streaming) reutilizando el mismo motor de análisis.
