import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.xlsx')

_cache = {}

METRIC_DESCRIPTIONS = {
    "% PRO Users Who Breakeven": "Usuarios Pro cuyo valor generado cubre el costo de membresía",
    "% Restaurants Sessions With Optimal Assortment": "Sesiones con ≥40 restaurantes disponibles",
    "Gross Profit UE": "Margen bruto de ganancia por orden",
    "Lead Penetration": "Tiendas habilitadas en Rappi / total de prospectos identificados",
    "MLTV Top Verticals Adoption": "Usuarios con órdenes en múltiples verticales (resto, super, pharma, liquors)",
    "Non-Pro PTC > OP": "Conversión de usuarios No Pro de 'Proceed to Checkout' a 'Order Placed'",
    "Perfect Orders": "Órdenes sin cancelaciones, defectos ni demoras / total órdenes",
    "Pro Adoption (Last Week Status)": "Usuarios Pro / total usuarios de Rappi",
    "Restaurants Markdowns / GMV": "Descuentos en restaurantes / total GMV de restaurantes",
    "Restaurants SS > ATC CVR": "Conversión de 'Select Store' a 'Add to Cart' en restaurantes",
    "Restaurants SST > SS CVR": "% usuarios que seleccionan una tienda luego de ver la lista de restaurantes",
    "Retail SST > SS CVR": "% usuarios que seleccionan una tienda luego de ver la lista de supermercados",
    "Turbo Adoption": "Usuarios que compran en Turbo / total usuarios con Turbo disponible",
}

COUNTRY_NAMES = {
    "AR": "Argentina", "BR": "Brasil", "CL": "Chile", "CO": "Colombia",
    "CR": "Costa Rica", "EC": "Ecuador", "MX": "México", "PE": "Perú", "UY": "Uruguay"
}

WEEK_COLS_METRICS = ['L8W_ROLL', 'L7W_ROLL', 'L6W_ROLL', 'L5W_ROLL', 'L4W_ROLL', 'L3W_ROLL', 'L2W_ROLL', 'L1W_ROLL', 'L0W_ROLL']
WEEK_COLS_ORDERS = ['L8W', 'L7W', 'L6W', 'L5W', 'L4W', 'L3W', 'L2W', 'L1W', 'L0W']
WEEK_LABELS = ['Sem -8', 'Sem -7', 'Sem -6', 'Sem -5', 'Sem -4', 'Sem -3', 'Sem -2', 'Sem -1', 'Actual']

# Metrics that are 0–1 ratios displayable as percentages (multiply × 100 for display)
# Note: Lead Penetration has some data > 1 (anomalous entries) — show raw value × 100 for those ≤1
RATIO_METRICS = {
    'Retail SST > SS CVR', 'Restaurants SST > SS CVR', 'Restaurants SS > ATC CVR',
    'Non-Pro PTC > OP', '% PRO Users Who Breakeven', 'Pro Adoption (Last Week Status)',
    'MLTV Top Verticals Adoption', '% Restaurants Sessions With Optimal Assortment',
    'Lead Penetration', 'Restaurants Markdowns / GMV', 'Perfect Orders', 'Turbo Adoption'
}
# Metrics that are plain numeric (not a ratio)
NUMERIC_METRICS = {'Gross Profit UE', 'Orders'}


def format_metric_value(metric: str, value) -> str:
    """Format a metric value appropriately based on its type."""
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if metric in RATIO_METRICS:
        if abs(value) <= 1.5:  # treat as 0-1 ratio → show as %
            return f"{value * 100:.1f}%"
        else:  # outlier: show raw value
            return f"{value:,.1f}"
    else:
        return f"{value:,.2f}"


def is_percentage_metric(metric: str, value=None) -> bool:
    """Return True if this metric value should be multiplied × 100 for chart display."""
    if metric not in RATIO_METRICS:
        return False
    if value is not None and abs(value) > 1.5:
        return False
    return True

def load_data():
    if 'metrics' not in _cache:
        xl = pd.read_excel(DATA_PATH, sheet_name=None)
        metrics = xl['RAW_INPUT_METRICS'].copy()
        orders = xl['RAW_ORDERS'].copy()
        # Rename orders week cols to match
        orders = orders.rename(columns={
            'L8W': 'L8W_ROLL', 'L7W': 'L7W_ROLL', 'L6W': 'L6W_ROLL',
            'L5W': 'L5W_ROLL', 'L4W': 'L4W_ROLL', 'L3W': 'L3W_ROLL',
            'L2W': 'L2W_ROLL', 'L1W': 'L1W_ROLL', 'L0W': 'L0W_ROLL'
        })
        _cache['metrics'] = metrics
        _cache['orders'] = orders
        _cache['combined'] = pd.concat([metrics, orders], ignore_index=True)
    return _cache['metrics'], _cache['orders'], _cache['combined']

def get_context_summary():
    metrics_df, orders_df, _ = load_data()
    return {
        "countries": sorted(metrics_df['COUNTRY'].unique().tolist()),
        "cities": sorted(metrics_df['CITY'].unique().tolist()),
        "metrics": sorted(metrics_df['METRIC'].unique().tolist()),
        "zone_types": sorted(metrics_df['ZONE_TYPE'].unique().tolist()),
        "prioritizations": sorted(metrics_df['ZONE_PRIORITIZATION'].unique().tolist()),
        "total_zones": metrics_df['ZONE'].nunique(),
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "country_names": COUNTRY_NAMES,
        "week_labels": WEEK_LABELS,
    }
