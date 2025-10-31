# urbancool_app.py
"""
UrbanCool - Streamlit app
Map urban heat islands from satellite/air temp + landcover and propose low-cost greening interventions prioritized by population exposure.

Usage:
    streamlit run urbancool_app.py
"""

import streamlit as st
st.set_page_config("UrbanCool — Urban Heat Islands & Low-cost Greening", layout="wide")

# Imports
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd
import pandas as pd
import json
import folium
from folium.features import GeoJson
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from io import BytesIO
from rasterstats import zonal_stats
from pyproj import CRS, Transformer

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def read_raster(path_or_file):
    """
    Retourne: dataset (rasterio dataset), array, transform, crs
    path_or_file: path string or uploaded file-like object
    """
    if hasattr(path_or_file, "read"):
        # Streamlit uploaded file -> read in memory
        file_bytes = BytesIO(path_or_file.read())
        ds = rasterio.open(file_bytes)
        arr = ds.read(1, masked=True).astype('float32')
        return ds, arr, ds.transform, ds.crs
    else:
        ds = rasterio.open(path_or_file)
        arr = ds.read(1, masked=True).astype('float32')
        return ds, arr, ds.transform, ds.crs

def reproject_match(src_ds, src_arr, target_ds, resampling=Resampling.bilinear):
    """
    Reprojette src raster sur la grille et CRS du target_ds
    Retourne array reprojected
    """
    dst_crs = target_ds.crs
    dst_transform = target_ds.transform
    dst_shape = (target_ds.height, target_ds.width)
    dst_arr = np.empty(dst_shape, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling
    )
    return dst_arr

def detect_uhi(temp_arr, method="percentile", param=90):
    a = np.array(temp_arr)
    a = np.where(np.isfinite(a), a, np.nan)
    if method == "percentile":
        thresh = np.nanpercentile(a, param)
        mask = (a >= thresh) & (~np.isnan(a))
        intensity = a - thresh
    else:  # zscore
        mean = np.nanmean(a)
        std = np.nanstd(a)
        z = (a - mean) / (std + 1e-9)
        mask = (z >= param) & (~np.isnan(a))
        intensity = z
    valid = np.where(mask, intensity, np.nan)
    if np.nanmax(valid) == np.nanmin(valid) or np.isnan(np.nanmax(valid)):
        norm = np.zeros_like(valid)
    else:
        vmax = np.nanmax(valid)
        vmin = np.nanmin(valid)
        norm = (valid - vmin) / (vmax - vmin + 1e-9)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    return mask.astype(np.uint8), norm.astype(np.float32)

def raster_to_polygons(mask_arr, transform, crs, min_area_pixels=10):
    shapes_gen = list(shapes(mask_arr.astype('uint8'), mask=mask_arr.astype(bool), transform=transform))
    geoms = []
    vals = []
    for geom, val in shapes_gen:
        if val == 0:
            continue
        g = shape(geom)
        if g.area == 0:
            continue
        geoms.append(g)
        vals.append(val)
    if len(geoms) == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'value'], crs=crs)
    gdf = gpd.GeoDataFrame({'geometry': geoms, 'value': vals}, crs=crs)
    return gdf

def compute_population_exposure(polygons_gdf, pop_ds, pop_arr, raster_transform):
    if polygons_gdf.empty:
        return []
    affine = raster_transform
    zs = zonal_stats(polygons_gdf.geometry, pop_arr, affine=affine, stats=['sum'], nodata=np.nan)
    pop_sums = [z.get('sum') if z.get('sum') is not None else 0.0 for z in zs]
    return pop_sums

def landcover_dominant_for_polygons(polygons_gdf, lc_arr, lc_transform):
    if polygons_gdf.empty:
        return []
    zs = zonal_stats(polygons_gdf.geometry, lc_arr, affine=lc_transform, stats=['majority'], categorical=False, nodata=np.nan)
    doms = [z.get('majority') if z.get('majority') is not None else np.nan for z in zs]
    return doms

def propose_intervention(dominant_lc, building_ratio=None):
    if np.isnan(dominant_lc):
        return "Tree planting / Cool roofs (general)", 0.5
    try:
        c = int(dominant_lc)
    except:
        return "Tree planting / Cool roofs (general)", 0.5
    if c in (5,):
        if building_ratio is not None and building_ratio > 0.3:
            return "Cool roofs & Green roofs (priority)", 0.9
        else:
            return "Street trees & pocket parks", 0.8
    elif c in (2,3):
        return "Manage vegetation (increase canopy continuity)", 0.4
    elif c in (6,):
        return "Tree planting + pervious paving", 0.7
    elif c in (1,):
        return "Protect water bodies / cooling corridors", 0.2
    else:
        return "Tree planting / Cool roofs", 0.6

def compute_priority_score(population, intensity_mean, suitability):
    return (population * (0.5 + intensity_mean)) * (0.5 + suitability)

# -------------------------
# Streamlit UI
# -------------------------
st.title("UrbanCool — Urban Heat Islands & Low-cost Greening")
st.markdown(
    """
    **Objectif** : détecter les îlots de chaleur urbains (UHI) à partir d'un raster température, croiser avec couverture du sol et population, 
    puis proposer des interventions (arbres, toits frais, toits végétalisés) priorisées par exposition.
    """
)

with st.expander("📥 Données à charger / instructions (cliquez pour ouvrir)"):
    st.markdown("""
    **Fichiers attendus** :
    - Raster température (GeoTIFF)
    - Raster landcover (GeoTIFF)
    - Raster population (GeoTIFF)
    - Raster bâtiment / rooftop ratio (optionnel)
    """)
    st.info("Si pas de population raster, analyse basée sur surface uniquement.")

col1, col2 = st.columns([1,1])
with col1:
    temp_file = st.file_uploader("1) Raster TEMPÉRATURE (GeoTIFF) — obligatoire", type=["tif","tiff"])
    temp_path_text = st.text_input("Ou chemin local pour le raster température (laisser vide si upload).")
with col2:
    lc_file = st.file_uploader("2) Raster LANDCOVER (GeoTIFF) — recommandé", type=["tif","tiff"])
    pop_file = st.file_uploader("3) Raster POPULATION (GeoTIFF) — optionnel", type=["tif","tiff"])
    built_file = st.file_uploader("4) Raster BÂTI — optionnel", type=["tif","tiff"])

if not temp_file and not temp_path_text:
    st.warning("Charge le raster température pour lancer l'analyse.")
    st.stop()

if temp_file:
    temp_ds, temp_arr, temp_transform, temp_crs = read_raster(temp_file)
else:
    temp_ds, temp_arr, temp_transform, temp_crs = read_raster(temp_path_text)

st.success("Raster température chargé.")
st.write(f"CRS température : {temp_crs}")

# Optionally load other rasters
if lc_file:
    lc_ds, lc_arr_raw, lc_transform_raw, lc_crs_raw = read_raster(lc_file)
    lc_arr = reproject_match(lc_ds, lc_arr_raw, temp_ds, resampling=Resampling.nearest)
else:
    lc_arr = None

if pop_file:
    pop_ds, pop_arr_raw, pop_transform_raw, pop_crs_raw = read_raster(pop_file)
    pop_arr = reproject_match(pop_ds, pop_arr_raw, temp_ds, resampling=Resampling.sum)
else:
    pop_arr = None

if built_file:
    b_ds, b_arr_raw, b_transform_raw, b_crs_raw = read_raster(built_file)
    built_arr = reproject_match(b_ds, b_arr_raw, temp_ds, resampling=Resampling.bilinear)
else:
    built_arr = None

temp_arr_masked = np.where(np.isfinite(temp_arr), temp_arr, np.nan)

st.sidebar.header("Paramètres UHI")
method = st.sidebar.selectbox("Méthode de détection", ["percentile", "zscore"])
param = st.sidebar.slider("Paramètre", 75, 99, 90) if method=="percentile" else st.sidebar.slider("Z-score", 1.0, 4.0, 2.0, step=0.1)

run_btn = st.button("➡️ Lancer l'analyse & proposer interventions")

if run_btn:
    mask, intensity_norm = detect_uhi(temp_arr_masked, method=method, param=param)
    st.success("UHI détectés.")
    uhi_gdf = raster_to_polygons(mask, temp_transform, temp_crs)
    if uhi_gdf.empty:
        st.warning("Aucune zone UHI détectée.")
    else:
        uhi_raster_crs = raster_to_polygons(mask, temp_transform, temp_crs)
        intensity_stats = zonal_stats(uhi_raster_crs.geometry, intensity_norm, affine=temp_transform, stats=['mean'], nodata=np.nan)
        intensity_means = [s.get('mean') if s.get('mean') is not None else 0.0 for s in intensity_stats]
        pop_sums = compute_population_exposure(uhi_raster_crs, pop_ds, pop_arr, temp_transform) if pop_arr is not None else [g.area for g in uhi_raster_crs.geometry]
        lc_doms = landcover_dominant_for_polygons(uhi_raster_crs, lc_arr, temp_transform) if lc_arr is not None else [np.nan]*len(uhi_raster_crs)
if built_arr is not None:
    built_stats = zonal_stats(
        uhi_raster_crs.geometry,
        built_arr,
        affine=temp_transform,
        stats=['mean'],
        nodata=np.nan
    )
    built_means = [s.get('mean') if s.get('mean') is not None else 0.0 for s in built_stats]
else:
    built_means = [None] * len(uhi_raster_crs)