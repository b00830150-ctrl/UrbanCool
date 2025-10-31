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
        # streamlit uploaded file -> write to temp buffer
        data = path_or_file.read()
        return read_raster_bytes(data)
    else:
        ds = rasterio.open(path_or_file)
        arr = ds.read(1, masked=True).astype('float32')
        return ds, arr, ds.transform, ds.crs

def read_raster_bytes(data_bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(data_bytes)
        tmp.flush()
        ds = rasterio.open(tmp.name)
        arr = ds.read(1, masked=True).astype('float32')
        return ds, arr, ds.transform, ds.crs

def reproject_match(src_ds, src_arr, target_ds, resampling=Resampling.bilinear):
    """
    Reprojette src raster sur la grille et CRS du target_ds
    Retourne array reprojected et transform
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
    # mask nodata if src had nodata
    return dst_arr

def detect_uhi(temp_arr, method="percentile", param=90):
    """
    Detect heat islands.
    method: 'percentile' or 'zscore'
    param: percentile threshold (int) or zscore threshold (float)
    Returns mask (bool) and intensity (float array normalized)
    """
    a = np.array(temp_arr)
    # treat masked or nan as nan
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
    # normalize intensity to 0-1 for present pixels
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
    """
    Vectorize mask (binary 0/1) to GeoDataFrame polygons (in raster CRS).
    min_area_pixels: remove tiny polygons
    """
    shapes_gen = list(shapes(mask_arr.astype('uint8'), mask=mask_arr.astype(bool), transform=transform))
    geoms = []
    vals = []
    for geom, val in shapes_gen:
        if val == 0:
            continue
        g = shape(geom)
        if g.area == 0:
            continue
        # optional area filter (in pixel units) - approximate by area threshold on bounds
        geoms.append(g)
        vals.append(val)
    if len(geoms) == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'value'], crs=crs)
    gdf = gpd.GeoDataFrame({'geometry': geoms, 'value': vals}, crs=crs)
    # drop very small polygons by area in pixels -> convert to meters if CRS is projected
    return gdf

def compute_population_exposure(polygons_gdf, pop_ds, pop_arr, raster_transform):
    """
    For each polygon, compute population exposed by summing population raster values inside polygon.
    Requires pop_arr aligned with the same grid as polygons raster (same transform & crs).
    Returns list of population sums.
    """
    if polygons_gdf.empty:
        return []
    # compute zonal stats using rasterstats (expects file or array)
    affine = raster_transform
    zs = zonal_stats(polygons_gdf.geometry, pop_arr, affine=affine, stats=['sum'], nodata=np.nan)
    pop_sums = [z.get('sum') if z.get('sum') is not None else 0.0 for z in zs]
    return pop_sums

def landcover_dominant_for_polygons(polygons_gdf, lc_arr, lc_transform):
    """
    For each polygon, compute dominant landcover class (most frequent) within polygon.
    lc_arr must align with polygons raster grid.
    Returns list of dominant classes.
    """
    if polygons_gdf.empty:
        return []
    zs = zonal_stats(polygons_gdf.geometry, lc_arr, affine=lc_transform, stats=['majority'], categorical=False, nodata=np.nan)
    doms = [z.get('majority') if z.get('majority') is not None else np.nan for z in zs]
    return doms

def propose_intervention(dominant_lc, building_ratio=None):
    """
    Map dominant landcover class to suggested intervention(s).
    This is heuristic and customizable.
    dominant_lc: numeric code from landcover raster (user must interpret)
    building_ratio: optional indicator of rooftop area fraction
    Returns a string recommendation and a suitability score (0-1)
    """
    # Example heuristic mapping (users should adapt to their landcover legend)
    # For many global landcovers: 1=water, 2=forest, 3=grass, 4=cropland, 5=urban/built, 6=barren...
    if np.isnan(dominant_lc):
        return "Tree planting / Cool roofs (general)", 0.5
    try:
        c = int(dominant_lc)
    except:
        return "Tree planting / Cool roofs (general)", 0.5
    if c in (5,):  # built-up
        if building_ratio is not None and building_ratio > 0.3:
            return "Cool roofs & Green roofs (priority)", 0.9
        else:
            return "Street trees & pocket parks", 0.8
    elif c in (2,3):  # vegetated classes
        return "Manage vegetation (increase canopy continuity)", 0.4
    elif c in (6,):
        return "Tree planting + pervious paving", 0.7
    elif c in (1,):  # water
        return "Protect water bodies / cooling corridors", 0.2
    else:
        return "Tree planting / Cool roofs", 0.6

def compute_priority_score(population, intensity_mean, suitability):
    """
    Simple priority score combining population exposed, mean intensity and suitability (0-1).
    Normalize to a wide range for ranking.
    """
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
    **Fichiers attendus** (optionnel selon disponibilité) :
    - Raster température (GeoTIFF) — satellite ou réseau de températures aériennes.
    - Raster landcover (GeoTIFF) — classes numériques (ex : 1=eau,2=forêt,3=herbe,5=urbain...).
    - Raster population (GeoTIFF) — population par pixel (optionnel).
    - Raster bâtiment / rooftop ratio (GeoTIFF) — fraction bâti par pixel (optionnel, utile pour prioriser toits).
    
    **Important** : Les rasters seront reprojetés sur la grille du raster température s'ils ont un CRS ou résolution différente.
    """)
    st.info("Si tu n'as pas de population raster, l'analyse se fera juste sur la zone et la surface, sinon la priorisation par population sera utilisée.")

col1, col2 = st.columns([1,1])
with col1:
    temp_file = st.file_uploader("1) Raster TEMPÉRATURE (GeoTIFF) — obligatoire", type=["tif","tiff"])
    temp_path_text = st.text_input("Ou chemin local pour le raster température (laisser vide si upload).")
with col2:
    lc_file = st.file_uploader("2) Raster LANDCOVER (GeoTIFF) — recommandé", type=["tif","tiff"])
    pop_file = st.file_uploader("3) Raster POPULATION (GeoTIFF) — optionnel", type=["tif","tiff"])
    built_file = st.file_uploader("4) Raster BÂTI (ratio rooftop or built fraction) — optionnel", type=["tif","tiff"])

use_example = st.checkbox("🔎 Charger un exemple intégré (démos) — *non implémenté dans cette version*", value=False, disabled=True)

# Load temp raster
if not temp_file and not temp_path_text:
    st.warning("Charge le raster température (obligatoire) pour lancer l'analyse.")
    st.stop()

# Read datasets
if temp_file:
    temp_ds, temp_arr, temp_transform, temp_crs = read_raster(temp_file)
else:
    temp_ds, temp_arr, temp_transform, temp_crs = read_raster(temp_path_text)

st.success("Raster température chargé.")
st.write(f"CRS température : {temp_crs}")

# Optionally load other rasters and reproject/resample to temperature grid
if lc_file:
    lc_ds, lc_arr_raw, lc_transform_raw, lc_crs_raw = read_raster(lc_file)
    st.write(f"Landcover CRS : {lc_crs_raw}")
    st.info("Reprojection landcover vers grille température...")
    lc_arr = reproject_match(lc_ds, lc_arr_raw, temp_ds, resampling=Resampling.nearest)
else:
    lc_arr = None

if pop_file:
    pop_ds, pop_arr_raw, pop_transform_raw, pop_crs_raw = read_raster(pop_file)
    st.write(f"Population CRS : {pop_crs_raw}")
    st.info("Reprojection population vers grille température...")
    pop_arr = reproject_match(pop_ds, pop_arr_raw, temp_ds, resampling=Resampling.sum)
else:
    pop_arr = None

if built_file:
    b_ds, b_arr_raw, b_transform_raw, b_crs_raw = read_raster(built_file)
    st.write(f"Built CRS : {b_crs_raw}")
    st.info("Reprojection built ratio vers grille température...")
    built_arr = reproject_match(b_ds, b_arr_raw, temp_ds, resampling=Resampling.bilinear)
else:
    built_arr = None

# Preprocess temp: mask invalids
temp_arr_masked = np.where(np.isfinite(temp_arr), temp_arr, np.nan)

# User UHI detection params
st.sidebar.header("Paramètres UHI")
method = st.sidebar.selectbox("Méthode de détection", ["percentile", "zscore"])
if method == "percentile":
    p = st.sidebar.slider("Percentile seuil (%)", 75, 99, 90)
    param = p
else:
    z = st.sidebar.slider("Z-score seuil", 1.0, 4.0, 2.0, step=0.1)
    param = z

run_btn = st.button("➡️ Lancer l'analyse & proposer interventions")

if run_btn:
    with st.spinner("Détection des îlots de chaleur..."):
        mask, intensity_norm = detect_uhi(temp_arr_masked, method=method, param=param)
    st.success("UHI détectés.")
    # Vectorize
    uhi_gdf = raster_to_polygons(mask, temp_transform, temp_crs)
    if uhi_gdf.empty:
        st.warning("Aucune zone UHI détectée avec ces paramètres.")
    else:
        uhi_gdf = uhi_gdf.to_crs(epsg=4326)  # convert to WGS84 for mapping
        st.write(f"{len(uhi_gdf)} polygones UHI générés (convertis en WGS84 pour la carte).")
        # Compute for each polygon: mean intensity, population exposure, dominant landcover
        # But intensity_norm is on temp grid: need to compute zonal stats for mean intensity
        # Prepare intensity array aligned with temp grid (float)
        intensity_arr = intensity_norm
        # Compute mean intensity per polygon (zonal_stats expects original geom crs; we must ensure correct affine & crs)
        # We have polygons in WGS84, but underlying raster is in temp_crs. So better to vectorize in temp_crs earlier.
        # Recreate polygon GDF in raster CRS for zonal stats:
        uhi_raster_crs = raster_to_polygons(mask, temp_transform, temp_crs)  # polygons in raster CRS
        if uhi_raster_crs.empty:
            st.error("Erreur lors de la vectorisation initiale.")
            st.stop()
        # Compute mean intensity
        intensity_stats = zonal_stats(uhi_raster_crs.geometry, intensity_arr, affine=temp_transform, stats=['mean'], nodata=np.nan)
        intensity_means = [s.get('mean') if s.get('mean') is not None else 0.0 for s in intensity_stats]
        # Compute population exposure
        if pop_arr is not None:
            pop_sums = compute_population_exposure(uhi_raster_crs, pop_ds, pop_arr, temp_transform)
        else:
            # fallback: approximate exposure by polygon area in m^2 (if projected) otherwise by pixel count
            try:
                # if temp_crs is projected, compute polygon area in m^2
                tmp_gdf = uhi_raster_crs.to_crs(temp_crs)
                areas = tmp_gdf.geometry.area  # if units are meters
                pop_sums = [a for a in areas]  # not population but area proxy
            except:
                pop_sums = [g.area for g in uhi_raster_crs.geometry]
        # Landcover dominant
        if lc_arr is not None:
            lc_doms = landcover_dominant_for_polygons(uhi_raster_crs, lc_arr, temp_transform)
        else:
            lc_doms = [np.nan] * len(uhi_raster_crs)
        # building ratio mean per polygon
        if built_arr is not None:
            built_stats = zonal_stats(uhi_raster_crs.geometry, built_arr, affine=temp_transform, stats=['mean'], nodata=np.nan)
            built_means = [s.get('mean') if s.get('mean') is not None else 0.0 for s in built_stats]
        else:
            built_means = [None] * len(uhi_raster_crs)
        # assemble final GeoDataFrame (convert to WGS84 for display)
        uhi_raster_crs['mean_intensity'] = intensity_means
        uhi_raster_crs['pop_exposed'] = pop_sums
        uhi_raster_crs['lc_majority'] = lc_doms
        uhi_raster_crs['built_mean'] = built_means
        # Propose interventions and compute suitability + priority
        recs = []
        suits = []
        scores = []
        for lc, bm, intensity, pop in zip(uhi_raster_crs['lc_majority'], uhi_raster_crs['built_mean'], uhi_raster_crs['mean_intensity'], uhi_raster_crs['pop_exposed']):
            rec, suit = propose_intervention(lc, building_ratio=bm)
            recs.append(rec)
            suits.append(suit)
            score = compute_priority_score(pop if pop is not None else 0.0, intensity if intensity is not None else 0.0, suit)
            scores.append(score)
        uhi_raster_crs['recommendation'] = recs
        uhi_raster_crs['suitability'] = suits
        uhi_raster_crs['priority_score'] = scores
        # Convert to WGS84 for display
        uhi_display = uhi_raster_crs.to_crs(epsg=4326)
        # Sort by priority
        uhi_display_sorted = uhi_display.sort_values('priority_score', ascending=False).reset_index(drop=True)

        st.subheader("Résultats — Top zones prioritaires")
        st.dataframe(uhi_display_sorted[['mean_intensity', 'pop_exposed', 'lc_majority', 'built_mean', 'recommendation', 'priority_score']].rename(columns={
            'mean_intensity': 'intensité moyenne (norm)',
            'pop_exposed': 'population exposée (ou proxy)',
            'lc_majority': 'landcover majoritaire',
            'built_mean': 'moy. bâti',
            'recommendation': 'Recommandation',
            'priority_score': 'Score priorité'
        }).round(3))

        # Map visualization with Folium
        st.subheader("Carte interactive")
        # compute center
        centroid = uhi_display_sorted.geometry.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="CartoDB positron")
        # Add temperature raster as simple RGB hillshade? We don't have RGB; optionally add transparent heat layer from mask
        # Add polygons colored by priority
        from branca.colormap import linear
        max_score = uhi_display_sorted['priority_score'].max() if not uhi_display_sorted.empty else 1.0
        colormap = linear.YlOrRd_09.scale(0, max_score)
        colormap.caption = "Score de priorité"
        colormap.add_to(m)
        # Add polygons
        for _, row in uhi_display_sorted.iterrows():
            geom = mapping(row.geometry)
            score = float(row['priority_score'])
            popup_html = f"""
            <b>Priorité :</b> {score:.2f}<br/>
            <b>Population exposée :</b> {row['pop_exposed']:.1f}<br/>
            <b>Intensité moy. :</b> {row['mean_intensity']:.3f}<br/>
            <b>Landcover :</b> {row['lc_majority']}<br/>
            <b>Recommandation :</b> {row['recommendation']}
            """
            gj = folium.GeoJson(geom,
                                style_function=lambda feat, s=score: {
                                    'fillColor': colormap(s),
                                    'color': 'black',
                                    'weight': 0.8,
                                    'fillOpacity': 0.6
                                })
            gj.add_child(folium.Popup(popup_html, max_width=300))
            gj.add_to(m)

        # Render folium map in Streamlit
        from streamlit.components.v1 import html as st_html
        map_html = m._repr_html_()
        st_html(map_html, height=600)

        # Export options
        st.subheader("Export")
        # GeoJSON
        geojson_bytes = uhi_display_sorted.to_json().encode('utf-8')
        st.download_button("Télécharger GeoJSON des zones UHI", data=geojson_bytes, file_name="urbancool_uhi.geojson", mime="application/geo+json")
        # CSV summary
        csv_buf = uhi_display_sorted.drop(columns='geometry').to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger tableau CSV", data=csv_buf, file_name="urbancool_uhi_summary.csv", mime="text/csv")

        # Simple plot: priority histogram
        st.subheader("Distribution des scores de priorité")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(uhi_display_sorted['priority_score'].fillna(0), bins=15)
        ax.set_xlabel("Score de priorité")
        ax.set_ylabel("Nombre de zones")
        st.pyplot(fig)

        st.success("Analyse terminée — adapte les heuristiques (mapping landcover ↔ intervention) selon ta légende landcover locale pour de meilleurs résultats.")

st.markdown("---")
st.write("ℹ️ Astuce : pour production/usage réel, ajoute des contrôles pour la légende landcover (mappage numérique→libellé), "
         "filtrage temporel des rasters (séries temporelles), et validations sur les CRS/projections.")