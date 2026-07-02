"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          VIETNAM COASTAL EROSION — FULL ML DATASET BUILDER                 ║
║          Objectif : prédire retreat_rate (m/an) par régression             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Sources                                                                   ║
║  ├── CoastSat output.pkl    → shorelines, dates, NDVI/NDWI                 ║
║  ├── CoastSat metadata.pkl  → cloud_cover, satellite                       ║
║  ├── Images Sentinel-2      → NDVI, NDWI par bande                        ║
║  ├── ERA5 (Copernicus CDS)  → vagues, vent                                ║
║  └── Google Earth Engine    → marée, pente SRTM, rivières HydroSHEDS      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Variables produites                                                       ║
║  ├── Identifiants    : segment_id, site, date, month, season               ║
║  ├── Spatial         : lat, lon, orientation, in_river_zone                ║
║  ├── Observation     : cloud_cover, satellite, days_since_prev             ║
║  ├── Shoreline       : cross_distance, change, cumul                       ║
║  ├── TARGET          : retreat_rate_m_yr  (régression ML)                 ║
║  ├── Vagues ERA5     : Hs_mean_7d, Hs_max_7d, wave_dir, wave_period       ║
║  ├── Vent ERA5       : wind_mean_7d, wind_dir_7d                          ║
║  ├── Marée GEE       : tide_height, tide_range                            ║
║  ├── Topo GEE/SRTM   : slope_deg, elevation_m                             ║
║  ├── Végétation S2   : NDVI, NDWI                                         ║
║  └── GEE HydroSHEDS  : dist_river_m                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP REQUIS
────────────
1. ERA5  →  pip install cdsapi
            Créer ~/.cdsapirc :
              url: https://cds.climate.copernicus.eu/api/v2
              key: <UID>:<API_KEY>

2. GEE   →  pip install earthengine-api
            earthengine authenticate

3. CoastSat dans ton PYTHONPATH :
            git clone https://github.com/kvos/CoastSat
            export PYTHONPATH=$PYTHONPATH:/chemin/CoastSat

LANCER
──────
   python build_full_dataset.py
   # Désactiver une source :  ERA5_ENABLED = False  /  GEE_ENABLED = False

────────────────────────────────────────────────────────────────────────────
CORRECTIFS (ce fichier) :
   1) Les shorelines CoastSat (output['shorelines'][i]) contiennent souvent
      PLUSIEURS morceaux de côte disjoints (nuages, îles, gaps de détection,
      bords d'image, embouchures...). Relier tous les points consécutifs par
      une LineString unique crée des "coutures" artificielles en ligne droite
      qui traversent la terre. build_transects() découpe maintenant la
      shoreline en sous-segments continus (split_shoreline_segments) AVANT de
      générer les transects.

   2) Sur les sites à large emprise, CoastSat détecte parfois un réseau de
      bassins d'aquaculture / canaux interconnectés comme une SEULE masse
      d'eau continue (les bassins communiquent entre eux). Le contour de
      cette masse d'eau est alors un contour réellement continu — donc non
      coupé par le correctif (1) — mais qui serpente sur des dizaines de km
      à l'intérieur d'une petite zone 2D, au lieu de suivre une ligne de
      côte. split_shoreline_segments() calcule maintenant l'élongation de
      chaque segment (PCA : rapport axe principal / axe secondaire) et
      écarte ceux qui ne sont pas suffisamment allongés (< MIN_SEGMENT_
      ELONGATION, 4.0 par défaut) : une vraie côte est fine et allongée,
      un réseau de bassins forme un enchevêtrement compact quelle que soit
      sa longueur totale. Le filtre par longueur minimale (MIN_SEGMENT_
      LENGTH_M) reste utile en complément pour les petits artefacts isolés.
────────────────────────────────────────────────────────────────────────────
"""

import os, sys, json, pickle, logging, warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString, Point

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DATA_ROOT   = Path(os.getcwd()) / 'data'
OUTPUT_CSV  = DATA_ROOT / 'vietnam_ml_dataset.csv'
CACHE_DIR   = DATA_ROOT / '_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRANSECT_SPACING = 100    # mètres
TRANSECT_LENGTH  = 200    # mètres

# Distance max (en mètres, dans le CRS EPSG_COASTSAT) entre deux points
# consécutifs d'une shoreline pour être considérés comme le MEME morceau
# de côte. Au-delà : on considère qu'il y a un gap / une couture -> on coupe.
# A ajuster selon la résolution Sentinel-2 (10-15 m/pixel) : une vraie
# discontinuité de détection saute généralement de plusieurs dizaines à
# centaines de mètres d'un coup.
MAX_SHORELINE_GAP_M = 75

# Longueur minimale (m) d'un sous-segment de shoreline, APRES découpage par
# gap, pour être conservé comme "vraie côte". En dessous de ce seuil, on
# considère qu'il s'agit d'un artefact de classification eau/terre : bord
# de bassin d'aquaculture, rizière inondée, mare, piscine, etc. — ces
# structures rectangulaires très fréquentes dans les deltas vietnamiens
# créent des quadrillages de "shorelines" à l'intérieur des terres.
# A ajuster selon le site : augmenter si des morceaux de vraie côte
# (petites baies, anses) sont encore filtrés à tort.
MIN_SEGMENT_LENGTH_M = 500

# Élongation minimale (ratio écart-type axe principal / axe secondaire,
# via PCA) pour qu'un segment soit considéré comme "de la vraie côte".
# Une vraie côte est un tracé fin et allongé dans une direction dominante.
# Un réseau de bassins d'aquaculture/canaux interconnectés forme au
# contraire un enchevêtrement compact dans une zone 2D : le trajet peut
# être très long (des dizaines de km en serpentant), mais reste confiné
# dans un petit périmètre -> élongation proche de 1. On écarte donc tout
# segment dont l'élongation est < ce seuil, quelle que soit sa longueur.
MIN_SEGMENT_ELONGATION = 4.0

EPSG_COASTSAT = 'EPSG:3857'
EPSG_WGS84    = 'EPSG:4326'
EPSG_UTM48N   = 'EPSG:32648'

ERA5_ENABLED = True
GEE_ENABLED  = True
GEE_PROJECT  = 'vietnam-coastal-erosion'   # ← ton projet GEE

SITES_BBOX = {
    'Hoi_An_IA':      [108.36, 15.86, 108.40, 15.90],
    'Da_Nang_My_Khe': [108.24, 16.05, 108.27, 16.08],
    'Mui_Ne':         [108.18, 10.92, 108.22, 10.95],
    'QN_Dien_Ban':    [108.30, 15.88, 108.36, 15.95],
    'QN_Thang_Binh':  [108.40, 15.65, 108.52, 15.85],
    'QN_Tam_Thanh':   [108.50, 15.50, 108.62, 15.68],
    'QN_Nui_Thanh':   [108.60, 15.40, 108.75, 15.55],
}

RIVER_EXCLUSION_ZONES = [
    [108.32, 15.85, 108.42, 15.98],
    [108.27, 16.00, 108.35, 16.10],
    [108.10, 15.72, 108.20, 15.82],
    [108.44, 15.62, 108.52, 15.70],
    [108.50, 15.47, 108.58, 15.55],
]

_to_wgs = pyproj.Transformer.from_crs(EPSG_COASTSAT, EPSG_WGS84, always_xy=True)
_to_utm = pyproj.Transformer.from_crs(EPSG_WGS84, EPSG_UTM48N,  always_xy=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — GÉOMÉTRIE & TRANSECTS
# ══════════════════════════════════════════════════════════════════════════════

def segment_elongation(seg):
    """
    Rapport d'élongation d'un segment via PCA : écart-type le long de
    l'axe principal / écart-type le long de l'axe secondaire.

    - Une ligne fine et allongée (vraie côte) a une élongation élevée
      (typiquement > 5-10) : la variance est concentrée sur un seul axe.
    - Un enchevêtrement compact dans une zone 2D (réseau de bassins
      d'aquaculture/canaux interconnectés) a une élongation proche de 1,
      même si le trajet total (la longueur du contour) est très long.
    """
    pts = np.asarray(seg, dtype=float)
    if len(pts) < 3:
        return np.inf  # trop peu de points pour juger -> on ne filtre pas dessus
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)  # ordre croissant
    minor, major = max(eigvals[0], 1e-9), eigvals[1]
    return float(np.sqrt(major / minor))


def split_shoreline_segments(shoreline_xy, max_gap=MAX_SHORELINE_GAP_M,
                              min_length=MIN_SEGMENT_LENGTH_M,
                              min_elongation=MIN_SEGMENT_ELONGATION):
    """
    Découpe un tableau de points de shoreline en plusieurs sous-segments
    CONTINUS, en coupant partout où la distance entre deux points
    consécutifs dépasse `max_gap`.

    Ceci évite que shapely.LineString relie par une ligne droite deux
    morceaux de côte totalement déconnectés (nuages, îles, gaps de
    détection, bords d'image, embouchures...), ce qui créerait des
    transects parasites en ligne droite traversant la terre.

    Filtre ensuite les segments qui ne ressemblent pas à une côte :
    - trop courts (< min_length) : petits artefacts isolés.
    - pas assez élancés (< min_elongation) : réseaux de bassins
      d'aquaculture / canaux interconnectés, qui forment un contour
      continu (donc non coupé par le filtre de gap) mais serpentent dans
      une zone 2D compacte plutôt que de suivre une ligne de côte.

    Retourne une liste de np.ndarray (chacun de shape (N, 2), N >= 2).
    """
    pts = np.asarray(shoreline_xy, dtype=float)
    if len(pts) < 2:
        return []

    dists  = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    breaks = np.where(dists > max_gap)[0] + 1
    raw_segments = np.split(pts, breaks)

    segments = []
    n_dropped_short = 0
    n_dropped_blob  = 0
    for seg in raw_segments:
        if len(seg) < 2:
            continue
        seg_len = np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])).sum()
        if seg_len < min_length:
            n_dropped_short += 1
            continue
        elong = segment_elongation(seg)
        if elong < min_elongation:
            n_dropped_blob += 1
            continue
        segments.append(seg)

    if n_dropped_short or n_dropped_blob:
        log.info(f"    split_shoreline_segments : {n_dropped_short} segment(s) "
                 f"courts (< {min_length} m) + {n_dropped_blob} segment(s) "
                 f"non-linéaires (élongation < {min_elongation}, probables "
                 f"bassins/canaux) écartés")

    return segments


def build_transects(shoreline_xy):
    """
    Construit les transects perpendiculaires à la côte, tous les
    TRANSECT_SPACING mètres, sur CHAQUE sous-segment continu de la
    shoreline (voir split_shoreline_segments).
    """
    segments = split_shoreline_segments(shoreline_xy)
    if not segments:
        log.warning("    build_transects : aucune shoreline exploitable après découpage")
        return {}

    transects = {}
    idx = 0
    n_skipped = 0

    for seg_pts in segments:
        line = LineString(seg_pts)
        if line.length <= 0:
            continue

        dists = np.arange(0, line.length, TRANSECT_SPACING)
        for d in dists:
            idx += 1
            p1 = line.interpolate(d)
            p2 = line.interpolate(min(d + 0.5, line.length))
            dx, dy = p2.x - p1.x, p2.y - p1.y
            mag    = max(np.hypot(dx, dy), 1e-9)
            ux, uy = -dy / mag, dx / mag
            start  = [p1.x - ux * TRANSECT_LENGTH/2, p1.y - uy * TRANSECT_LENGTH/2]
            end    = [p1.x + ux * TRANSECT_LENGTH/2, p1.y + uy * TRANSECT_LENGTH/2]
            transects[f'TS_{idx:04d}'] = np.array([start, end])

    n_segments = len(segments)
    if n_segments > 1:
        log.info(f"    build_transects : shoreline découpée en {n_segments} segments "
                 f"continus (gap max toléré = {MAX_SHORELINE_GAP_M} m) → {idx} transects")
    else:
        log.info(f"    build_transects : {idx} transects (1 segment continu)")

    return transects


def transect_metadata(transects):
    meta = {}
    for key, c in transects.items():
        mx, my = (c[0][0]+c[1][0])/2, (c[0][1]+c[1][1])/2
        lon, lat = _to_wgs.transform(mx, my)
        orient = float(np.degrees(np.arctan2(c[1][0]-c[0][0], c[1][1]-c[0][1])) % 360)
        in_river = any(
            z[0] <= lon <= z[2] and z[1] <= lat <= z[3]
            for z in RIVER_EXCLUSION_ZONES
        )
        meta[key] = {
            'latitude':     round(lat, 6),
            'longitude':    round(lon, 6),
            'orientation':  round(orient, 1),
            'in_river_zone': int(in_river),
        }
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — COASTSAT INTERSECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_distances(output, transects):
    try:
        from coastsat import SDS_transects
        settings = {
            'along_dist': 25, 'min_points': 3, 'max_std': 15,
            'max_range': 30, 'min_chainage': -100,
            'multiple_inter': 'auto', 'auto_prc': 0.1,
        }
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        cross = SDS_transects.compute_intersection_QC(output, transects, settings)
        sys.stderr.close()
        sys.stderr = _stderr
        return cross
    except ImportError:
        log.warning("    coastsat absent — fallback géométrique")
        return _geometric_intersection(output, transects)


def _geometric_intersection(output, transects):
    dates, shorelines = output['dates'], output['shorelines']
    cross = {k: np.full(len(dates), np.nan) for k in transects}
    for i, sl in enumerate(shorelines):
        if sl is None or len(sl) < 2:
            continue
        sl_line = LineString(sl)
        for key, coords in transects.items():
            ts_line = LineString(coords)
            try:
                inter = sl_line.intersection(ts_line)
                if inter.is_empty: continue
                pt = inter if inter.geom_type == 'Point' else list(inter.geoms)[0]
                cross[key][i] = ts_line.project(pt)
            except Exception:
                continue
    return cross


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SÉRIES TEMPORELLES
# ══════════════════════════════════════════════════════════════════════════════

def retreat_rate_linreg(dates, distances):
    """Régression linéaire → (rate m/yr, r²). TARGET ML."""
    pairs = [(d, float(v)) for d, v in zip(dates, distances)
             if v is not None and not np.isnan(float(v))]
    if len(pairs) < 3: return np.nan, np.nan
    t0 = pairs[0][0]
    t  = np.array([(d-t0).days / 365.25 for d, _ in pairs])
    y  = np.array([v for _, v in pairs])
    if t.std() < 1e-9: return np.nan, np.nan
    coeffs = np.polyfit(t, y, 1)
    y_pred = np.polyval(coeffs, t)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan
    return round(float(coeffs[0]), 4), round(r2, 4)


def temporal_features(dates, distances):
    arr   = np.array(distances, dtype=float)
    n     = len(arr)
    chg   = np.full(n, np.nan)
    cumul = np.full(n, np.nan)
    days  = np.full(n, np.nan)
    first = next((i for i, v in enumerate(arr) if not np.isnan(v)), None)
    if first is None: return chg, cumul, days
    for i in range(1, n):
        chg[i]   = arr[i] - arr[i-1]
        cumul[i] = arr[i] - arr[first]
        dt       = dates[i] - dates[i-1]
        days[i] = dt.total_seconds() / 86400.0
    return chg, cumul, days


def season_vietnam(month):
    if month in [11,12,1,2]: return 1  # mousson NE
    if month in [3,4,5]:     return 2  # transition printemps
    if month in [6,7,8,9]:   return 3  # mousson SW
    return 4                            # transition automne


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — NDVI / NDWI  (images Sentinel-2)
# ══════════════════════════════════════════════════════════════════════════════

def extract_spectral_indices(output, site_path):
    """Retourne dict {date_str: {NDVI, NDWI}}. Essaie im_ms puis .tif."""
    result = {}

    # Stratégie 1 : arrays im_ms en mémoire
    if 'im_ms' in output and output.get('im_ms'):
        for date, im in zip(output.get('dates', []), output['im_ms']):
            try:
                if not isinstance(im, np.ndarray) or im.ndim != 3 or im.shape[2] < 4:
                    continue
                red, green, nir = im[:,:,0].astype(float), im[:,:,1].astype(float), im[:,:,3].astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = np.where((nir+red)>0,   (nir-red)/(nir+red),   np.nan)
                    ndwi = np.where((green+nir)>0, (green-nir)/(green+nir), np.nan)
                ds = str(pd.Timestamp(date).date())
                result[ds] = {'NDVI': round(float(np.nanmedian(ndvi)),4),
                              'NDWI': round(float(np.nanmedian(ndwi)),4)}
            except Exception:
                continue
        if result:
            log.info(f"    NDVI/NDWI : {len(result)} dates (im_ms)")
            return result

    # Stratégie 2 : fichiers .tif sur disque
    try:
        import rasterio
        tifs = list(site_path.glob('**/*ms*.tif')) + list(site_path.glob('**/*S2*.tif'))
        for tif in tifs:
            try:
                with rasterio.open(str(tif)) as src:
                    if src.count < 4: continue
                    red   = src.read(1).astype(float)
                    green = src.read(2).astype(float)
                    nir   = src.read(4).astype(float)
                    nd = src.nodata or 0
                    red[red==nd] = np.nan; nir[nir==nd] = np.nan
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndvi = np.where((nir+red)>0,   (nir-red)/(nir+red),   np.nan)
                        ndwi = np.where((green+nir)>0, (green-nir)/(green+nir), np.nan)
                    parts = tif.stem.split('_')
                    date_str = next((p for p in parts if len(p)==8 and p.isdigit()), None)
                    if date_str:
                        ds = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
                        result[ds] = {'NDVI': round(float(np.nanmedian(ndvi)),4),
                                      'NDWI': round(float(np.nanmedian(ndwi)),4)}
            except Exception:
                continue
        if result:
            log.info(f"    NDVI/NDWI : {len(result)} dates (.tif)")
    except ImportError:
        log.warning("    rasterio absent — NDVI/NDWI = NaN")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — ERA5  (Copernicus CDS)
# ══════════════════════════════════════════════════════════════════════════════

import zipfile

def _fetch_era5_one_year(c, year, area, tmp_path):
    """Télécharge ERA5 pour une seule année. Retourne DataFrame ou None."""
    c.retrieve('reanalysis-era5-single-levels', {
        'product_type': 'reanalysis',
        'variable': [
            'significant_height_of_combined_wind_waves_and_swell',
            'mean_wave_direction', 'peak_wave_period',
            '10m_u_component_of_wind', '10m_v_component_of_wind',
        ],
        'year':  [str(year)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'day':   [f'{d:02d}' for d in range(1, 32)],
        'time':  ['00:00', '06:00', '12:00', '18:00'],
        'area':  area,
        'format': 'netcdf',
    }, str(tmp_path))

    # 1. Extraction de TOUS les fichiers .nc du ZIP
    nc_files_extracted = []
    if zipfile.is_zipfile(tmp_path):
        with zipfile.ZipFile(tmp_path, 'r') as z:
            nc_files = [f for f in z.namelist() if f.endswith('.nc')]
            for f in nc_files:
                z.extract(f, path=tmp_path.parent)
                nc_files_extracted.append(tmp_path.parent / f)
        tmp_path.unlink() # Nettoyage du zip original
    else:
        nc_files_extracted.append(tmp_path)

    import netCDF4 as nc4

    data_dict = {}
    ts = None

    # 2. Parcourir chaque fichier extrait pour trouver les variables éparpillées
    for nc_file in nc_files_extracted:
        ds = nc4.Dataset(str(nc_file))

        # Récupération de l'axe temporel (on ne le fait qu'une fois)
        if ts is None:
            if 'time' in ds.variables: t_var = 'time'
            elif 'valid_time' in ds.variables: t_var = 'valid_time'
            else: t_var = next((v for v in ds.variables if 'time' in v.lower()), None)

            if t_var:
                ts = nc4.num2date(ds[t_var][:], ds[t_var].units)

        # Extraction des variables cibles si elles sont présentes dans CE fichier
        for v_target in ['swh', 'mwd', 'pp1d', 'u10', 'v10']:
            if v_target in ds.variables:
                d = ds[v_target][:]
                a = d.data.astype(float)
                if hasattr(d, 'mask'): a[d.mask] = np.nan
                # Moyenne spatiale si 3D (time, lat, lon)
                data_dict[v_target] = a.mean(axis=tuple(range(1, a.ndim))) if a.ndim > 1 else a

        ds.close()
        nc_file.unlink(missing_ok=True) # Nettoyage immédiat du fichier .nc

    # 3. Sécurité : remplacer par des NaN si une variable n'a été trouvée dans aucun fichier
    for v in ['swh', 'mwd', 'pp1d', 'u10', 'v10']:
        if v not in data_dict:
            data_dict[v] = np.full(len(ts), np.nan) if ts is not None else []

    u10 = data_dict['u10']
    v10 = data_dict['v10']

    # 4. Construction du DataFrame final
    df = pd.DataFrame({
        'datetime':   pd.to_datetime([t.strftime('%Y-%m-%d %H:%M') for t in ts]) if ts is not None else [],
        'swh':        data_dict['swh'],
        'mwd':        data_dict['mwd'],
        'pp1d':       data_dict['pp1d'],
        'wind_speed': np.hypot(u10, v10),
        'wind_dir':   (np.degrees(np.arctan2(u10, v10)) + 360) % 360,
    }).set_index('datetime')

    return df

def fetch_era5(site_name, dates, bbox):
    """
    Télécharge ERA5 année par année (limite CDS) puis fusionne.
    Cache dans CACHE_DIR/<site>_era5.pkl.
    """
    cache = CACHE_DIR / f'{site_name}_era5.pkl'
    if cache.exists():
        df = pd.read_pickle(cache)
        log.info(f"    ERA5 : cache ({len(df)} entrées)")
        return df

    try:
        import cdsapi
    except ImportError:
        log.warning("    cdsapi absent (pip install cdsapi) — ERA5 = NaN")
        return pd.DataFrame()

    # N W S E, arrondi au 0.25° ERA5
    area  = [np.ceil(bbox[3]*4)/4, np.floor(bbox[0]*4)/4,
             np.floor(bbox[1]*4)/4, np.ceil(bbox[2]*4)/4]
    years = sorted({pd.Timestamp(d).year for d in dates})
    log.info(f"    ERA5 : {len(years)} années à télécharger ({years[0]}–{years[-1]}) …")

    try:
        c      = cdsapi.Client(quiet=True)
        frames = []

        for year in years:
            # Cache intermédiaire par année
            yr_cache = CACHE_DIR / f'{site_name}_era5_{year}.pkl'
            if yr_cache.exists():
                frames.append(pd.read_pickle(yr_cache))
                log.info(f"    ERA5 {year} : cache")
                continue

            tmp = CACHE_DIR / f'{site_name}_era5_{year}_raw.nc'
            try:
                log.info(f"    ERA5 {year} : téléchargement …")
                df_yr = _fetch_era5_one_year(c, year, area, tmp)
                df_yr.to_pickle(yr_cache)
                frames.append(df_yr)
                log.info(f"    ERA5 {year} : {len(df_yr)} entrées OK")
            except Exception as e:
                log.warning(f"    ERA5 {year} échoué : {e}")
                tmp.unlink(missing_ok=True)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index().drop_duplicates()
        df.to_pickle(cache)
        log.info(f"    ERA5 : {len(df)} entrées totales sauvegardées")
        return df

    except Exception as e:
        log.warning(f"    ERA5 échoué : {e}")
        return pd.DataFrame()


def era5_window(df_era5, date, days=7):
    empty = {'Hs_mean_7d':np.nan,'Hs_max_7d':np.nan,'wave_dir':np.nan,
             'wave_period':np.nan,'wind_mean':np.nan,'wind_dir':np.nan}
    if df_era5.empty: return empty
    ts = pd.Timestamp(date).tz_localize(None)
    win = df_era5.loc[ts - pd.Timedelta(days=days) : ts]
    if win.empty: return empty
    return {
        'Hs_mean_7d':  round(float(win['swh'].mean()),  3),
        'Hs_max_7d':   round(float(win['swh'].max()),   3),
        'wave_dir':    round(float(win['mwd'].mean()),  1),
        'wave_period': round(float(win['pp1d'].mean()), 2),
        'wind_mean':   round(float(win['wind_speed'].mean()), 3),
        'wind_dir':    round(float(win['wind_dir'].mean()),   1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — GOOGLE EARTH ENGINE  (marée, slope SRTM, dist_river HydroSHEDS)
# ══════════════════════════════════════════════════════════════════════════════

_gee_ok = False

def _init_gee():
    global _gee_ok
    if _gee_ok: return True
    try:
        import ee
        try:    ee.Initialize(project=GEE_PROJECT)
        except: ee.Authenticate(); ee.Initialize(project=GEE_PROJECT)
        _gee_ok = True
        log.info("    GEE : initialisé")
        return True
    except Exception as e:
        log.warning(f"    GEE : échec ({e})")
        return False


def gee_static_features(site_name, bbox, ts_meta):
    """slope (SRTM), elevation (SRTM), dist_river_m (HydroSHEDS) par transect."""
    cache = CACHE_DIR / f'{site_name}_gee_static.json'
    if cache.exists():
        with open(cache) as f:
            log.info("    GEE static : cache chargé")
            return json.load(f)
    if not GEE_ENABLED or not _init_gee(): return {}

    try:
        import ee
        srtm      = ee.Image('USGS/SRTMGL1_003')
        slope_im  = ee.Terrain.slope(srtm)
        # HydroSHEDS : distance au réseau hydrographique principal
        rivers    = ee.Image('WWF/HydroSHEDS/15ACC').gt(100)
        dist_riv  = rivers.fastDistanceTransform().sqrt().multiply(
            ee.Image.pixelArea().sqrt()
        )
        combo = ee.Image.cat([slope_im, srtm, dist_riv]) \
                       .rename(['slope','elevation','dist_river'])

        result = {}
        for key, m in ts_meta.items():
            pt = ee.Geometry.Point([m['longitude'], m['latitude']])
            try:
                vals = combo.sample(pt, 30).first().toDictionary().getInfo()
                result[key] = {
                    'slope':        round(vals.get('slope',       np.nan), 4),
                    'elevation':    round(vals.get('elevation',   np.nan), 2),
                    'dist_river_m': round(vals.get('dist_river',  np.nan), 1),
                }
            except Exception:
                result[key] = {'slope': np.nan, 'elevation': np.nan, 'dist_river_m': np.nan}

        with open(cache, 'w') as f: json.dump(result, f)
        log.info(f"    GEE static : {len(result)} transects")
        return result
    except Exception as e:
        log.warning(f"    GEE static échoué : {e}")
        return {}


def gee_tide(site_name, lat, lon, dates):
    """Hauteur de marée instantanée (HYCOM sea surface elevation) par date."""
    cache = CACHE_DIR / f'{site_name}_gee_tide.json'
    if cache.exists():
        with open(cache) as f:
            log.info("    GEE marée : cache chargé")
            return json.load(f)
    if not GEE_ENABLED or not _init_gee(): return {}

    try:
        import ee
        pt     = ee.Geometry.Point([lon, lat])
        result = {}

        for date in dates:
            ts = pd.Timestamp(date)
            ds = str(ts.date())
            t1 = (ts - timedelta(hours=12)).strftime('%Y-%m-%d')
            t2 = (ts + timedelta(hours=12)).strftime('%Y-%m-%d')
            try:
                col = (ee.ImageCollection('HYCOM/sea_surface_elevation')
                         .filterBounds(pt)
                         .filterDate(t1, t2))
                n = col.size().getInfo()
                if n == 0:
                    result[ds] = {'tide_height': np.nan, 'tide_range': np.nan}
                    continue
                heights = []
                for img in col.toList(n).getInfo():
                    try:
                        v = (ee.Image(img['id'])
                               .sample(pt, 1000).first()
                               .toDictionary().getInfo())
                        # HYCOM variable name
                        h = v.get('water_temp_salinity_0', v.get('surf_el', None))
                        if h is not None: heights.append(float(h))
                    except Exception:
                        continue
                result[ds] = {
                    'tide_height': round(float(np.mean(heights)), 3) if heights else np.nan,
                    'tide_range':  round(float(np.ptp(heights)),  3) if heights else np.nan,
                }
            except Exception:
                result[ds] = {'tide_height': np.nan, 'tide_range': np.nan}

        with open(cache, 'w') as f: json.dump(result, f)
        log.info(f"    GEE marée : {len(result)} dates")
        return result
    except Exception as e:
        log.warning(f"    GEE marée échoué : {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — METADATA CoastSat (cloud_cover, satellite)
# ══════════════════════════════════════════════════════════════════════════════

def select_reference_date_index(dates, obs_meta):
    """
    Choisit l'index de la date à utiliser comme référence géométrique pour
    build_transects(), au lieu de systématiquement prendre dates[0].

    Critère : cloud_cover minimal parmi les dates avec georef_ok == 1.
    Si aucune métadonnée n'est disponible (obs_meta vide), retombe sur 0
    (comportement d'origine).
    """
    if not obs_meta:
        return 0, None

    candidates = []
    for i, d in enumerate(dates):
        ds = str(pd.Timestamp(d).date())
        m = obs_meta.get(ds)
        if m is None:
            continue
        cc = m.get('cloud_cover', np.nan)
        georef_ok = m.get('georef_ok', np.nan)
        if np.isnan(cc):
            continue
        # Pénalise fortement les dates avec géoréférencement raté
        penalty = 0.0 if georef_ok == 1 else 1.0
        candidates.append((i, cc + penalty, cc))

    if not candidates:
        return 0, None

    best_i, _, best_cc = min(candidates, key=lambda x: x[1])
    return best_i, best_cc


def load_obs_metadata(site_path, site_name):
    """
    Retourne dict {date_str: {cloud_cover, satellite, georef_ok}}.

    Structure CoastSat metadata.pkl :
      {satellite: {dates, filenames, acc_georef, epsg, ...}}
      - acc_georef : liste de 'PASSED'/'FAILED' (qualité géoréférencement)
      - cloud_cover : liste de floats [0-1], parfois absent
    """
    pkl = site_path / f'{site_name}_metadata.pkl'
    if not pkl.exists(): return {}
    try:
        with open(pkl,'rb') as f: meta = pickle.load(f)
        result = {}
        if not isinstance(meta, dict): return {}

        for sat, sm in meta.items():
            if not isinstance(sm, dict): continue
            dates_sat   = sm.get('dates', [])
            acc_georef  = sm.get('acc_georef', [])   # 'PASSED' / 'FAILED'
            cloud_cover = sm.get('cloud_cover', [])  # floats ou absent

            for i, d in enumerate(dates_sat):
                ds = str(pd.Timestamp(d).date())

                # acc_georef → booléen georef_ok
                georef_val = acc_georef[i] if i < len(acc_georef) else None
                if isinstance(georef_val, str):
                    georef_ok = int(georef_val.upper() == 'PASSED')
                elif georef_val is not None:
                    try:    georef_ok = int(bool(float(georef_val)))
                    except: georef_ok = np.nan
                else:
                    georef_ok = np.nan

                # cloud_cover → float [0-1]
                cc_val = cloud_cover[i] if i < len(cloud_cover) else None
                try:    cc = float(cc_val) if cc_val is not None else np.nan
                except: cc = np.nan

                result[ds] = {
                    'cloud_cover': cc,
                    'georef_ok':   georef_ok,
                    'satellite':   sat,
                }
        log.info(f"    metadata.pkl : {len(result)} dates chargées")
        return result
    except Exception as e:
        log.warning(f"    metadata.pkl : {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — TRAITEMENT COMPLET D'UN SITE
# ══════════════════════════════════════════════════════════════════════════════

def process_site(site_name):
    site_path = DATA_ROOT / site_name
    ckpt      = CACHE_DIR / f'{site_name}_processed.pkl'

    if ckpt.exists():
        log.info(f"[CACHE]   {site_name}")
        return pd.read_pickle(ckpt)

    pkl_file = site_path / f'{site_name}_output.pkl'
    if not pkl_file.exists():
        log.warning(f"[SKIP]    {site_name} — output.pkl absent")
        return None

    log.info(f"[START]   {site_name}")
    with open(pkl_file,'rb') as f: output = pickle.load(f)

    dates      = [pd.Timestamp(d) for d in output['dates']]
    shorelines = output['shorelines']
    if not dates or not shorelines:
        log.warning(f"[SKIP]    {site_name} — données vides"); return None

    # Metadata observations (chargées AVANT les transects pour pouvoir
    # choisir la meilleure date de référence géométrique)
    obs_meta = load_obs_metadata(site_path, site_name)

    # Choix de la date de référence : la moins nuageuse plutôt que la
    # première venue (shorelines[0] peut être une date de mauvaise
    # qualité, ce qui produit une géométrie de côte incomplète/erratique)
    ref_idx, ref_cc = select_reference_date_index(dates, obs_meta)
    if ref_cc is not None:
        log.info(f"          date de référence géométrique : {dates[ref_idx].date()} "
                 f"(index {ref_idx}, cloud_cover={ref_cc:.2f})")
    else:
        log.info(f"          date de référence géométrique : {dates[ref_idx].date()} "
                 f"(index {ref_idx}, pas de metadata cloud_cover disponible)")

    # Transects (construits sur la date de référence choisie ci-dessus,
    # et robustes aux shorelines discontinues / réseaux de bassins grâce
    # au split par gap + filtre longueur/élongation)
    transects = build_transects(shorelines[ref_idx])
    if not transects:
        log.warning(f"[SKIP]    {site_name} — aucun transect généré"); return None
    ts_meta   = transect_metadata(transects)
    log.info(f"          {len(transects)} transects × {len(dates)} dates")

    # Cross distances
    cross = compute_cross_distances(output, transects)

    # NDVI/NDWI
    spectral = extract_spectral_indices(output, site_path)

    # ERA5
    bbox    = SITES_BBOX.get(site_name, [108.0,10.0,109.0,17.0])
    df_era5 = fetch_era5(site_name, dates, bbox) if ERA5_ENABLED else pd.DataFrame()

    # GEE static
    gee_static = gee_static_features(site_name, bbox, ts_meta)

    # GEE marée (centroïde du site)
    lats = [v['latitude']  for v in ts_meta.values()]
    lons = [v['longitude'] for v in ts_meta.values()]
    tide_map = gee_tide(site_name, float(np.mean(lats)), float(np.mean(lons)), dates) \
               if GEE_ENABLED else {}

    # Construction du tableau long
    rows = []
    for ts_key, dist_array in cross.items():
        m        = ts_meta.get(ts_key, {})
        gee_s    = gee_static.get(ts_key, {})
        dist_arr = np.array(dist_array, dtype=float)

        rate, r2         = retreat_rate_linreg(dates, dist_arr)
        chg, cumul, days = temporal_features(dates, dist_arr)

        for i, date in enumerate(dates):
            ds     = str(date.date())
            era5_v = era5_window(df_era5, date)
            ndvi_v = spectral.get(ds, {})
            tide_v = tide_map.get(ds, {})
            obs_v  = obs_meta.get(ds, {})

            rows.append({
                # Identifiants
                'segment_id':         f'{site_name}_{ts_key}',
                'site_name':          site_name,
                'date':               ds,
                'month':              date.month,
                'season':             season_vietnam(date.month),
                # Spatial
                'latitude':           m.get('latitude',    np.nan),
                'longitude':          m.get('longitude',   np.nan),
                'orientation_deg':    m.get('orientation', np.nan),
                'in_river_zone':      m.get('in_river_zone', 0),
                # Observation
                'cloud_cover':        obs_v.get('cloud_cover', np.nan),
                'georef_ok':          obs_v.get('georef_ok',   np.nan),
                'satellite':          obs_v.get('satellite',   'S2'),
                'days_since_prev':    days[i],
                # Shoreline
                'cross_distance_m':   dist_arr[i],
                'shoreline_change_m': chg[i],
                'cumul_change_m':     cumul[i],
                # TARGET
                'retreat_rate_m_yr':  rate,
                'retreat_r2':         r2,
                # Vagues ERA5
                'Hs_mean_7d':         era5_v['Hs_mean_7d'],
                'Hs_max_7d':          era5_v['Hs_max_7d'],
                'wave_dir_deg':       era5_v['wave_dir'],
                'wave_period_s':      era5_v['wave_period'],
                # Vent ERA5
                'wind_mean_7d':       era5_v['wind_mean'],
                'wind_dir_deg':       era5_v['wind_dir'],
                # Marée GEE
                'tide_height_m':      tide_v.get('tide_height', np.nan),
                'tide_range_m':       tide_v.get('tide_range',  np.nan),
                # Topographie GEE/SRTM
                'slope_deg':          gee_s.get('slope',       np.nan),
                'elevation_m':        gee_s.get('elevation',   np.nan),
                # Végétation S2
                'NDVI':               ndvi_v.get('NDVI', np.nan),
                'NDWI':               ndvi_v.get('NDWI', np.nan),
                # GIS
                'dist_river_m':       gee_s.get('dist_river_m', np.nan),
            })

    df = pd.DataFrame(rows)
    df.to_pickle(ckpt)
    log.info(f"[DONE]    {site_name} → {len(df):,} lignes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

COLUMN_ORDER = [
    'segment_id','site_name','date','month','season',
    'latitude','longitude','orientation_deg','in_river_zone',
    'cloud_cover','georef_ok','satellite','days_since_prev',
    'cross_distance_m','shoreline_change_m','cumul_change_m',
    'retreat_rate_m_yr','retreat_r2',
    'Hs_mean_7d','Hs_max_7d','wave_dir_deg','wave_period_s',
    'wind_mean_7d','wind_dir_deg',
    'tide_height_m','tide_range_m',
    'slope_deg','elevation_m',
    'NDVI','NDWI',
    'dist_river_m',
]

if __name__ == '__main__':
    log.info("═"*65)
    log.info("  VIETNAM COASTAL ML DATASET BUILDER")
    log.info(f"  ERA5={'ON' if ERA5_ENABLED else 'OFF'}   GEE={'ON' if GEE_ENABLED else 'OFF'}")
    log.info(f"  MAX_SHORELINE_GAP_M = {MAX_SHORELINE_GAP_M} m")
    log.info(f"  MIN_SEGMENT_LENGTH_M = {MIN_SEGMENT_LENGTH_M} m")
    log.info(f"  MIN_SEGMENT_ELONGATION = {MIN_SEGMENT_ELONGATION}")
    log.info("═"*65)

    sites = sorted([d.name for d in DATA_ROOT.iterdir()
                    if d.is_dir() and not d.name.startswith('_')])
    log.info(f"  {len(sites)} sites : {', '.join(sites)}")

    results = []
    for site in sites:
        try:
            df = process_site(site)
            if df is not None: results.append(df)
        except Exception as e:
            log.error(f"[ERROR]   {site} : {e}", exc_info=True)

    if not results:
        log.error("Aucun site traité."); sys.exit(1)

    master = pd.concat(results, ignore_index=True)
    extra  = [c for c in master.columns if c not in COLUMN_ORDER]
    master = master[COLUMN_ORDER + extra]
    DATA_ROOT.mkdir(exist_ok=True)
    master.to_csv(OUTPUT_CSV, index=False)

    n_total = len(master)
    n_river = int(master['in_river_zone'].sum())
    n_clean = n_total - n_river
    pct_nan = (master.isna().mean()*100).round(1)

    log.info("═"*65)
    log.info(f"  OUTPUT    : {OUTPUT_CSV}")
    log.info(f"  Lignes    : {n_total:,}  ({n_river:,} river, {n_clean:,} ML-ready)")
    log.info(f"  Sites     : {master['site_name'].nunique()}")
    log.info(f"  Transects : {master['segment_id'].nunique()}")
    log.info("")
    log.info("  NaN par variable :")
    for col in COLUMN_ORDER[4:]:
        p    = pct_nan.get(col, 0)
        flag = " ⚠ source externe manquante" if p > 80 else ""
        log.info(f"    {col:<28s} {p:5.1f}%{flag}")
    log.info("")
    log.info("  Distribution retreat_rate (transects uniques, clean) :")
    clean_ts = master[master['in_river_zone']==0].drop_duplicates('segment_id')
    rates = clean_ts['retreat_rate_m_yr'].dropna()
    if len(rates):
        log.info(f"    min={rates.min():.2f}  median={rates.median():.2f}"
                 f"  mean={rates.mean():.2f}  max={rates.max():.2f} m/an")
        for label, lo, hi in [
            ('forte érosion  (<-5)',   -999,-5),
            ('modérée        (-5,-2)', -5,-2),
            ('faible         (-2,-0.5)',-2,-0.5),
            ('stable         (±0.5)',  -0.5,0.5),
            ('accrétion      (>0.5)',   0.5,999),
        ]:
            n = ((rates>=lo)&(rates<hi)).sum()
            log.info(f"    {label:<32s}: {n:4d} ({100*n/len(rates):.1f}%)")
    log.info("═"*65)
    log.info("  DONE — dataset prêt pour entraînement ML")