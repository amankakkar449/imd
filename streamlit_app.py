# streamlit_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from matplotlib.colors import Normalize

st.set_page_config(layout="wide", page_title="Tmax Kriging Dashboard")

# ----------------------
# === CONFIG / PATHS ===
# ----------------------
DATA_FOLDER = r"D:\IMD\dashboard"          # folder all year CSVs files
SHAPEFILE = r"D:\IMD\dashboard\DISTRICT_BOUNDARY.shp" # shapefile with STATE_UT column

# CSV column names (from your actual CSV)
COL_DATE = "date"     # melted date column
COL_LON = "X"         # longitude column (capital X in your file)
COL_LAT = "Y"         # latitude column (capital Y in your file)
COL_TMAX = "tmax"     # Tmax value after melt

# Grid resolution for interpolation (smaller = finer = slower)
GRID_RES = 0.1  # ~0.1° (~11km)

# Colormap
CMAP = "hot"

# ----------------------
# === Helper Functions ===
# ----------------------
@st.cache_data(show_spinner=False)
def load_csv(folder, start_year, end_year):
    dfs = []
    for yr in range(start_year, end_year + 1):
        path = f"{folder}\\{yr}_tmax.csv"
        try:
            df = pd.read_csv(path)

            # unpivot daily columns (other than X, Y)
            df_melted = df.melt(id_vars=["X", "Y"], var_name="date", value_name="tmax")

            # convert to datetime
            df_melted["date"] = pd.to_datetime(df_melted["date"], errors="coerce")

            # clean missing values (-9999 → NaN)
            df_melted["tmax"] = df_melted["tmax"].replace(-9999.0, np.nan)

            # drop rows with no valid tmax
            df_melted = df_melted.dropna(subset=["tmax", "date"])
            df_melted["year"] = df_melted["date"].dt.year

            dfs.append(df_melted)
        except FileNotFoundError:
            st.warning(f"No CSV found for year {yr}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["X", "Y", "date", "tmax", "year"])



@st.cache_data(show_spinner=False)
def load_shapefile(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def aggregate_monthly(df):
    df = df.copy()
    df["year"] = df[COL_DATE].dt.year
    df["month"] = df[COL_DATE].dt.month
    df["lon_round"] = df[COL_LON].round(5)
    df["lat_round"] = df[COL_LAT].round(5)
    agg = df.groupby(
        ["year", "month", "lon_round", "lat_round"], as_index=False
    )[COL_TMAX].mean()
    agg = agg.rename(
        columns={"lon_round": "lon", "lat_round": "lat", COL_TMAX: "tmax_mean"}
    )
    return agg

def create_grid(bounds, res=GRID_RES):
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx + res, res)
    ys = np.arange(miny, maxy + res, res)
    return xs, ys[::-1]

def kriging_to_grid(x, y, values, gridx, gridy, variogram_model="linear"):
    OK = OrdinaryKriging(
        x, y, values, variogram_model=variogram_model, enable_plotting=False, verbose=False
    )
    z, ss = OK.execute("grid", gridx, gridy)
    return np.array(z)

def raster_clip_and_mask(zgrid, gridx, gridy, mask_gdf):
    left, right = gridx.min(), gridx.max()
    top, bottom = gridy.max(), gridy.min()
    ncols, nrows = len(gridx), len(gridy)
    transform = rasterio.transform.from_bounds(left, bottom, right, top, ncols, nrows)
    profile = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    memfile = rasterio.io.MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(zgrid.astype("float32"), 1)
        geoms = [mapping(g) for g in mask_gdf.geometry]
        out_image, out_transform = mask(dataset, geoms, crop=True, nodata=np.nan)
    return out_image[0], out_transform

def plot_grid_array(arr, transform, vmin=None, vmax=None, title=None, cmap=CMAP):
    fig, ax = plt.subplots(figsize=(3, 3))
    if np.all(np.isnan(arr)):
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        left, top = transform[2], transform[5]
        resx, resy = transform[0], transform[4]
        right = left + arr.shape[1] * resx
        bottom = top + arr.shape[0] * resy
        extent = (left, right, bottom, top)
        ax.imshow(arr, origin="upper", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return fig, ax

# ----------------------
# === UI & Workflow ===
# ----------------------
st.sidebar.header("Filters")
gdf_shp = load_shapefile(SHAPEFILE)

# State dropdown
all_states = sorted(gdf_shp["STATE_UT"].dropna().astype(str).unique().tolist())
selected_state = st.sidebar.selectbox("Select State", all_states)

# Year range and month
year_start = st.sidebar.selectbox("From Year", list(range(1981, 1991)), index=0)
year_end = st.sidebar.selectbox("To Year", list(range(1981, 1991)), index=9)
month_name = st.sidebar.selectbox(
    "Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
)
month_num = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index(month_name) + 1

st.title(f"🌡️ Tmax Kriging Dashboard ({year_start}–{year_end}) — {month_name}")

# Load data
df = load_csv(DATA_FOLDER, year_start, year_end)
agg = aggregate_monthly(df)

# Filter data
years = list(range(year_start, year_end + 1))
agg_sel = agg[(agg["year"].between(year_start, year_end)) & (agg["month"] == month_num)]

# State mask
mask_gdf = gdf_shp[gdf_shp["STATE_UT"] == selected_state]
mask_bounds = mask_gdf.total_bounds
pad = 0.3
bounds = (
    mask_bounds[0] - pad,
    mask_bounds[1] - pad,
    mask_bounds[2] + pad,
    mask_bounds[3] + pad,
)
gridx, gridy = create_grid(bounds, res=GRID_RES)

# Global vmin/vmax
yearly_means, vmin, vmax = [], None, None
for yr in years:
    row = agg[(agg["year"] == yr) & (agg["month"] == month_num)]
    if row.empty:
        yearly_means.append(np.nan)
        continue
    vals = row["tmax_mean"].values
    yearly_means.append(np.nanmean(vals))
    vmin = np.nanmin(vals) if vmin is None else min(vmin, np.nanmin(vals))
    vmax = np.nanmax(vals) if vmax is None else max(vmax, np.nanmax(vals))
if vmin is None or vmax is None:
    vmin, vmax = 20, 45

# === Maps grid ===
st.markdown("### Maps")
cols = st.columns(5)
for i, yr in enumerate(years):
    with cols[i % 5]:
        data_points = agg[(agg["year"] == yr) & (agg["month"] == month_num)]
        if data_points.empty:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.text(0.5, 0.5, "No station data", ha="center", va="center")
            ax.axis("off")
            st.pyplot(fig)
            continue
        x, y, vals = data_points["lon"].values, data_points["lat"].values, data_points["tmax_mean"].values
        try:
            zgrid = kriging_to_grid(x, y, vals, gridx, gridy, variogram_model="linear")
            masked_arr, out_transform = raster_clip_and_mask(zgrid, gridx, gridy, mask_gdf)
            fig, ax = plot_grid_array(masked_arr, out_transform, vmin=vmin, vmax=vmax,
                          title=f"{month_name} {yr}", cmap=CMAP)

# state boundary overlay
            mask_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

# zoom to state extent
            bbox = mask_gdf.total_bounds
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])

            st.pyplot(fig)

        except Exception as e:
            st.write("Interpolation failed:", e)

# === Common colorbar ===
st.markdown("### Legend")
fig, ax = plt.subplots(figsize=(8, 1))
norm = Normalize(vmin=vmin, vmax=vmax)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=ax, orientation="horizontal")
cb.set_label("Tmax (°C)")
st.pyplot(fig)

# === Time series ===
st.markdown("### Time series of monthly mean (selected month)")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(years, yearly_means, marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Tmax (°C)")
ax.set_title(f"{month_name} mean Tmax ({year_start}-{year_end})")
st.pyplot(fig)

st.info("⚠️ Kriging is computed on-the-fly. For faster performance, precompute rasters and load them.")
