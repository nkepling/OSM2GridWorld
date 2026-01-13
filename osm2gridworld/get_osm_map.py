import os
import osmnx as ox
import numpy as np
import geopandas as gpd
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box, Point
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
import ast
from pathlib import Path

# --- Configuration ---
CRS_UTM = "EPSG:26916" 

# Define roads that cars can drive on
CAR_ROAD_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 
    'residential', 'unclassified', 'motorway_link', 'trunk_link', 
    'primary_link', 'secondary_link', 'tertiary_link', 
    'living_street', 'service'
}

# Updated widths for car-centric roads
WIDTH_DEFAULTS = {
    'motorway': 14.0, 'trunk': 12.0, 'primary': 10.0,
    'secondary': 8.0, 'tertiary': 7.0, 'residential': 6.0,
    'unclassified': 6.0, 'living_street': 6.0, 'service': 4.0,
    'motorway_link': 6.0, 'trunk_link': 6.0, 'primary_link': 6.0,
    'secondary_link': 6.0, 'tertiary_link': 6.0
}

def clean_highway_tag(val):
    """
    Parses the highway tag (which might be a string, a list, or a stringified list)
    and returns a single representative car-drivable type if present, else None.
    """
    if pd.isna(val): return None
    
    # Handle lists (e.g., ['primary', 'residential'])
    if isinstance(val, list):
        for v in val:
            if v in CAR_ROAD_TYPES: return v
        return None # No car road types found in list

    # Handle stringified lists (artifact of previous string conversion)
    val_str = str(val)
    if val_str.startswith('[') and val_str.endswith(']'):
        try:

            val_list = ast.literal_eval(val_str)
            if isinstance(val_list, list):
                for v in val_list:
                    if v in CAR_ROAD_TYPES: return v
            return None
        except:
            pass # Fall through if eval fails

    if val_str in CAR_ROAD_TYPES:
        return val_str
        
    return None

def get_road_width(row, default_map):
    # If explicit width exists, use it
    if 'width' in row and pd.notna(row['width']):
        try:
            return float(str(row['width']).split()[0])
        except:
            pass
            
    # Otherwise fallback to type-based width
    h_type = row['clean_highway'] # Use the cleaned tag we generated
    GLOBAL_MIN_WIDTH = 4.0 
    return default_map.get(h_type, GLOBAL_MIN_WIDTH)

def create_colored_map_with_legend(semantic_grid, label_map, filename="nashville_semantic_legend.png"):
    max_id = semantic_grid.max()
    np.random.seed(42) 
    palette = np.random.randint(50, 255, size=(max_id + 1, 3), dtype=np.uint8)
    
    if max_id >= 0: palette[0] = [220, 220, 220]
    if max_id >= 1: palette[1] = [150, 150, 150]
    if max_id >= 2: palette[2] = [0, 0, 0]
    
    colored_image = palette[semantic_grid]

    fig, ax = plt.subplots(figsize=(12, 14)) 
    ax.imshow(colored_image, origin='upper', interpolation='nearest')

    n_rows, n_cols = semantic_grid.shape
    if n_rows <= 2000:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1, alpha=0.3)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values(): spine.set_visible(False)
    
    ax.set_title(f"Semantic Grid & Amenity Map\n({len(label_map)} Classes Detected)", fontsize=16, pad=20)

    unique_ids = np.unique(semantic_grid)
    legend_patches = []
    for id_val in unique_ids:
        if id_val in label_map:
            color = palette[id_val] / 255.0
            patch = mpatches.Patch(color=color, label=label_map[id_val])
            legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to '{filename}'")

def generate_complete_map(lat, lon, dist=200, cache_file=None, filename_prefix="nashville", output_dir=None):
    # Setup output directory
    if output_dir is None:
        output_dir = Path("data/osm")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center_point = Point(lon, lat)
    gdf_center = gpd.GeoDataFrame(geometry=[center_point], crs="EPSG:4326").to_crs(CRS_UTM)
    cx, cy = gdf_center.geometry[0].x, gdf_center.geometry[0].y

    minx, miny, maxx, maxy = cx - dist, cy - dist, cx + dist, cy + dist
    bbox_utm = box(minx, miny, maxx, maxy)
    out_shape = (dist * 2, dist * 2)
    transform = from_origin(minx, maxy, 1, 1)

    print(f"Generating {out_shape[1]}x{out_shape[0]} grid centered at {lat}, {lon}...")

    # if cache_file and Path(cache_file).exists():
    #     print(f"Loading cached data from '{cache_file}'...")
    #     gdf = gpd.read_file(cache_file)
    #     if gdf.crs != CRS_UTM: gdf = gdf.to_crs(CRS_UTM)
    # else:
    print("Fetching from OSM...")
    tags = {'building': True, 'highway': True, 'width': True, 'amenity': True}
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=dist + 50)
    except Exception as e:
        print(f"OSM Download Failed: {e}")
        return None

    if gdf.empty:
        print("No OSM data found.")
        return None

    gdf = gdf.to_crs(CRS_UTM)
    gdf = gpd.clip(gdf, bbox_utm)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].astype(str)

    if cache_file:
        print(f"Saving data to '{cache_file}'...")
        # gdf.to_file(cache_file, driver="GPKG")

    # --- Generate Semantic ID Map & Metadata ---
    gdf['unique_id'] = np.arange(1, len(gdf) + 1)
    
    gdf['clean_highway'] = gdf['highway'].apply(clean_highway_tag)
    
    metadata = {}
    
    def clean_tag_str(val):
        if pd.isna(val): return None
        s = str(val)
        if s.startswith("['"): 
            try: return ast.literal_eval(s)[0] 
            except: return s
        return s

    for idx, row in gdf.iterrows():
        uid = int(row['unique_id'])
        info = {
            "name": clean_tag_str(row.get('name')),
            "highway": row.get('clean_highway'),
            "amenity": clean_tag_str(row.get('amenity')),
            "building": clean_tag_str(row.get('building')),
        }
        metadata[uid] = {k: v for k, v in info.items() if v is not None}

    meta_path = output_dir / f"{filename_prefix}_semantic_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    # Rasterize Semantic ID Grid
    id_grid = np.zeros(out_shape, dtype=np.int32)
    
    roads = gdf[gdf['clean_highway'].notna()].copy()
    
    if not roads.empty:
        roads['est_width'] = roads.apply(lambda row: get_road_width(row, WIDTH_DEFAULTS), axis=1)
        roads['geometry'] = roads.geometry.buffer(roads['est_width'] / 2)
        
        shapes = ((geom, uid) for geom, uid in zip(roads.geometry, roads['unique_id']))
        road_layer = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.int32)
        id_grid = np.where(road_layer > 0, road_layer, id_grid)
    else:
        road_layer = np.zeros(out_shape, dtype=np.int32)

    buildings = gdf[gdf['building'].notna()]
    if not buildings.empty:
        shapes = ((geom, uid) for geom, uid in zip(buildings.geometry, buildings['unique_id']))
        build_layer = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.int32)
        id_grid = np.where(build_layer > 0, build_layer, id_grid)
    else:
        build_layer = np.zeros(out_shape, dtype=np.int32)

    amenities = gdf[gdf['amenity'].notna()].copy()
    if not amenities.empty:
        amenities['geometry'] = amenities.geometry.buffer(4)
        shapes = ((geom, uid) for geom, uid in zip(amenities.geometry, amenities['unique_id']))
        amen_layer = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.int32)
        id_grid = np.where(amen_layer > 0, amen_layer, id_grid)
        
        print("Generating Context Map...")
        def utm_to_pixel(x, y):
            col = int((x - minx) / 1.0)
            row = int((maxy - y) / 1.0)
            return col, row

        amenity_points = []
        amenity_ids = []
        for idx, row in amenities.iterrows():
            geom = row.geometry.centroid
            c, r = utm_to_pixel(geom.x, geom.y)
            if 0 <= c < out_shape[1] and 0 <= r < out_shape[0]:
                amenity_points.append([r, c])
                amenity_ids.append(int(row['unique_id']))
        
        if amenity_points:
            tree = cKDTree(amenity_points)
            rows, cols = np.indices(out_shape)
            all_pixels = np.stack((rows.ravel(), cols.ravel()), axis=-1)
            dists, indices = tree.query(all_pixels, k=1)
            nearest_ids = np.array([amenity_ids[i] for i in indices])
            context_grid = nearest_ids.reshape(out_shape)
            np.save(output_dir / f"{filename_prefix}_context_id_map.npy", context_grid)
    
    np.save(output_dir / f"{filename_prefix}_semantic_id_map.npy", id_grid)
    print(f"Saved files to {output_dir}")

    # --- Create Planner Arrays ---
    road_mask = (road_layer > 0).astype(np.uint8)
    building_mask = (build_layer > 0).astype(np.uint8)

    road_cost_map = np.ones(out_shape, dtype=np.float32)
    road_cost_map[road_mask == 1] = 0.0
    np.save(output_dir / f"{filename_prefix}_road_map.npy", road_cost_map)
    print(f"Saved '{filename_prefix}_road_map.npy'")

    obstacle_map = building_mask.astype(np.float32)
    np.save(output_dir / f"{filename_prefix}_obstacle_map.npy", obstacle_map)
    print(f"Saved '{filename_prefix}_obstacle_map.npy'")

    # --- Create Visual Map ---
    semantic_vis = np.zeros(out_shape, dtype=np.int32)
    semantic_vis[road_mask == 1] = 1
    semantic_vis[building_mask == 1] = 2 

    label_map = {0: "Background", 1: "Road", 2: "Building"}
    next_vis_id = 3

    if not amenities.empty:
        unique_amenities = amenities['amenity'].unique()
        for amenity_type in unique_amenities:
            if pd.isna(amenity_type): continue
            
            a_str = str(amenity_type).replace("['","").replace("']","")
            
            current_id = next_vis_id
            label_map[current_id] = a_str.replace("_", " ").title()
            next_vis_id += 1
            
            subset = amenities[amenities['amenity'] == amenity_type]
            shapes = ((geom, current_id) for geom in subset.geometry)
            amenity_layer = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.int32)
            semantic_vis[amenity_layer == current_id] = current_id

    create_colored_map_with_legend(
        semantic_vis, 
        label_map,
        filename=str(output_dir / f"{filename_prefix}_semantic_legend.png")
    )
    return road_cost_map


def create_colored_map_with_context_and_path(semantic_grid, label_map, road_cost_map, target_path, context_description, filename="nashville_path_visualized.png"):
    max_id = semantic_grid.max()
    np.random.seed(42) 
    
    palette = np.random.randint(50, 255, size=(max_id + 1, 3), dtype=np.uint8)
    
    if max_id >= 0: palette[0] = [220, 220, 220]   # 0: Background
    if max_id >= 1: palette[1] = [150, 150, 150]   # 1: Road
    if max_id >= 2: palette[2] = [30, 30, 30]      # 2: Generic Building
    
    colored_image = palette[semantic_grid]

    fig, ax = plt.subplots(figsize=(12, 12)) 
    
    ax.imshow(road_cost_map, origin='upper', cmap='bone', alpha=0.8, interpolation='nearest')
    ax.imshow(colored_image, origin='upper', interpolation='nearest', alpha=0.7)

    if target_path is not None and len(target_path) > 0:
        path_x = [p[0] for p in target_path]
        path_y = [p[1] for p in target_path]
        
        ax.plot(path_x, path_y, color='black', linewidth=4, alpha=1.0)
        ax.scatter(path_x[0], path_y[0], c='lime', s=150, edgecolors='black', zorder=10)
        ax.scatter(path_x[-1], path_y[-1], c='blue', s=150, marker='*', edgecolors='black', zorder=10)

    # GRID SETTINGS
    n_rows, n_cols = semantic_grid.shape
    if n_rows <= 2000:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1, alpha=0.3)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    ax.set_title(f"Semantic Map & Trajectory", fontsize=18, pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to '{filename}' (No Legend)")



def visualize_roads_and_obstacles(filename_prefix, output_dir):
    output_dir = Path(output_dir)
    
    obstacle_path = output_dir / f"{filename_prefix}_obstacle_map.npy"
    road_path = output_dir / f"{filename_prefix}_road_map.npy"
    
    obstacle_map = np.load(obstacle_path)
    road_map = np.load(road_path)

    COLOR_BACKGROUND = [0.9, 0.9, 0.9]  # Light Grey
    COLOR_ROAD       = [0.6, 0.6, 0.6]  # Darker Grey
    COLOR_BUILDING   = [0.0, 0.0, 0.0]  # Black

    h, w = obstacle_map.shape
    canvas = np.ones((h, w, 3)) * COLOR_BACKGROUND
    canvas[road_map == 1] = COLOR_ROAD
    canvas[obstacle_map == 1] = COLOR_BUILDING
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(canvas, origin='upper', interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, w, 1))
    ax.set_yticks(np.arange(-0.5, h, 1))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Major grid
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
    
    # Minor grid
    ax.grid(which='minor', color='black', linestyle=':', linewidth=0.5, alpha=0.01)

    plt.title(f"{h} x {w} Grid Map: Buildings (Black) & Roads (Light Grey)")
    plt.tight_layout()
    
    save_path = output_dir / f"{filename_prefix}_obstacle_and_road_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    pass
