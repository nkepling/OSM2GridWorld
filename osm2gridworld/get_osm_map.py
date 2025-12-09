import osmnx as ox
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box, Point
from rasterio.features import rasterize
from rasterio.transform import from_origin

# --- Configuration ---
CRS_UTM = "EPSG:26916"  # UTM Zone 16N (Nashville)
def get_road_width(row, default_map):
    """
    Helper to parse width or fallback to highway type defaults.
    If 'highway' is a list, it selects the type resulting in the MINIMUM width.
    """
    if 'width' in row and pd.notna(row['width']):
        try:
            return float(str(row['width']).split()[0])
        except:
            pass
    
    h_type = row['highway']
    GLOBAL_MIN_WIDTH = 4.0 
    
    if isinstance(h_type, list):
        widths = [default_map.get(t, GLOBAL_MIN_WIDTH) for t in h_type]
        return min(widths)
    
    return default_map.get(h_type, GLOBAL_MIN_WIDTH)

def create_colored_map_with_legend(semantic_grid, label_map, filename="nashville_semantic_legend.png"):
    """
    Creates an RGB image where every ID gets a unique color, overlays a 1m grid,
    and saves it with a legend.
    """
    max_id = semantic_grid.max()
    np.random.seed(42) 
    
    palette = np.random.randint(50, 255, size=(max_id + 1, 3), dtype=np.uint8)
    
    # Hardcode basic layer colors
    if max_id >= 0: palette[0] = [220, 220, 220]   # 0: Background (Light Grey)
    if max_id >= 1: palette[1] = [150, 150, 150]   # 1: Road (Medium Grey)
    if max_id >= 2: palette[2] = [0, 0, 0]         # 2: Building (Black)
    
    # Apply palette
    colored_image = palette[semantic_grid]
    fig, ax = plt.subplots(figsize=(12, 14)) 
    ax.imshow(colored_image, origin='upper', interpolation='nearest')

    n_rows, n_cols = semantic_grid.shape
    
    # Only draw grid if image isn't massive (limit to avoid rendering crashes)
    if n_rows <= 2000:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        
        # Grid settings: Very thin linewidth (0.1) because density is high
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1, alpha=0.3)
        
        # Hide axis labels/spines but keep grid
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    ax.set_title(f"Semantic Grid & Amenity Map\n({len(label_map)} Classes Detected)", fontsize=16, pad=20)

    unique_ids = np.unique(semantic_grid)
    legend_patches = []
    
    for id_val in unique_ids:
        if id_val in label_map:
            label_text = label_map[id_val]
            color = palette[id_val] / 255.0
            patch = mpatches.Patch(color=color, label=label_text)
            legend_patches.append(patch)


    ax.legend(
        handles=legend_patches, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.02),
        ncol=4, 
        fontsize=10,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight') # High DPI to resolve thin grid lines
    plt.close()
    print(f"Visualization saved to '{filename}'")

def generate_complete_map(lat, lon, dist=200):
    """
    Main function to generate Planner Arrays and Semantic Visualization.
    """
    center_point = Point(lon, lat)
    gdf_center = gpd.GeoDataFrame(geometry=[center_point], crs="EPSG:4326").to_crs(CRS_UTM)
    center_x, center_y = gdf_center.geometry[0].x, gdf_center.geometry[0].y

    minx, miny, maxx, maxy = center_x - dist, center_y - dist, center_x + dist, center_y + dist
    bbox_utm = box(minx, miny, maxx, maxy)
    out_shape = (dist * 2, dist * 2)
    transform = from_origin(minx, maxy, 1, 1) # 1 meter resolution

    print(f"Generating {out_shape[1]}x{out_shape[0]} grid centered at {lat}, {lon}...")

    print("Fetching OSM data...")
    tags = {'building': True, 'highway': True, 'width': True, 'amenity': True}
    
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=dist + 50)
    except Exception as e:
        print(f"OSM Download Failed: {e}")
        return

    if gdf.empty:
        print("No OSM data found.")
        return

    # Project and Clip
    gdf = gdf.to_crs(CRS_UTM)
    gdf = gpd.clip(gdf, bbox_utm)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    # --- 3. Process Layers for Planner Arrays ---
    
    # A. ROADS (Buffered)
    roads = gdf[gdf['highway'].notna()].copy()
    if not roads.empty:
        width_defaults = {
            'motorway': 14.0, 'trunk': 12.0, 'primary': 10.0,
            'secondary': 8.0, 'tertiary': 7.0, 'residential': 6.0,
            'service': 4.0, 'footway': 2.0, 'cycleway': 2.0
        }
        # Calculate width & buffer
        roads['est_width'] = roads.apply(lambda row: get_road_width(row, width_defaults), axis=1)
        roads['geometry'] = roads.geometry.buffer(roads['est_width'] / 2)
        
        shapes = ((geom, 1) for geom in roads.geometry)
        road_mask = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    else:
        road_mask = np.zeros(out_shape, dtype=np.uint8)

    # B. BUILDINGS
    buildings = gdf[gdf['building'].notna()]
    if not buildings.empty:
        shapes = ((geom, 1) for geom in buildings.geometry)
        building_mask = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    else:
        building_mask = np.zeros(out_shape, dtype=np.uint8)

    # --- 4. Save Planner Arrays ---
    road_cost_map = np.ones(out_shape, dtype=np.float32)
    road_cost_map[road_mask == 1] = 0.0
    np.save("road_map.npy", road_cost_map)
    print("Saved 'road_map.npy' (0.0=Road, 1.0=Obstacle)")

    obstacle_map = building_mask.astype(np.float32)
    np.save("obstacle_map.npy", obstacle_map)
    print("Saved 'obstacle_map.npy' (1.0=Building, 0.0=Free)")

    # --- 5. Generate Semantic Grid ---
    
    semantic_grid = np.zeros(out_shape, dtype=np.int32)
    semantic_grid[road_mask == 1] = 1
    semantic_grid[building_mask == 1] = 2 

    label_map = {0: "Background", 1: "Road", 2: "Building"}
    next_id = 3

    # C. AMENITIES
    amenities = gdf[gdf['amenity'].notna()].copy()
    if not amenities.empty:
        amenities['geometry'] = amenities.geometry.buffer(4)
        
        unique_amenities = amenities['amenity'].unique()
        print(f"Processing {len(unique_amenities)} amenity types...")
        
        for amenity_type in unique_amenities:
            if pd.isna(amenity_type): continue
            
            # Map ID
            current_id = next_id
            label_map[current_id] = amenity_type.replace("_", " ").title()
            next_id += 1
            
            subset = amenities[amenities['amenity'] == amenity_type]
            shapes = ((geom, current_id) for geom in subset.geometry)
            
            amenity_layer = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.int32)
            semantic_grid[amenity_layer == current_id] = current_id

    # --- 6. Visualize ---
    create_colored_map_with_legend(semantic_grid, label_map)

if __name__ == "__main__":
    # Downtown Nashville: Broadway
    # dist=250 creates a 500m x 500m map
    generate_complete_map(lat=36.1611, lon=-86.7764, dist=250)