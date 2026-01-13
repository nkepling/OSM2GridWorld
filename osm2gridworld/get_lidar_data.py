import numpy as np 
import laspy
import math
import pandas as pd
import wget
from scipy.stats import binned_statistic_2d
from pyproj import Transformer
import requests
import pathlib


def download_usgs_satellite(bbox, filename="usgs_satellite.png", width=500, height=500, scale_factor=1):
    """
    Downloads high-res aerial imagery from The National Map (USGS) WMS.
    bbox: dict with 'west', 'south', 'east', 'north' (Lat/Lon)
    width/height: pixel dimensions of the output image
    """
    # The National Map Orthoimagery Endpoint
    wms_url = "https://basemap.nationalmap.gov/arcgis/services/USGSImageryOnly/MapServer/WmsServer"
    
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',
        'REQUEST': 'GetMap',
        'BBOX': f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
        'CRS': 'EPSG:4326', # Requesting in Lat/Lon
        'WIDTH': width,
        'HEIGHT': height,
        'LAYERS': '0', # Layer 0 is the imagery
        'STYLES': '',
        'FORMAT': 'image/png',
        'DPI': str(96 * scale_factor),
        'MAP_RESOLUTION': str(96 * scale_factor),
        'FORMAT_OPTIONS': f'dpi:{96 * scale_factor}'
    }
    
    print(f"Requesting USGS Aerial Imagery...")
    response = requests.get(wms_url, params=params)
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Saved aerial image to {filename}")
    else:
        print(f"Failed to download: {response.status_code}")
        print(response.text)

def inspect_laz(filename):
    print(f"--- Inspecting {filename} ---")
    with laspy.open(filename) as fh:
        mins = fh.header.mins
        maxs = fh.header.maxs
        scale = fh.header.scales
        offset = fh.header.offsets
        
        print(f"File Coordinate Bounds:")
        print(f"  X: {mins[0]:.2f} to {maxs[0]:.2f}")
        print(f"  Y: {mins[1]:.2f} to {maxs[1]:.2f}")
        print(f"  Z: {mins[2]:.2f} to {maxs[2]:.2f}")
        
        # Check the CRS (Coordinate Reference System)
        crs = fh.header.parse_crs()
        print(f"\nCoordinate System (CRS):")
        print(crs)


def convert_bbox_to_utm(lat_bbox, from_epsg="EPSG:4326", to_epsg="EPSG:26916"):
    # EPSG:26916 is UTM Zone 16N (Nashville).
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    
    west, south = transformer.transform(lat_bbox['west'], lat_bbox['south'])
    east, north = transformer.transform(lat_bbox['east'], lat_bbox['north'])
    
    return {
        'west': west, 'east': east,
        'south': south, 'north': north
    }

def get_bounding_box(lat, lon, size_meters=500):
    radius = size_meters / 2
    
    # Earth's circumference is ~40,000 km, so 1 degree lat is ~111,111 meters
    lat_offset = radius / 111111

    lon_offset = radius / (111111 * math.cos(math.radians(lat)))
    
    return {
        "north": lat + lat_offset,
        "south": lat - lat_offset,
        "east":  lon + lon_offset,
        "west":  lon - lon_offset
    }


def generate_height_map(laz_path, target_lat, target_lon, side_length=500, grid_size=500):
    """
    Generates a height map (2D numpy array) from a .laz LiDAR file for a square area
    centered at (target_lat, target_lon) with side length in meters.

    laz_path: Path to the .laz file 
    target_lat, target_lon: Center of the desired area in lat/lon
    side_length: Length of one side of the square area in meters
    grid_size: Number of pixels along one side of the output height map

    """
    print(f"--- Processing {laz_path} ---")
    
    with laspy.open(laz_path) as fh:
        # READ CRS & TRANSFORM TARGET
        las_crs = fh.header.parse_crs()
        transformer = Transformer.from_crs("EPSG:4326", las_crs, always_xy=True)
        target_x, target_y = transformer.transform(target_lon, target_lat)
        
        # DEFINE DESIRED BOUNDING BOX FIRST
        half_side = side_length / 2
        bbox = {
            'west': target_x - half_side,
            'east': target_x + half_side,
            'south': target_y - half_side,
            'north': target_y + half_side
        }
        
        # CHECK FOR INTERSECTION (Overlap)
        f_mins, f_maxs = fh.header.mins, fh.header.maxs
        
        # Intersection Logic: 
        no_overlap = (bbox['east'] < f_mins[0]) or \
                     (bbox['west'] > f_maxs[0]) or \
                     (bbox['north'] < f_mins[1]) or \
                     (bbox['south'] > f_maxs[1])

        if no_overlap:
            print(" Target Area does not intersect this file. Skipping...")
            return None

        print("Intersection found! Reading data...")
        # Only now do we pay the cost of reading the file
        las = fh.read()

    # FILTER POINTS
    mask = (
        (las.x >= bbox['west']) & (las.x < bbox['east']) & 
        (las.y >= bbox['south']) & (las.y < bbox['north'])
    )
    
    x_filt = las.x[mask]
    y_filt = las.y[mask]
    z_filt = las.z[mask]

    if len(x_filt) == 0:
        print("Warning: Bounding box overlaps, but no points found (sparse data?).")
        return None

    # CREATE GRID
    x_bins = np.linspace(bbox['west'], bbox['east'], grid_size + 1)
    y_bins = np.linspace(bbox['south'], bbox['north'], grid_size + 1)

    # Max height (Surface)
    surface_grid, _, _, _ = binned_statistic_2d(
        x_filt, y_filt, z_filt, 
        statistic='max', 
        bins=[x_bins, y_bins]
    )
    
    # Min height (Ground) - Optional for normalization
    ground_grid, _, _, _ = binned_statistic_2d(
        x_filt, y_filt, z_filt, 
        statistic='min', 
        bins=[x_bins, y_bins]
    )

    # 6. CALCULATE HEIGHT (Normalize)
    # Fill empty pixels (off the edge of the file) with 0
    surface_grid = np.nan_to_num(surface_grid, nan=0.0)
    ground_grid = np.nan_to_num(ground_grid, nan=0.0)
    
    # Calculate Obstacle Height (Object - Ground)
    # If you want raw elevation (ASL), just use 'surface_grid'
    object_height_map = surface_grid - ground_grid
    object_height_map = np.clip(object_height_map, 0, None)  # No negative heights
    # object_height_map = surface_grid
    # object_height_map = surface_grid
    
    # Rotate to face North-Up (Standard map orientation)
    object_height_map = np.rot90(object_height_map)

    return object_height_map.astype(np.float32)
        


def visualize_height_map(height_map, filename='height_map.png', output_dir=None):
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = pathlib.Path("data/lidar/")
    
    # 1. Filter out absolute zeros to calculate stats on actual objects
    valid_heights = height_map[height_map > 0.1] 
    
    if len(valid_heights) == 0:
        print("Map is empty or all zero.")
        return

    #  Calculate the 95th percentile (ignores top 5% outliers)
    vmax_val = np.percentile(valid_heights, 95)
    
    plt.figure(figsize=(10, 8))
    
    # 3. Apply the clip using vmin and vmax
    # 'jet' or 'nipy_spectral' often show low-value contrast better than 'terrain'
    plt.imshow(height_map, cmap='nipy_spectral', vmin=0, vmax=vmax_val)
    
    plt.colorbar(label=f'Height (m) [Clipped at {vmax_val:.1f}m]')
    plt.title(f'LiDAR Height Map')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.savefig(output_dir / filename)
    plt.close()

def read_data_sheet(filepath, output_path):
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(filepath, header=None)
    print(df.head())

    urls = df.iloc[:, 14]

    for url in urls:
        filename = url.split('/')[-1]
        dest_path = output_path / filename
        
        if dest_path.exists():
            print(f"Skipping {filename}, already exists.")
            continue
            
        print(f"Downloading {url} to {dest_path}...")
        wget.download(url, out=str(dest_path))
        print(" Done.")
    return df

if __name__ == "__main__":
    pass