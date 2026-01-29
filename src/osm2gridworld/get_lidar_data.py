import math
import pathlib
import requests

import laspy
import numpy as np
import pandas as pd
import wget
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from pyproj import Transformer

def download_usgs_satellite(bbox, filename="usgs_satellite.png", width=500, height=500, scale_factor=1):
    """
    Downloads high-resolution aerial imagery from The National Map (USGS) WMS service.

    Args:
        bbox (dict): A dictionary containing 'west', 'south', 'east', 'north' coordinates.
        filename (str): The path to save the downloaded image.
        width (int): The width of the requested image in pixels.
        height (int): The height of the requested image in pixels.
        scale_factor (int): Multiplier for DPI and map resolution to improve quality.

    Returns:
        None
    """
    wms_url = "https://basemap.nationalmap.gov/arcgis/services/USGSImageryOnly/MapServer/WmsServer"
    
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',
        'REQUEST': 'GetMap',
        'BBOX': f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
        'CRS': 'EPSG:4326',
        'WIDTH': width,
        'HEIGHT': height,
        'LAYERS': '0',
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
    """
    Prints header information, coordinate bounds, and CRS from a LAZ/LAS file.

    Args:
        filename (str): Path to the .laz file.

    Returns:
        None
    """
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
        
        crs = fh.header.parse_crs()
        print(f"\nCoordinate System (CRS):")
        print(crs)

def convert_bbox_to_utm(lat_bbox, from_epsg="EPSG:4326", to_epsg="EPSG:26916"):
    """
    Converts a bounding box from one coordinate reference system to another.

    Args:
        lat_bbox (dict): Dictionary with 'west', 'south', 'east', 'north' keys.
        from_epsg (str): Source CRS code.
        to_epsg (str): Target CRS code.

    Returns:
        dict: Transformed bounding box with keys 'west', 'east', 'south', 'north'.
    """
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    
    west, south = transformer.transform(lat_bbox['west'], lat_bbox['south'])
    east, north = transformer.transform(lat_bbox['east'], lat_bbox['north'])
    
    return {
        'west': west, 'east': east,
        'south': south, 'north': north
    }

def get_bounding_box(lat, lon, size_meters=500):
    """
    Calculates a bounding box centered on a specific coordinate given a side length in meters.

    Args:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        size_meters (int): The total width/height of the box in meters.

    Returns:
        dict: A dictionary containing the 'north', 'south', 'east', and 'west' boundaries.
    """
    radius = size_meters / 2
    
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
    Generates a normalized height map (Digital Surface Model minus Digital Terrain Model) 
    from a LiDAR file for a specific area.
    
    

    Args:
        laz_path (str): Path to the source .laz file.
        target_lat (float): Latitude of the area center.
        target_lon (float): Longitude of the area center.
        side_length (int): The physical width of the area in meters.
        grid_size (int): The pixel resolution of the output grid.

    Returns:
        np.ndarray or None: A 2D array representing object heights, or None if no overlap exists.
    """
    print(f"--- Processing {laz_path} ---")
    
    with laspy.open(laz_path) as fh:
        las_crs = fh.header.parse_crs()
        transformer = Transformer.from_crs("EPSG:4326", las_crs, always_xy=True)
        target_x, target_y = transformer.transform(target_lon, target_lat)
        
        half_side = side_length / 2
        bbox = {
            'west': target_x - half_side,
            'east': target_x + half_side,
            'south': target_y - half_side,
            'north': target_y + half_side
        }
        
        f_mins, f_maxs = fh.header.mins, fh.header.maxs
        
        no_overlap = (bbox['east'] < f_mins[0]) or \
                     (bbox['west'] > f_maxs[0]) or \
                     (bbox['north'] < f_mins[1]) or \
                     (bbox['south'] > f_maxs[1])

        if no_overlap:
            print(" Target Area does not intersect this file. Skipping...")
            return None

        print("Intersection found! Reading data...")
        las = fh.read()

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

    x_bins = np.linspace(bbox['west'], bbox['east'], grid_size + 1)
    y_bins = np.linspace(bbox['south'], bbox['north'], grid_size + 1)

    surface_grid, _, _, _ = binned_statistic_2d(
        x_filt, y_filt, z_filt, 
        statistic='max', 
        bins=[x_bins, y_bins]
    )
    
    ground_grid, _, _, _ = binned_statistic_2d(
        x_filt, y_filt, z_filt, 
        statistic='min', 
        bins=[x_bins, y_bins]
    )

    surface_grid = np.nan_to_num(surface_grid, nan=0.0)
    ground_grid = np.nan_to_num(ground_grid, nan=0.0)
    
    object_height_map = surface_grid - ground_grid
    object_height_map = np.clip(object_height_map, 0, None)
    
    object_height_map = np.rot90(object_height_map)

    return object_height_map.astype(np.float32)

def visualize_height_map(height_map, filename='height_map.png', output_dir=None):
    """
    Visualizes and saves a 2D height map array as an image file using a spectral colormap.

    Args:
        height_map (np.ndarray): 2D array of height values.
        filename (str): Name of the output file.
        output_dir (pathlib.Path or None): Directory to save the file. Defaults to "data/lidar/".

    Returns:
        None
    """
    if output_dir is None:
        output_dir = pathlib.Path("data/lidar/")
    
    valid_heights = height_map[height_map > 0.1] 
    
    if len(valid_heights) == 0:
        print("Map is empty or all zero.")
        return

    vmax_val = np.percentile(valid_heights, 95)
    
    plt.figure(figsize=(10, 8))
    
    plt.imshow(height_map, cmap='nipy_spectral', vmin=0, vmax=vmax_val)
    
    plt.colorbar(label=f'Height (m) [Clipped at {vmax_val:.1f}m]')
    plt.title(f'LiDAR Height Map')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.savefig(output_dir / filename)
    plt.close()

def read_data_sheet(filepath, output_path):
    """
    Reads a CSV data sheet containing download URLs and downloads files to the output path.

    Args:
        filepath (str): Path to the CSV file.
        output_path (pathlib.Path): Directory where files should be downloaded.

    Returns:
        pd.DataFrame: The DataFrame loaded from the CSV.
    """
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