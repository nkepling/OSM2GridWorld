# OSM2GridWorld
Library to take OpenStreetMaps maps and generate gridworld environments for natural language guided planning problems. Also include functions to download and process LiDAR data for height maps.

## Installation

It's recommended to use uv environment to manage dependencies, not strictly required. 

Clone this repository. 

```bash
git clone https://github.com/nkepling/OSM2GridWorld.git
```
Navigate to the project directory. 

```bash
cd OSM2GridWorld
```

Create a virtual environment. 

```bash
uv venv
```
Activate the virtual environment.

```bash
source .venv/bin/activate
``` 

Install the package in editable mode.

```bash
pip install -e .
```

## Usage

See `main.py` for an example of how to use the library to generate a gridworld environment and visualize height maps from LiDAR data. You can simply run the `main.py` script which will download data and generate maps for Huntsville, AL. Data should be stored in the `data/` directory.

### Getting LiDAR Data

To download the raw LiDAR data for Huntsville, AL can all this code snippet below. It will use wget to download the .laz files and place them in the `data/` directory.

```python
from osm2gridworld.get_lidar_data import generate_calibrated_map, visualize_height_map, read_data_sheet
import pathlib 

data_sheet_path = pathlib.Path("data/Huntsville/huntsville_available_data.csv")

read_data_sheet(data_sheet_path)

```

If you want to get LiDAR data for a different location, you will need to manually download the .laz files from the USGS national map downloader and place them in the `data/lidar` directory. The first step is look at the USGS national map downloader: https://apps.nationalmap.gov/downloader/ to check availability of LiDAR data for your area of interest. Enter your bounding box coordinates and select "Elevation Source Data (3DEP) - Lidar, IfSAR" under data. Then check the "Lidar Point Cloud (LPC)" subcategory. Download the .laz files for your area of interest and place them the `data/lidar` directory in this repository.


### Generating Height Maps

See example code below to generate a calibrated height map for Huntsville, AL using the downloaded .laz files. The area of interest is not completely covered by a single .laz file, so this code merges data from multiple files to create a complete height map.

```python
from osm2gridworld.get_lidar_data import  visualize_height_map, get_bounding_box, read_data_sheet, generate_height_map

def get_huntsville_height_map(lat,lon):
    print("Generating calibrated height map...")


    grid_pixels = 1000  # 1000x1000 output image

    data_dir = pathlib.Path("data/lidar/Huntsville/")

    combined_map = np.zeros((grid_pixels, grid_pixels), dtype=np.float32)
    files_processed = 0

    print(f"Scanning directory: {data_dir} for .laz files...")
    
  
    for path in data_dir.iterdir():
        if path.suffix == ".laz":
            
            # # Generate map for this specific file
            chunk_map = generate_height_map(
                str(path), 
                lat, lon, 
                side_length=1000, 
                grid_size=grid_pixels
            )
            
            if chunk_map is not None:
                combined_map = np.maximum(combined_map, chunk_map)
                files_processed += 1

    
    if files_processed > 0:
        print(f"\n--- Processing Complete ---")
        print(f"Merged data from {files_processed} file(s).")
        
        # Visualize the final result
        visualize_height_map(combined_map, filename="huntsville_combined_height.png")
    
        np.save("huntsville_height_map.npy", combined_map)

        visualize_height_map(combined_map, filename="huntsville_combined_height.png")
        print("Saved numpy array to huntsville_height_map.npy")
    else:
        print("\n No overlapping data found in any file.")

```

### Example coordinates: 

`lat, lon =  34.737235, -86.691023` Hunstville, AL

`lat, lon =  34.737235, -86.691023` Nashville, TN



