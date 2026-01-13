import numpy as np
import matplotlib.pyplot as plt
from osm2gridworld.get_lidar_data import  visualize_height_map, get_bounding_box, read_data_sheet, generate_height_map
from osm2gridworld.get_osm_map import generate_complete_map, visualize_roads_and_obstacles
import pathlib

def get_hunstsville_raw_lidar_data():
    data_sheet_path = pathlib.Path("data/lidar/Huntsville/huntsville_available_data.csv")
    output_path = pathlib.Path("data/lidar/Huntsville/")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    read_data_sheet(data_sheet_path, output_path)


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
    
        np.save(data_dir/"huntsville_height_map.npy", combined_map)

        visualize_height_map(combined_map, filename="huntsville_combined_height.png",output_dir=data_dir)
        print("Saved numpy array to huntsville_height_map.npy")
    else:
        print("\n No overlapping data found in any file.")


def get_huntsville_semantic_gridworld_map(lat,lon):

    print("Generating Semantic Gridworld map for Huntsville, dist is len from center to edge in meters...")

    generate_complete_map(lat, lon, dist=500, filename_prefix="huntsville",output_dir="data/osm/huntsville/")



def main():
    lat, lon =  34.737235, -86.69102 # Huntville, AL

    size_meters = 500
    pixels = 500
    
    print("Getting bounding box in Huntsville centered at lat:", lat, "lon:", lon) 
    bbox = get_bounding_box(lat, lon, size_meters)
    print("Bounding box (lat/lon):", bbox)

    get_hunstsville_raw_lidar_data()

    get_huntsville_height_map(lat,lon)

    get_huntsville_semantic_gridworld_map(lat,lon)

    visualize_roads_and_obstacles(filename_prefix="huntsville", output_dir="data/osm/huntsville/")

    lat_long_map = np.load(pathlib.Path("data/osm/huntsville/huntsville_latlon_map.npy"))

    print("Four Corners of the map (lat, lon): ", lat_long_map[0,0], lat_long_map[0,-1], lat_long_map[-1,0], lat_long_map[-1,-1])




    
    







    




if __name__ == "__main__":
    main()
