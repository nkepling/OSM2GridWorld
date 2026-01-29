from osm2gridworld.get_osm_map import generate_complete_map




def main():

    # Downtown Nashville on Broadway
    lat=36.160044
    lon=-86.779407

    FILENAME_PREFIX = "nashville"
    OUTPUT_DIR = "data/osm_data/Nashville"

    #NOTE: dist is the distnace from center of the map to its edge. 
    generate_complete_map(lat, lon, dist=500, cache_file=None, filename_prefix=FILENAME_PREFIX, output_dir=OUTPUT_DIR)





if __name__ == "__main__":
    main()