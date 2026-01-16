import osmnx as ox
import requests
import sys

def download_osm_data():
    try:
        place = "Nashik Phata, Pimpri-Chinchwad, Maharashtra, India"
        print(f"Geocoding '{place}'...")
        lat, lon = ox.geocode(place)
        print(f"Coordinates: {lat}, {lon}")

        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:xml][timeout:180];
        (
          node(around:3500,{lat},{lon});
          way(around:3500,{lat},{lon});
          relation(around:3500,{lat},{lon});
        );
        (._;>;);
        out meta;
        """
        
        print("Downloading from Overpass API (this may take a while)...")
        response = requests.get(overpass_url, params={'data': query})
        
        if response.status_code == 200:
            output_path = "data/metro_corridor.osm"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully saved OSM data to {output_path}")
        else:
            print(f"Error downloading data: {response.status_code}")
            print(response.text)
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_osm_data()
