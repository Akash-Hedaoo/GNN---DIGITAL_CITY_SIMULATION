"""
Prepare GeoJSON from SUMO Network
=================================
One-time preprocessing to convert SUMO network edges to GeoJSON
with proper WGS84 (lat/lon) coordinates for map overlay.

Author: Traffic Digital Twin Project
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import pyproj

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETWORK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "networks", "pune.net.xml")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "maps", "pune_edges.geojson")

# =============================================================================
# Coordinate Transformation
# =============================================================================

def get_projection_from_network(root):
    """Extract projection info from network and create transformer."""
    location = root.find('location')
    
    if location is None:
        raise ValueError("No location element found in network file")
    
    # Get network offset
    net_offset = location.get('netOffset', '0,0')
    offset_x, offset_y = map(float, net_offset.split(','))
    
    # Get original boundary (in WGS84)
    orig_boundary = location.get('origBoundary', '0,0,0,0')
    orig_parts = list(map(float, orig_boundary.split(',')))
    orig_min_lon, orig_min_lat = orig_parts[0], orig_parts[1]
    
    # Get converted boundary (in local projection)
    conv_boundary = location.get('convBoundary', '0,0,0,0')
    conv_parts = list(map(float, conv_boundary.split(',')))
    conv_min_x, conv_min_y = conv_parts[0], conv_parts[1]
    
    # Get projection string if available
    proj_string = location.get('projParameter', None)
    
    if proj_string:
        # Use the projection from network file
        try:
            local_crs = pyproj.CRS.from_proj4(proj_string)
            transformer = pyproj.Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
            return transformer, offset_x, offset_y
        except:
            pass
    
    # Fallback: UTM zone calculation based on center longitude
    center_lon = (orig_parts[0] + orig_parts[2]) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if orig_parts[1] > 0 else 'south'
    
    # Create transformer from UTM to WGS84
    utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
    transformer = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    
    return transformer, offset_x, offset_y


def transform_coordinates(x, y, transformer, offset_x, offset_y):
    """Transform local SUMO coordinates to WGS84 lat/lon."""
    # Apply offset (SUMO uses negative offset, so we add it back)
    abs_x = x - offset_x
    abs_y = y - offset_y
    
    # Transform to WGS84
    lon, lat = transformer.transform(abs_x, abs_y)
    
    return [lon, lat]

# =============================================================================
# GeoJSON Generation
# =============================================================================

def parse_shape_to_coords(shape_str, transformer, offset_x, offset_y):
    """Parse SUMO shape string to list of [lon, lat] coordinates."""
    coords = []
    
    for point in shape_str.split():
        try:
            parts = point.split(',')
            x, y = float(parts[0]), float(parts[1])
            lon_lat = transform_coordinates(x, y, transformer, offset_x, offset_y)
            coords.append(lon_lat)
        except (ValueError, IndexError):
            continue
    
    return coords


def create_geojson(network_path, output_path):
    """Convert SUMO network to GeoJSON."""
    
    print(f"[INFO] Loading network from: {network_path}")
    
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found: {network_path}")
    
    tree = ET.parse(network_path)
    root = tree.getroot()
    
    # Get coordinate transformer
    print("[INFO] Setting up coordinate transformation...")
    transformer, offset_x, offset_y = get_projection_from_network(root)
    
    # Create GeoJSON features
    features = []
    edge_count = 0
    skipped_count = 0
    
    print("[INFO] Processing edges...")
    
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        
        # Skip internal edges (start with ':')
        if edge_id.startswith(':'):
            skipped_count += 1
            continue
        
        # Get shape from edge or first lane
        shape = edge.get('shape')
        
        if not shape:
            # Try to get from first lane
            lane = edge.find('lane')
            if lane is not None:
                shape = lane.get('shape')
        
        if not shape:
            skipped_count += 1
            continue
        
        # Parse shape to coordinates
        coords = parse_shape_to_coords(shape, transformer, offset_x, offset_y)
        
        if len(coords) < 2:
            skipped_count += 1
            continue
        
        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "edge_id": edge_id,
                "from": edge.get('from', ''),
                "to": edge.get('to', ''),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        }
        
        features.append(feature)
        edge_count += 1
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    
    print(f"[INFO] Processed {edge_count} edges")
    print(f"[INFO] Skipped {skipped_count} edges (internal or no shape)")
    print(f"[INFO] GeoJSON saved to: {output_path}")
    
    # Calculate map center
    if features:
        all_lons = []
        all_lats = []
        for feature in features:
            for coord in feature['geometry']['coordinates']:
                all_lons.append(coord[0])
                all_lats.append(coord[1])
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        print(f"[INFO] Map center: ({center_lat:.6f}, {center_lon:.6f})")
        
        # Save center coordinates
        center_path = os.path.join(os.path.dirname(output_path), "map_center.json")
        with open(center_path, 'w') as f:
            json.dump({"lat": center_lat, "lon": center_lon}, f)
        print(f"[INFO] Map center saved to: {center_path}")
    
    return geojson


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUMO Network to GeoJSON Converter")
    print("="*60 + "\n")
    
    try:
        geojson = create_geojson(NETWORK_PATH, OUTPUT_PATH)
        print("\n[SUCCESS] GeoJSON generation complete!")
        print(f"[INFO] Total features: {len(geojson['features'])}")
    except Exception as e:
        print(f"\n[ERROR] Failed to create GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
