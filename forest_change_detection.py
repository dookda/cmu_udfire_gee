"""
Forest Change Detection using Sentinel-2 Data (2015 vs 2024)
Detects forest cover changes within a specified boundary
"""

import ee
import geemap
import json
from datetime import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# Define the area of interest (AOI)
aoi_geojson = {
    "type": "Feature",
    "properties": {
        "name": "Converted Polygon"
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [100.88964783624543, 17.025106397335236],
                [100.91140353057993, 17.025033245134257],
                [100.9111120771334, 17.000597462154097],
                [100.89209936939791, 16.99974772362222],
                [100.88964783624543, 17.025106397335236]
            ]
        ]
    }
}

# Convert GeoJSON to Earth Engine geometry
aoi = ee.Geometry.Polygon(aoi_geojson['geometry']['coordinates'])

def mask_s2_clouds(image):
    """
    Mask clouds in Sentinel-2 imagery using the QA band
    """
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def get_sentinel2_composite(start_date, end_date, aoi):
    """
    Get cloud-free Sentinel-2 composite for a specific time period
    """
    collection = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_s2_clouds)

    # Calculate median composite
    composite = collection.median().clip(aoi)
    return composite

def calculate_ndvi(image):
    """
    Calculate Normalized Difference Vegetation Index (NDVI)
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi

def classify_forest(ndvi_image, threshold=0.5):
    """
    Classify forest areas based on NDVI threshold
    Forest areas typically have NDVI > 0.5
    """
    forest = ndvi_image.gt(threshold).rename('forest')
    return forest

# Define time periods
# Note: Sentinel-2 is available from June 2015, so we use late 2015
period_2015_start = '2015-06-01'
period_2015_end = '2016-03-31'

period_2024_start = '2024-01-01'
period_2024_end = '2024-12-31'

print("Processing Sentinel-2 data...")
print(f"Area of Interest: {aoi_geojson['properties']['name']}")
print(f"Period 1: {period_2015_start} to {period_2015_end}")
print(f"Period 2: {period_2024_start} to {period_2024_end}")

# Get composites for both periods
print("\nCreating 2015 composite...")
composite_2015 = get_sentinel2_composite(period_2015_start, period_2015_end, aoi)

print("Creating 2024 composite...")
composite_2024 = get_sentinel2_composite(period_2024_start, period_2024_end, aoi)

# Calculate NDVI for both periods
print("\nCalculating NDVI...")
ndvi_2015 = calculate_ndvi(composite_2015)
ndvi_2024 = calculate_ndvi(composite_2024)

# Classify forest areas
print("Classifying forest areas...")
forest_2015 = classify_forest(ndvi_2015, threshold=0.5)
forest_2024 = classify_forest(ndvi_2024, threshold=0.5)

# Detect changes
print("Detecting forest changes...")
# Forest loss: was forest in 2015 (1) but not in 2024 (0)
forest_loss = forest_2015.And(forest_2024.Not()).rename('forest_loss')

# Forest gain: was not forest in 2015 (0) but is forest in 2024 (1)
forest_gain = forest_2015.Not().And(forest_2024).rename('forest_gain')

# No change - remained forest
forest_stable = forest_2015.And(forest_2024).rename('forest_stable')

# No change - remained non-forest
non_forest_stable = forest_2015.Not().And(forest_2024.Not()).rename('non_forest_stable')

# Create a change map
# 0: No change (non-forest), 1: Forest stable, 2: Forest loss, 3: Forest gain
change_map = ee.Image(0) \
    .where(forest_stable, 1) \
    .where(forest_loss, 2) \
    .where(forest_gain, 3) \
    .rename('change_category')

# Calculate statistics
print("\nCalculating statistics...")
area_image = ee.Image.pixelArea()

# Calculate areas in hectares
def calculate_area(image, aoi):
    area = image.multiply(area_image).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,  # Sentinel-2 resolution
        maxPixels=1e9
    )
    return area

forest_2015_area = calculate_area(forest_2015, aoi)
forest_2024_area = calculate_area(forest_2024, aoi)
forest_loss_area = calculate_area(forest_loss, aoi)
forest_gain_area = calculate_area(forest_gain, aoi)

# Get the values and convert to hectares
stats = {
    'forest_2015_ha': forest_2015_area.getInfo()['forest'] / 10000 if 'forest' in forest_2015_area.getInfo() else 0,
    'forest_2024_ha': forest_2024_area.getInfo()['forest'] / 10000 if 'forest' in forest_2024_area.getInfo() else 0,
    'forest_loss_ha': forest_loss_area.getInfo()['forest_loss'] / 10000 if 'forest_loss' in forest_loss_area.getInfo() else 0,
    'forest_gain_ha': forest_gain_area.getInfo()['forest_gain'] / 10000 if 'forest_gain' in forest_gain_area.getInfo() else 0,
}

stats['net_change_ha'] = stats['forest_2024_ha'] - stats['forest_2015_ha']
stats['percent_change'] = (stats['net_change_ha'] / stats['forest_2015_ha'] * 100) if stats['forest_2015_ha'] > 0 else 0

print("\n" + "="*60)
print("FOREST CHANGE DETECTION RESULTS")
print("="*60)
print(f"Forest area in 2015: {stats['forest_2015_ha']:.2f} hectares")
print(f"Forest area in 2024: {stats['forest_2024_ha']:.2f} hectares")
print(f"Forest loss: {stats['forest_loss_ha']:.2f} hectares")
print(f"Forest gain: {stats['forest_gain_ha']:.2f} hectares")
print(f"Net change: {stats['net_change_ha']:.2f} hectares ({stats['percent_change']:.2f}%)")
print("="*60)

# Create an interactive map
print("\nCreating interactive map...")
Map = geemap.Map(center=[17.0125, 100.9], zoom=13)

# Add boundary
Map.addLayer(aoi, {'color': 'yellow'}, 'AOI Boundary')

# Visualization parameters
rgb_vis = {
    'min': 0.0,
    'max': 0.3,
    'bands': ['B4', 'B3', 'B2']
}

ndvi_vis = {
    'min': 0,
    'max': 1,
    'palette': ['red', 'yellow', 'green']
}

change_vis = {
    'min': 0,
    'max': 3,
    'palette': ['white', 'green', 'red', 'blue']
}

# Add layers to map
Map.addLayer(composite_2015, rgb_vis, 'Sentinel-2 2015', False)
Map.addLayer(composite_2024, rgb_vis, 'Sentinel-2 2024', False)
Map.addLayer(ndvi_2015, ndvi_vis, 'NDVI 2015', False)
Map.addLayer(ndvi_2024, ndvi_vis, 'NDVI 2024', False)
Map.addLayer(forest_2015, {'palette': ['white', 'darkgreen']}, 'Forest 2015', False)
Map.addLayer(forest_2024, {'palette': ['white', 'darkgreen']}, 'Forest 2024', False)
Map.addLayer(change_map, change_vis, 'Forest Change Map', True)

# Add legend
legend_dict = {
    'No Change (Non-Forest)': 'white',
    'Forest (Stable)': 'green',
    'Forest Loss': 'red',
    'Forest Gain': 'blue'
}
Map.add_legend(title='Forest Change', legend_dict=legend_dict)

# Display the map
print("\nDisplaying interactive map...")
Map

# Export results
print("\nExporting results...")

# Export change map
export_task = ee.batch.Export.image.toDrive(
    image=change_map.clip(aoi),
    description='forest_change_2015_2024',
    folder='GEE_Exports',
    region=aoi,
    scale=10,
    maxPixels=1e9
)

print("\nTo export the change map to Google Drive, uncomment and run:")
print("# export_task.start()")
print("# print('Export task started. Check your Google Drive folder: GEE_Exports')")

# Save statistics to JSON
with open('forest_change_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("\nStatistics saved to: forest_change_stats.json")

print("\nDone! Use Map to interact with the visualization.")
