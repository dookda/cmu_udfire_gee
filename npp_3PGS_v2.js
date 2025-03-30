// Google Earth Engine code for simplified 3-PG based NPP and Biomass estimation
// Date: March 29, 2025

// Define region of interest (e.g., a simple geometry, replace with your area)
var region = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");

// Set time range
var startDate = '2024-01-01';
var endDate = '2024-01-31';

// Load MODIS datasets with correct band names
var modLaiFpar = ee.ImageCollection('MODIS/061/MOD15A2H')
    .filterDate(startDate, endDate)
    .filterBounds(region)
    .select('Fpar_500m'); // Corrected band name: Fpar_500m

var modTemp = ee.ImageCollection("MODIS/061/MOD21A1D")
    .filterDate(startDate, endDate)
    .filterBounds(region)
    .select('LST_1KM'); // Daytime Land Surface Temperature

var precip = ee.ImageCollection('NOAA/CPC/Precipitation')
    .filterDate(startDate, endDate)
    .filterBounds(region)
    .select('precipitation'); // Precipitation in mm/daily

// Function to convert LST to Celsius (MODIS LST is in Kelvin * 0.02)
var kelvinToCelsius = function (image) {
    return image.multiply(0.02).subtract(273.15)
        .copyProperties(image, ['system:time_start']);
};

function reProject(image) {
    return image.clip(region).reproject({ crs: "EPSG:32647", scale: 500 });
}

var modisLaiFpar = modLaiFpar.map(reProject);
var modisTemp = modTemp.map(reProject);
var precipitation = precip.map(reProject);

// Process temperature data
var temperature = modisTemp.map(kelvinToCelsius);

// Simplified 3-PG Parameters (example values, adjust based on species/region)
var params = {
    epsilon: 1.0,       // Light use efficiency (gC/MJ PAR), typical range 0.5-1.5
    tOpt: 25,           // Optimum temperature for growth (°C)
    tMin: 15,            // Minimum temperature for growth (°C)
    tMax: 40,           // Maximum temperature for growth (°C)
    wMax: 50,           // Maximum soil water content (mm)
    wMin: 15             // Minimum soil water content (mm)
};

// Function to calculate NPP using simplified 3-PG approach
var calculateNPP = function (image) {
    var fpar = image.select('Fpar_500m').multiply(0.01); // Corrected band name, MODIS FPAR is scaled 0-100

    var start = ee.Date(image.get('system:time_start'));
    var end = start.advance(8, 'day');
    var temp = temperature.filterDate(start, end).mean();
    var precip = precipitation.filterDate(start, end).sum();
    // var temp = temperature.filterDate(image.get('system:time_start')).mean();
    // var precip = precipitation.filterDate(image.get('system:time_start')).sum(); // Total precip for 8-day period

    // Solar radiation approximation (MJ/m²/day, constant for simplicity)
    var solarRad = ee.Image(20); // Average value, replace with actual data if available

    // Temperature modifier (simplified Gaussian function)
    var tempModifier = ee.Image(1)
        .where(temp.lt(params.tMin).or(temp.gt(params.tMax)), 0)
        .where(temp.gte(params.tMin).and(temp.lte(params.tMax)),
            ee.Image(1).multiply(
                ee.Image(-1).multiply(temp.subtract(params.tOpt)).pow(2)
                    .divide(ee.Image(params.tMax - params.tMin).pow(2))
            ).exp());

    // Water modifier (simple linear function based on precipitation)
    var waterModifier = precip
        .clamp(params.wMin, params.wMax)
        .subtract(params.wMin)
        .divide(params.wMax - params.wMin);

    // NPP calculation (gC/m²/8-day period)
    var npp = fpar
        .multiply(solarRad)          // PAR absorbed
        .multiply(params.epsilon)    // Light use efficiency
        .multiply(tempModifier)      // Temperature effect
        .multiply(waterModifier)     // Water availability effect
        .rename('NPP');

    return npp.set('system:time_start', image.get('system:time_start'));
};

// Calculate NPP for the time period
var nppCollection = modisLaiFpar.map(calculateNPP);

// Annual NPP (sum of 8-day periods)
var annualNPP = nppCollection.sum().clip(region);

// Biomass calculation (convert NPP to biomass)
// Assuming 50% carbon content and adjusting for molecular weight (C to CH2O)
var carbonToBiomassFactor = 2.5; // 30/12 = 2.5 (CH2O/C ratio)
var biomass = annualNPP.multiply(carbonToBiomassFactor).rename('Biomass');

// Visualization parameters
var nppViz = { min: 0, max: 20, palette: ['red', 'yellow', 'green'] };
var biomassViz = { min: 0, max: 30, palette: ['brown', 'yellow', 'green'] };

// Add layers to map
Map.centerObject(region, 11);
Map.addLayer(annualNPP, nppViz, 'Annual NPP (gC/m²/year)');
Map.addLayer(biomass, biomassViz, 'Biomass (g/m²/year)');

// Calculate mean values for the region
var nppMean = annualNPP.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: 500,
    maxPixels: 1e9
});

var biomassMean = biomass.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: 500,
    maxPixels: 1e9
});

// Print results
print('Mean Annual NPP (gC/m²/year):', nppMean);
print('Mean Biomass (g/m²/year):', biomassMean);

// Optional: Export biomass image
// Export.image.toDrive({
//   image: biomass,
//   description: '3PG_Biomass_2020',
//   scale: 500,
//   region: region,
//   maxPixels: 1e9
// });