// Google Earth Engine code for simplified 3-PG based NPP and Biomass estimation
// Date: March 29, 2025

// -------------------------
// UI Initialization
// -------------------------
ui.root.clear();
var map = ui.Map();

var legendPanel = ui.Panel({
    style: {
        width: '180px',
        padding: '8px',
        backgroundColor: 'rgba(255, 255, 255, 0.9)'
    }
});
legendPanel.style().set({ position: 'bottom-left', margin: '0px 0px 30px 30px' });

var rightPanel = ui.Panel({ widgets: [ui.Label('rightPanel')], style: { width: '30%' } });
var leftPanel = ui.Panel({ style: { width: '20%' } });
var midPanel = ui.SplitPanel({ firstPanel: map, secondPanel: rightPanel, orientation: 'horizontal' });
var mainPanel = ui.SplitPanel({ firstPanel: leftPanel, secondPanel: ui.Panel(midPanel), orientation: 'horizontal' });
ui.root.add(mainPanel);

// -------------------------
// Load FeatureCollections for Sites
// -------------------------
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/meatha_n");

// -------------------------
// Utility Functions
// -------------------------
function removeLayerByName(name) {
    var layers = map.layers();
    for (var i = layers.length - 1; i >= 0; i--) {
        var layer = layers.get(i);
        if (layer.getName() === name) {
            map.remove(layer);
        }
    }
}

function kelvinToCelsius(image) {
    return image.multiply(0.02).subtract(273.15)
        .copyProperties(image, ['system:time_start']);
}

function getGeom(coord) {
    return ee.Geometry.LineString(coord);
}

function convertPolygonToLine(feature) {
    var polygon = feature.geometry();
    var coords = polygon.coordinates();
    var linearRings = coords.map(getGeom);
    return ee.Feature(ee.Geometry.MultiLineString(linearRings));
}

// -------------------------
// 3-PG Model Parameters
// -------------------------
var params = {
    epsilon: 1.0,   // Light use efficiency (gC/MJ PAR)
    tOpt: 28,       // Optimum temperature for growth (°C)
    tMin: 15,       // Minimum temperature for growth (°C)
    tMax: 40,       // Maximum temperature for growth (°C)
    wMax: 50,       // Maximum soil water content (mm)
    wMin: 15        // Minimum soil water content (mm)
};

// Function to calculate NPP using simplified 3-PG approach. Note: temperature and precipitation are passed as parameters.
function calculateNPP(image, temperature, precipitation) {
    var fpar = image.select('Fpar').multiply(0.01);
    var start = ee.Date(image.get('system:time_start'));
    var end = start.advance(8, 'day');
    var temp = temperature.filterDate(start, end).mean();
    var precip = precipitation.filterDate(start, end).sum();
    var solarRad = ee.Image(20);

    var tempModifier = ee.Image(1)
        .where(temp.lt(params.tMin).or(temp.gt(params.tMax)), 0)
        .where(temp.gte(params.tMin).and(temp.lte(params.tMax)),
            ee.Image(1).multiply(
                ee.Image(-1).multiply(temp.subtract(params.tOpt)).pow(2)
                    .divide(ee.Image(params.tMax - params.tMin).pow(2))
            ).exp());

    var waterModifier = precip
        .clamp(params.wMin, params.wMax)
        .subtract(params.wMin)
        .divide(params.wMax - params.wMin);

    var npp = fpar
        .multiply(solarRad)
        .multiply(params.epsilon)
        .multiply(tempModifier)
        .multiply(waterModifier)
        .rename('NPP');

    return npp.set('system:time_start', image.get('system:time_start'));
}

// -------------------------
// Set Time Range
// -------------------------
// var startDate = '2024-01-01';
// var endDate = '2024-01-31';

// -------------------------
// Main Function to Calculate NPP and Biomass
// -------------------------
function initCalculateNPP(currentRegion) {
    // Clear all layers on the map before recalculating
    map.layers().reset();

    // (Optional) Remove additional layers if necessary
    // removeLayerByName('study area');
    // removeLayerByName('Annual NPP (gC/m²/year)');
    // removeLayerByName('Biomass (g/m²/year)');

    // Load and filter datasets with the selected region
    var modLaiFpar = ee.ImageCollection('NASA/VIIRS/002/VNP15A2H')
        .filterDate(startDate, endDate)
        .filterBounds(currentRegion)
        .select('Fpar');
    print(modLaiFpar);
    var modTemp = ee.ImageCollection("MODIS/061/MOD21A1D")
        .filterDate(startDate, endDate)
        .filterBounds(currentRegion)
        .select('LST_1KM');
    print(modTemp);
    var noaaPrecip = ee.ImageCollection('NOAA/CPC/Precipitation')
        .filterDate(startDate, endDate)
        .filterBounds(currentRegion)
        .select('precipitation');
    print(noaaPrecip);
    // Function to reproject images to currentRegion
    function reProject(image) {
        return image.clip(currentRegion).reproject({ crs: "EPSG:32647", scale: 500 });
    }
    modLaiFpar = modLaiFpar.map(reProject);
    modTemp = modTemp.map(reProject);
    var precipitation = noaaPrecip.map(reProject);

    var temperature = modTemp.map(function (image) {
        return image.multiply(0.02).subtract(273.15)
            .copyProperties(image, ['system:time_start']);
    });

    print(temperature);

    // Calculate NPP for the time period
    var nppCollection = modLaiFpar.map(function (image) {
        return calculateNPP(image, temperature, precipitation);
    });

    var annualNPP = nppCollection.sum().clip(currentRegion);
    var carbonToBiomassFactor = 2.5;
    var biomass = annualNPP.multiply(carbonToBiomassFactor).rename('Biomass');

    var nppViz = { min: 0, max: 20, palette: ['red', 'yellow', 'green'] };
    var biomassViz = { min: 0, max: 30, palette: ['brown', 'yellow', 'green'] };
    var visPolygonBorder = { color: 'red', width: 2 };

    map.centerObject(currentRegion, 11);
    map.addLayer(modLaiFpar, { min: 0, max: 1, palette: ['red', 'yellow', 'green'] }, 'Fpar');
    map.addLayer(temperature, { min: 0, max: 40, palette: ['blue', 'green', 'yellow', 'red'] }, 'Temperature (°C)');
    map.addLayer(noaaPrecip, { min: 0, max: 100, palette: ['blue', 'green', 'yellow', 'red'] }, 'Precipitation (mm)');
    map.addLayer(annualNPP, nppViz, 'Annual NPP (gC/m²/year)');
    map.addLayer(biomass, biomassViz, 'Biomass (g/m²/year)');
    map.addLayer(currentRegion.map(convertPolygonToLine), visPolygonBorder, "study area", true);

    var nppMean = annualNPP.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: currentRegion,
        scale: 500,
        maxPixels: 1e9
    });
    var biomassMean = biomass.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: currentRegion,
        scale: 500,
        maxPixels: 1e9
    });

    print('Mean Annual NPP (gC/m²/year):', nppMean);
    print('Mean Biomass (g/m²/year):', biomassMean);
}


// -------------------------
// UI Elements on Left Panel
// -------------------------
var txtTitle = ui.Label({
    value: 'ติดตามปริมาณเชื้อเพลิง',
    style: { margin: '4px 8px', fontSize: '20px', fontWeight: 'bold' }
});
leftPanel.add(txtTitle);

var txtSubTitle = ui.Label({
    value: 'คำนวณปริมาณเชื้อเพลิงด้วยวิธี 3PGs จากข้อมูลรายวันของดาวเทียม TERRA/AQUA MODIS',
    style: { margin: '4px 8px' }
});
leftPanel.add(txtSubTitle);

var siteSelectTitle = ui.Label({
    value: "เลือกพื้นที่",
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(siteSelectTitle);

var siteItems = [
    { label: "ปากทับ อุตรดิตถ์", value: "ud" },
    { label: "แม่ทาเหนือ เชียงใหม่", value: "mt" }
];
var siteSelectUi = ui.Select({ items: siteItems, value: 'mt', style: { width: '80%' } });
leftPanel.add(siteSelectUi);

var txtDateUi = ui.Label({
    value: 'เลือกวันที่',
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(txtDateUi);

var dateSliderUi = ui.DateSlider({
    start: '2010-01-01',
    value: '2023-11-15',
    style: { width: '80%' }
});
leftPanel.add(dateSliderUi);

var startDate = '2024-01-01';
var endDate = '2024-01-31';

// Bind the slider's value to update startDate and endDate
dateSliderUi.onChange(function (dateRange) {
    // Create an ee.Date object from the slider's end value.
    var edate = ee.Date(dateRange.end());
    // Format the end date and compute the start date as 8 days before.
    endDate = edate.format('YYYY-MM-dd').getInfo();
    startDate = edate.advance(-30, 'day').format('YYYY-MM-dd').getInfo();
    print('New date range:', startDate, endDate);

    // Recalculate with the currently selected site.
    var selectedRegion = (siteSelectUi.getValue() === 'ud') ? ud : mt;
    initCalculateNPP(selectedRegion);
});

var txtBiomassUi = ui.Label({
    value: 'ปริมาณเชื้อเพลิงแต่ละวิธี',
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(txtBiomassUi);

var chkbNpp = ui.Checkbox({ label: 'วิธี 3PGs (Net Primary Productivity: NPP)', value: true });
leftPanel.add(chkbNpp);
var chkbBm = ui.Checkbox({ label: 'วิธี Linear regression', value: true });
leftPanel.add(chkbBm);

var txtOtherDataUi = ui.Label({
    value: 'ชั้นข้อมูลอื่นๆที่ใช้ในการคำนวณปริมาณเชื้อเพลิง',
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(txtOtherDataUi);

var chkbNdvi = ui.Checkbox({ label: 'Normalized Difference Vegetation Index: NDVI', value: true });
leftPanel.add(chkbNdvi);

var chkbNdmi = ui.Checkbox({ label: 'Normalized Difference Moisture Index: NDMI', value: false });
leftPanel.add(chkbNdmi);

var chkbPar = ui.Checkbox({ label: 'Photosynthetically Active Radiation: PAR', value: false });
leftPanel.add(chkbPar);
var chkbApar = ui.Checkbox({ label: 'Absorption Photosynthetically Active Radiation: APAR', value: false });
leftPanel.add(chkbApar);
var chkbGpp = ui.Checkbox({ label: 'Gross Primary Productivity: GPP', value: false });
leftPanel.add(chkbGpp);

// -------------------------
// UI Event: Site Selection
// -------------------------
siteSelectUi.onChange(function (site) {
    var selectedRegion = (site === 'ud') ? ud : mt;
    initCalculateNPP(selectedRegion); // Pass the selected region
});

// Initialize with the default region (mt)
initCalculateNPP(mt);
