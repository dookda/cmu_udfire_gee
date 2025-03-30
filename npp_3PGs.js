// Data collections
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/meatha_n");

// Initialize UI
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

//=============================================================================
// Helper functions (most remain unchanged)
//=============================================================================
function getDataset(dateEnd, dateComposite) {

    var d = ee.Date(dateEnd);
    var dateStart = d.advance(dateComposite, 'day').format('yyyy-MM-dd');

    print(dateStart, dateEnd);
    var dStart = ee.Date('2023-11-02')
    var dEnd = ee.Date('2023-11-16')

    var mdData = ee.ImageCollection('MODIS/061/MOD09GA')
        .filter(ee.Filter.date(dStart, dEnd));
    print(mdData);

    var mcdData = ee.ImageCollection('MODIS/062/MCD18A1')
        // .filter(ee.Filter.date('2000-01-01', '2001-01-31'))
        .filter(ee.Filter.date(dateStart, dateEnd))
        .select('GMT_0600_DSR');
    print(mcdData);

    var firms = ee.ImageCollection("FIRMS")
        .filter(ee.Filter.date(dateStart, dateEnd))
        .select('T21');

    return { md: mdData, mcd: mcdData, firms: firms };
}

function calIndex(image) {
    var ndvi = image.normalizedDifference({ bandNames: ['sur_refl_b02', 'sur_refl_b01'] }).rename('NDVI');
    var ndmi = image.normalizedDifference({ bandNames: ['sur_refl_b02', 'sur_refl_b06'] }).rename('NDMI');
    var combined = ndvi.addBands(ndmi);
    return combined.copyProperties(image, ['system:time_start']);
}

function reProject(image) {
    return image.clip(site).reproject({ crs: "EPSG:32647", scale: 500 });
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

function calDiffNdvi(image) {
    var imagePair = ee.List(image);
    var currentImage = ee.Image(imagePair.get(0));
    var previousImage = ee.Image(imagePair.get(1));
    var ndviDiff = currentImage.subtract(previousImage).rename('NDVIdiff');
    return currentImage.addBands(ndviDiff).select('NDVIdiff');
}

function calBM(image) {
    // Biomass equation: Vol = (7.25923*(NDVI^3)) - (13.419*(NDVI^2)) + (6.4542*(NDVI)) - 0.2305
    var biomass = image.expression(
        '7.25923 * pow(NDVI, 3) - 13.419 * pow(NDVI, 2) + 6.4542 * NDVI - 0.2305',
        { 'NDVI': image.select('NDVI') }
    ).rename('BIOMASS');
    return image.addBands(biomass);
}

function calFpar(image) {
    var fpar = image.select('NDVI').multiply(1.5).subtract(-0.1).rename('FPAR');
    return image.addBands(fpar);
}

function calPar(image) {
    var dsr24hr = image.select('GMT_0600_DSR').multiply(18000).divide(1000000);
    var par = dsr24hr.multiply(0.45).rename('PAR');
    return image.addBands(par.copyProperties(image, ['system:time_start']));
}

function calApar(image) {
    var apar = image.select('FPAR').multiply(image.select('PAR')).rename('APAR');
    var gpp = apar.multiply(1.8).rename('GPP');
    var npp = gpp.multiply(0.45).rename('NPP');
    return image.addBands(apar).addBands(gpp).addBands(npp);
}

function mergeBands(feature) {
    var image1 = ee.Image(feature.get('primary'));
    var image2 = ee.Image(feature.get('secondary'));
    return image1.addBands(image2);
}

function showBmChart(mdCollection, bandArr, site) {
    var chartBmUi = ui.Chart.image.series({
        imageCollection: mdCollection.select(bandArr),
        region: site,
        reducer: ee.Reducer.mean(),
        scale: 500,
        xProperty: 'system:time_start'
    });
    chartBmUi.setOptions({
        hAxis: { title: 'วันที่' },
        vAxis: { title: 'เชื้อเพลิง (Kg/m^2)' },
        curveType: 'function'
    });
    rightPanel.add(chartBmUi);
}

function showChart(mdCollection, bandArr, site) {
    var chartUi = ui.Chart.image.series({
        imageCollection: mdCollection.select(bandArr),
        region: site,
        reducer: ee.Reducer.mean(),
        scale: 500,
        xProperty: 'system:time_start'
    });
    chartUi.setOptions({
        hAxis: { title: 'วันที่' },
        vAxis: { title: 'index' },
        curveType: 'function'
    });
    rightPanel.add(chartUi);
}

function makeColorBarParams(palette) {
    var nSteps = 10;
    return {
        bbox: [0, 0, nSteps, 0.1],
        dimensions: '100x10',
        format: 'png',
        min: 0,
        max: nSteps,
        palette: palette
    };
}

function showLegend(indexName, visPalette) {
    var legendTitle = ui.Label({ value: indexName, style: { fontWeight: 'normal' } });
    var colorBar = ui.Thumbnail({
        image: ee.Image.pixelLonLat().select(0).int(),
        params: makeColorBarParams(visPalette.palette),
        style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px' }
    });
    var legendLabels = ui.Panel({
        widgets: [
            ui.Label(visPalette.min.toFixed(1), { margin: '4px 8px' }),
            ui.Label(((visPalette.max - visPalette.min) / 2 + visPalette.min).toFixed(1), {
                margin: '4px 8px',
                textAlign: 'center',
                stretch: 'horizontal'
            }),
            ui.Label(visPalette.max.toFixed(1), { margin: '4px 8px' })
        ],
        layout: ui.Panel.Layout.flow('horizontal')
    });
    legendPanel.add(legendTitle);
    legendPanel.add(colorBar);
    legendPanel.add(legendLabels);
}

function showMinValue(mdCollection) {
    var min = mdCollection.median();
    return min.reduceRegion({
        reducer: ee.Reducer.min(),
        geometry: site,
        scale: 30,
        maxPixels: 1e9
    });
}

function showMaxValue(mdCollection) {
    var max = mdCollection.max();
    return max.reduceRegion({
        reducer: ee.Reducer.max(),
        geometry: site,
        scale: 30,
        maxPixels: 1e9
    });
}

//=============================================================================
// Optimized showMap using a configuration loop
//=============================================================================
function showMap(mdCollection, dateEnd) {
    // Define palettes (you can adjust these if needed)
    var palette = {
        ndvi: ['red', 'yellow', 'green'],
        ndmi: ['DCF2F1', '7FC7D9', '365486', '0F1035'],
        sr: ['F3EDC8', 'EAD196', 'BF3131', '7D0A0A'],
        bm: ['43766C', 'F8FAE5', 'B19470', '76453B']
    };
    var visPolygonBorder = { color: 'red', width: 2 };

    // Clear panels and center map
    rightPanel.clear();
    legendPanel.clear();
    map.clear();
    map.centerObject(site);
    map.add(legendPanel);

    // Configuration for each layer to be visualized
    var layers = [
        { checkbox: chkbNdvi, band: 'NDVI', label: 'NDVI', palette: palette.ndvi, chartType: 'band' },
        { checkbox: chkbNdviDiff, band: 'NDVIdiff', label: 'NDVIdiff', palette: palette.ndvi, chartType: 'band' },
        { checkbox: chkbNdmi, band: 'NDMI', label: 'NDMI', palette: palette.ndmi, chartType: 'band' },
        { checkbox: chkbFpar, band: 'FPAR', label: 'FPAR', palette: palette.ndvi, chartType: 'band' },
        { checkbox: chkbSr, band: 'GMT_0900_DSR', label: 'SR (W/m^2)', palette: palette.sr, chartType: 'band' },
        { checkbox: chkbPar, band: 'PAR', label: 'PAR', palette: palette.ndvi, chartType: 'band' },
        { checkbox: chkbApar, band: 'APAR', label: 'APAR', palette: palette.ndvi, chartType: 'band' },
        { checkbox: chkbGpp, band: 'GPP', label: 'GPP', palette: palette.bm, chartType: 'bm' },
        { checkbox: chkbNpp, band: 'NPP', label: 'ปริมาณเชื้อเพลิงเฉลี่ยวิธี NPP (Kg/m^2)', palette: palette.bm, chartType: 'bm' },
        { checkbox: chkbBm, band: 'BIOMASS', label: 'ปริมาณเชื้อเพลิงเฉลี่ยวิธี Linear regression (Kg/m^2)', palette: palette.bm, chartType: 'bm' }
    ];

    var chartBandArr = [];
    var chartBmArr = [];

    // Loop through each layer configuration
    layers.forEach(function (layer) {
        if (layer.checkbox.getValue()) {
            var currentBand = layer.band;
            var min = showMinValue(mdCollection.select(currentBand));
            var max = showMaxValue(mdCollection.select(currentBand));
            print(min);
            print(max);
            var vis = {
                // min: min.get(currentBand).getInfo(),
                min: 0,
                max: 7, //max.get(currentBand).getInfo(),
                palette: layer.palette
            };
            map.addLayer(mdCollection.select(currentBand).median(), vis, layer.label, true, 0.8);
            showLegend(layer.label, vis);
            if (layer.chartType === 'band') {
                chartBandArr.push(currentBand);
            } else if (layer.chartType === 'bm') {
                chartBmArr.push(currentBand);
            }
        }
    });

    // Add study area (converted to a line)
    map.addLayer(site.map(convertPolygonToLine), visPolygonBorder, "study area", true);

    // Create charts
    showBmChart(mdCollection, chartBmArr, site);
    showChart(mdCollection, chartBandArr, site);

    // Update charts when clicking on the map
    map.onClick(function (e) {
        var ptn = ee.Geometry.Point(e.lon, e.lat);
        rightPanel.clear();
        showBmChart(mdCollection, chartBmArr, ptn);
        showChart(mdCollection, chartBandArr, ptn);
    });
}

//=============================================================================
// Export and Zonal Statistics Functions (unchanged)
//=============================================================================
function exportToCSV(sampledValues, endDate) {
    Export.table.toDrive({
        collection: sampledValues,
        description: 'sampling_point_5d_' + endDate,
        fileFormat: 'CSV'
    });
}

function zonalStat(mdCollection, feature, dateEnd) {
    var sampledValues = mdCollection.median().sampleRegions({
        collection: feature,
        scale: 500,
        properties: ['id'],
        geometries: true
    });
    exportToCSV(sampledValues, dateEnd);
    return sampledValues;
}

//=============================================================================
// Optimized loadData function
//=============================================================================
var site = mt;
function loadData() {
    // Use a mapping for site selection
    var sites = { 'ud': ud, 'mt': mt };
    var siteSelected = siteSelectUi.getValue();
    site = sites[siteSelected] || mt;

    var dd = dateSliderUi.getValue();
    var dateEnd = ee.Date(dd[1]).format('YYYY-MM-dd');
    var dateComposite = dateCompositeUi.getValue() * -1;

    // Retrieve image collections
    var dataset = getDataset(dateEnd, dateComposite);
    var filter = ee.Filter.equals({ leftField: 'system:time_start', rightField: 'system:time_start' });
    var join = ee.Join.inner();

    // Reproject collections
    var mdProj = dataset.md.map(reProject);
    var mcdProj = dataset.mcd.map(reProject);

    // Compute indices
    var mdIndex = mdProj.map(calIndex);
    var mdBiomass = mdIndex.map(calBM);
    var mdIndexFpar = mdBiomass.map(calFpar);
    var mcdPar = mcdProj.map(calPar);
    var joinPar = join.apply(mdIndexFpar, mcdPar, filter);
    var mdIndexFparPar = joinPar.map(mergeBands);

    // NDVI difference calculation
    var mdNdvi = mdIndex.select('NDVI');
    var ndviList = mdNdvi.toList(mdNdvi.size());
    var ndviDiff = ndviList.slice(1).zip(ndviList.slice(0, -1)).map(calDiffNdvi);

    mdNdvi = mdIndex.select('NDVI');
    var ndviCount = mdNdvi.size();
    print('NDVI image count:', ndviCount);

    ndviDiff;
    if (ndviCount.gt(1)) {
        ndviList = mdNdvi.toList(ndviCount);
        ndviDiff = ndviList.slice(1).zip(ndviList.slice(0, -1)).map(calDiffNdvi);
    } else {
        print('Not enough images to calculate NDVI difference');
        ndviDiff = ee.List([]);
    }

    // Join NDVI difference with other collections


    var joinNdvidiff = join.apply(mdIndexFparPar, ndviDiff, filter);
    var mdIndexFparParNdvidiff = joinNdvidiff.map(mergeBands);

    var listIndexFparParNdvidiff = mdIndexFparParNdvidiff.toList(mdIndexFparParNdvidiff.size());
    var allCollection = ee.ImageCollection.fromImages(listIndexFparParNdvidiff);
    var mdCollection = allCollection.map(calApar);

    // Show the map with the computed collection
    showMap(mdCollection, dateEnd.getInfo());
}

//=============================================================================
// UI ELEMENTS
//=============================================================================
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

var txtCloudSlideUi = ui.Label({
    value: 'เลือก % การปกคลุมของเมฆ',
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(txtCloudSlideUi);

var cloudSliderUi = ui.Slider({ min: 0, max: 100, value: 50, style: { width: '90%' } });
leftPanel.add(cloudSliderUi);

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

var txtDateCompositeUi = ui.Label({
    value: 'เลือกจำนวนวันย้อนหลัง',
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(txtDateCompositeUi);

var dateItems = [
    { label: '3 วัน', value: 3 },
    { label: '5 วัน', value: 5 },
    { label: '7 วัน', value: 7 },
    { label: '14 วัน', value: 14 },
    { label: '30 วัน', value: 30 },
    { label: '60 วัน', value: 60 },
    { label: '120 วัน', value: 120 },
    { label: '180 วัน', value: 180 },
    { label: '360 วัน', value: 360 }
];
var dateCompositeUi = ui.Select({ items: dateItems, value: 14, style: { width: '80%' } });
leftPanel.add(dateCompositeUi);

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
var chkbNdviDiff = ui.Checkbox({ label: 'NDVI diff', value: false });
leftPanel.add(chkbNdviDiff);
var chkbNdmi = ui.Checkbox({ label: 'Normalized Difference Moisture Index: NDMI', value: false });
leftPanel.add(chkbNdmi);
var chkbFpar = ui.Checkbox({ label: 'Fraction of Photosynthetically Active Radiation: FPAR', value: false });
leftPanel.add(chkbFpar);
var chkbSr = ui.Checkbox({ label: 'Surface Radiation: SR', value: false });
leftPanel.add(chkbSr);
var chkbPar = ui.Checkbox({ label: 'Photosynthetically Active Radiation: PAR', value: false });
leftPanel.add(chkbPar);
var chkbApar = ui.Checkbox({ label: 'Absorption Photosynthetically Active Radiation: APAR', value: false });
leftPanel.add(chkbApar);
var chkbGpp = ui.Checkbox({ label: 'Gross Primary Productivity: GPP', value: false });
leftPanel.add(chkbGpp);

// Attach event listeners
cloudSliderUi.onChange(loadData);
siteSelectUi.onChange(loadData);
dateSliderUi.onChange(loadData);
dateCompositeUi.onChange(loadData);
chkbNdvi.onChange(loadData);
chkbNdviDiff.onChange(loadData);
chkbNdmi.onChange(loadData);
chkbSr.onChange(loadData);
chkbFpar.onChange(loadData);
chkbPar.onChange(loadData);
chkbApar.onChange(loadData);
chkbGpp.onChange(loadData);
chkbNpp.onChange(loadData);
chkbBm.onChange(loadData);

// Initial data load
loadData();

// Map style settings (unchanged)
var mapGrayscale = [
    { "featureType": "administrative", "elementType": "all", "stylers": [{ "saturation": "-100" }] },
    { "featureType": "administrative.province", "elementType": "all", "stylers": [{ "visibility": "off" }] },
    { "featureType": "landscape", "elementType": "all", "stylers": [{ "saturation": -100 }, { "lightness": 65 }, { "visibility": "on" }] },
    { "featureType": "poi", "elementType": "all", "stylers": [{ "saturation": -100 }, { "lightness": "50" }, { "visibility": "simplified" }] },
    { "featureType": "road", "elementType": "all", "stylers": [{ "saturation": "-100" }] },
    { "featureType": "road.highway", "elementType": "all", "stylers": [{ "visibility": "simplified" }] },
    { "featureType": "road.arterial", "elementType": "all", "stylers": [{ "lightness": "30" }] },
    { "featureType": "road.local", "elementType": "all", "stylers": [{ "lightness": "40" }] },
    { "featureType": "transit", "elementType": "all", "stylers": [{ "saturation": -100 }, { "visibility": "simplified" }] },
    { "featureType": "water", "elementType": "geometry", "stylers": [{ "hue": "#ffff00" }, { "lightness": -25 }, { "saturation": -97 }] },
    { "featureType": "water", "elementType": "labels", "stylers": [{ "lightness": -25 }, { "saturation": -100 }] }
];
// map.setOptions('Map Grayscale', { 'Map Grayscale': mapGrayscale });
