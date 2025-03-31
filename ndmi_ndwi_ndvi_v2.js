// Data collections
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/meatha_n");

// Global variable to hold the current study area.
var currentSite = ud;  // Default site is 'ud'.

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


var rightPanel = ui.Panel({ widgets: [ui.Label('สัญลักษณ์')], style: { width: '30%' } });
var leftPanel = ui.Panel({ style: { width: '20%' } });
var midPanel = ui.SplitPanel({ firstPanel: map, secondPanel: rightPanel, orientation: 'horizontal' });
var mainPanel = ui.SplitPanel({ firstPanel: leftPanel, secondPanel: ui.Panel(midPanel), orientation: 'horizontal' });

rightPanel.add(legendPanel);
ui.root.add(mainPanel);

// Function to compute NDVI and add it as a new band.
function compute_ndvi(image) {
    var ndvi = image.normalizedDifference(['B8', 'B4'])
        .rename('NDVI')
        .clip(currentSite);
    return image.addBands(ndvi);
}

function compute_ndmi(image) {
    var ndmi = image.normalizedDifference(['B8', 'B11'])
        .rename('NDMI')
        .clip(currentSite);
    return image.addBands(ndmi);
}

function compute_ndwi(image) {
    var ndwi = image.normalizedDifference(['B3', 'B8'])
        .rename('NDWI')
        .clip(currentSite);
    return image.addBands(ndwi);
}

// Function to integrate 3PGS logic and estimate biomass.
// The equations calculate FPAR, PAR, APAR, GPP, and NPP.
function computeBiomass(image) {
    // FPAR: Fraction of absorbed photosynthetically active radiation.
    // (Note: subtract(-0.1) is equivalent to adding 0.1)
    var fpar = image.select('NDVI').multiply(1.5).add(0.1).rename('FPAR');

    // dsr24hr: Converts daily solar radiation to a 24-hour value.
    var dsr24hr = image.select('GMT_0900_DSR')
        .multiply(18000)
        .divide(1000000)
        .rename('DSR24hr');

    // PAR: Photosynthetically active radiation (assumes 45% of DSR is PAR).
    var par = dsr24hr.multiply(0.45).rename('PAR');

    // APAR: Absorbed PAR is the product of FPAR and PAR.
    var apar = fpar.multiply(par).rename('APAR');

    // GPP: Gross primary production (conversion factor 1.8).
    var gpp = apar.multiply(1.8).rename('GPP');

    // NPP: Net primary production (45% of GPP).
    var npp = gpp.multiply(0.45).rename('NPP');

    // Biomass: carbon To Biomass Factor = 2.5;
    var bm = npp.multiply(2.5).rename('BM');

    // Biomass TUM equation: Vol = (7.25923*(NDVI^3)) - (13.419*(NDVI^2)) + (6.4542*(NDVI)) - 0.2305
    var bmt = image.expression(
        '7.25923 * pow(NDVI, 3) - 13.419 * pow(NDVI, 2) + 6.4542 * NDVI - 0.2305',
        { 'NDVI': image.select('NDVI') }
    ).rename('BMT');

    // Add the computed bands to the image.
    return image.addBands([fpar, dsr24hr, par, apar, gpp, npp, bm, bmt]);
}


function getDataset(dateEnd, dateComposite) {
    var d = ee.Date(dateEnd);
    var dateStart = d.advance(-dateComposite, 'day').format('yyyy-MM-dd');

    // Create a composite from MOD09GA
    var mdData = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filter(ee.Filter.date(dateStart, dateEnd))
        .filterBounds(currentSite)
    // .map(compute_ndvi)
    // .map(compute_ndmi)
    // .map(compute_ndwi)
    // .select(['NDVI', 'NDMI', 'NDWI'])

    // var stackedImage = mdData.addBands(mcdData);

    return mdData;
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

function createDateCharts(ndmiCollection, ndwiCollection, ndviCollection) {
    // Clear the chart panel.
    chartPanel.clear();

    // Helper function to compute daily mean for a given image collection and band name.
    function computeDailyMean(collection, bandName) {
        // Add a date property formatted as YYYY-MM-dd.
        var collectionWithDate = collection.map(function (image) {
            var dateStr = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
            return image.set('date', dateStr);
        });
        // Get a distinct list of available dates.
        var distinctDates = collectionWithDate.aggregate_array('date').distinct();
        // For each date, compute the mean value if there are images.
        var dailyFeatures = distinctDates.map(function (dateStr) {
            dateStr = ee.String(dateStr);
            var dailyImages = collectionWithDate.filter(ee.Filter.eq('date', dateStr));
            var dailyMean = ee.Number(
                dailyImages.mean().reduceRegion({
                    reducer: ee.Reducer.mean(),
                    geometry: bound,
                    scale: 10,
                    bestEffort: true
                }).get(bandName)
            );
            return ee.Feature(null, { date: dateStr, mean: dailyMean });
        });
        return ee.FeatureCollection(dailyFeatures).sort('date');
    }

    // Compute daily means for each index.
    var ndmiDaily = computeDailyMean(ndmiCollection, 'NDMI');
    var ndwiDaily = computeDailyMean(ndwiCollection, 'NDWI');
    var ndviDaily = computeDailyMean(ndviCollection, 'NDVI');

    // Create charts from the feature collections with vertical x-axis labels.
    var ndmiChart = ui.Chart.feature.byFeature(ndmiDaily, 'date', 'mean')
        .setOptions({
            title: 'Daily NDMI',
            hAxis: {
                title: 'Date',
                slantedText: true,
                slantedTextAngle: 90
            },
            vAxis: { title: 'NDMI' }
        });

    var ndwiChart = ui.Chart.feature.byFeature(ndwiDaily, 'date', 'mean')
        .setOptions({
            title: 'Daily NDWI',
            hAxis: {
                title: 'Date',
                slantedText: true,
                slantedTextAngle: 90
            },
            vAxis: { title: 'NDWI' }
        });

    var ndviChart = ui.Chart.feature.byFeature(ndviDaily, 'date', 'mean')
        .setOptions({
            title: 'Daily NDVI',
            hAxis: {
                title: 'Date',
                slantedText: true,
                slantedTextAngle: 90
            },
            vAxis: { title: 'NDVI' }
        });

    // Add charts to the chart panel.
    chartPanel.add(ndmiChart);
    chartPanel.add(ndwiChart);
    chartPanel.add(ndviChart);
}


function showLegend(indexName, visPalette) {
    // Clear previous legend items.
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

var palette = {
    ndvi: ['d7191c', 'fdae61', 'ffffbf', 'a6d96a', '1a9641'],
    ndmi: ['e66101', 'fdb863', 'f7f7f7', 'b2abd2', '5e3c99'],
    ndwi: ['d01c8b', 'f1b6da', 'f7f7f7', 'b8e186', '4dac26'],
    sr: ['F3EDC8', 'EAD196', 'BF3131', '7D0A0A'],
    bm: ['43766C', 'F8FAE5', 'B19470', '76453B'],
};

var visPolygonBorder = { color: 'red', width: 2 };

function updateMap(dateEnd) {
    // Clear previous map layers.
    map.layers().reset();
    legendPanel.clear();
    var dataset = getDataset(dateEnd, 30);

    var ndvi_imgs = dataset.map(compute_ndvi);
    var ndvi_imgs_sel = ndvi_imgs.select('NDVI');
    var ndviStats = ndvi_imgs_sel.median().reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: currentSite,
        scale: 500,
        bestEffort: true
    });
    ndviStats.evaluate(function (stats) {
        var visParams = {
            min: stats.NDVI_min,
            max: stats.NDVI_max,
            palette: palette.ndvi
        };

        map.centerObject(currentSite, 11);
        map.addLayer(ndvi_imgs_sel.median(), visParams, 'NDVI');
        showLegend("NDVI", visParams);
    });

    var ndmi_imgs = dataset.map(compute_ndmi);
    var ndmi_imgs_sel = ndmi_imgs.select('NDMI');
    var ndmiStats = ndmi_imgs_sel.median().reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: currentSite,
        scale: 500,
        bestEffort: true
    });
    ndmiStats.evaluate(function (stats) {
        var visParams = {
            min: stats.NDMI_min,
            max: stats.NDMI_max,
            palette: palette.ndvi
        };

        map.centerObject(currentSite, 11);
        map.addLayer(ndmi_imgs_sel.median(), visParams, 'NDMI');
        showLegend("NDMI", visParams);
    });

    var ndwi_imgs = dataset.map(compute_ndwi);
    var ndwi_imgs_sel = ndwi_imgs.select('NDWI');
    var ndwiStats = ndwi_imgs_sel.median().reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: currentSite,
        scale: 500,
        bestEffort: true
    });
    ndwiStats.evaluate(function (stats) {
        var visParams = {
            min: stats.NDWI_min,
            max: stats.NDWI_max,
            palette: palette.ndvi
        };

        map.centerObject(currentSite, 11);
        map.addLayer(ndwi_imgs_sel.median(), visParams, 'NDWI');
        showLegend("NDWI", visParams);
    });


    map.addLayer(currentSite.map(convertPolygonToLine), visPolygonBorder, "study area", true);
}

// Create a date slider widget for selecting the end date.
// The slider range is set from January 1, 2020 to December 31, 2020.
var dateSlider = ui.DateSlider({
    start: '2020-01-01',
    value: '2024-12-31',
    period: 1,
    onChange: function (date) {
        // If the slider returns a DateRange, select the end date.
        var selectedDate = (date.start) ? date.end() : date;
        var dateStr = ee.Date(selectedDate).format('yyyy-MM-dd').getInfo();
        updateMap(dateStr);
    },
    style: { width: '80%' }
});

// Create a select widget to switch between study sites.
var siteSelect = ui.Select({
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' },
    items: [
        { label: "ปากทับ อุตรดิตถ์", value: "ud" },
        { label: "แม่ทาเหนือ เชียงใหม่", value: "mt" }
    ],
    value: 'ud',  // default value
    onChange: function (selected) {
        // Switch the current study area based on the selection.
        if (selected === 'ud') {
            currentSite = ud;
        } else if (selected === 'mt') {
            currentSite = mt;
        }
        // Update the map with the current date from the slider.
        var currentDate = dateSlider.getValue();
        var selectedDate = (currentDate[0]) ? currentDate[0] : currentDate[1];
        var dateStr = ee.Date(selectedDate).format('yyyy-MM-dd').getInfo();
        updateMap(dateStr);
    }
});

// Add the site select and date slider (with labels) to the left panel.
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
leftPanel.add(siteSelect);

var txtDateUi = ui.Label({
    value: 'เลือกวันที่',
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' }
});
leftPanel.add(txtDateUi);
leftPanel.add(dateSlider);

// Retrieve the default date from the slider and update the map.
var defaultValue = dateSlider.getValue();
var defaultDateStr = (defaultValue[0])
    ? ee.Date(defaultValue[0]).format('yyyy-MM-dd').getInfo()
    : ee.Date(defaultValue).format('yyyy-MM-dd').getInfo();

print('Default date selected:', defaultDateStr);
updateMap(defaultDateStr);
