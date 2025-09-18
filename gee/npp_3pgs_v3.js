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
function calcNDVI(image) {
    var ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
        .rename('NDVI');
    return image.addBands(ndvi);
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
    var mdData = ee.ImageCollection('MODIS/061/MOD09GA')
        .filter(ee.Filter.date(dateStart, dateEnd))
        .filterBounds(currentSite)
        .map(calcNDVI)
        .select(['sur_refl_b02', 'sur_refl_b01', 'NDVI'])
        .median()
        .clip(currentSite);

    // Create a composite from MCD18A1
    var mcdData = ee.ImageCollection('MODIS/062/MCD18A1')
        .filter(ee.Filter.date('2023-11-01', '2024-03-30'))
        .filterBounds(currentSite)
        .select('GMT_0900_DSR')
        .median()
        .clip(currentSite);

    // Stack the images by adding the mcdData bands to mdData.
    var stackedImage = mdData.addBands(mcdData);

    return stackedImage;
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

// Function to show legend on the map.
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
    ndvi: ['red', 'yellow', 'green'],
    ndmi: ['DCF2F1', '7FC7D9', '365486', '0F1035'],
    sr: ['F3EDC8', 'EAD196', 'BF3131', '7D0A0A'],
    bm: ['43766C', 'F8FAE5', 'B19470', '76453B']
};

var visPolygonBorder = { color: 'red', width: 2 };
// Function to update the map based on the selected end date.
function updateMap(dateEnd) {
    // Clear previous map layers.
    map.layers().reset();
    legendPanel.clear();

    // Retrieve dataset for a 10-day composite ending at dateEnd.
    var dataset = getDataset(dateEnd, 30);

    var computeDataset = computeBiomass(dataset);
    print('computeDataset Image:', computeDataset);

    // Create an NDVI composite image from the mean of the collection.
    // var ndvi = calcNDVI(dataset);
    var ndviClipped = computeDataset.select('NDVI');
    // var ndviClipped = ndviComposite.clip(currentSite);

    // Compute NDVI min and max values over the current study area.
    var ndviStats = ndviClipped.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: currentSite,
        scale: 500,
        bestEffort: true
    });

    // Once statistics are computed, update the map and legend.
    ndviStats.evaluate(function (stats) {
        var visParams = {
            min: stats.NDVI_min,
            max: stats.NDVI_max,
            palette: palette.ndvi
        };

        map.centerObject(currentSite, 11);
        map.addLayer(ndviClipped, visParams, 'NDVI');
        // Display the legend using the computed visualization parameters.
        showLegend("NDVI", visParams);
    });

    // computeBiomass function is called to add biomass bands.
    var bmClipped = computeDataset.select('BM');
    var bmStats = bmClipped.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: currentSite,
        scale: 500,
        bestEffort: true
    });
    bmStats.evaluate(function (stats) {
        var bmParams = {
            min: stats.BM_min,
            max: stats.BM_max,
            palette: palette.bm
        };

        map.addLayer(bmClipped, bmParams, 'Biomass 3PGs');
        showLegend("Biomass 3PGs (kg/m²)", bmParams);
    });


    var bmtClipped = computeDataset.select('BMT');
    var bmtStats = bmtClipped.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: currentSite,
        scale: 500,
        bestEffort: true
    });
    bmtStats.evaluate(function (stats) {
        var bmtParams = {
            min: stats.BMT_min,
            max: stats.BMT_max,
            palette: palette.bm
        };

        map.addLayer(bmtClipped, bmtParams, 'Biomass (Tum Equa)');
        showLegend("Biomass (Tum Equa) (kg/m²)", bmtParams);
    });

    // Add study area (converted to a line)
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
