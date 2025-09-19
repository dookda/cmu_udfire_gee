// Data collections
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/paktab");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/meatha_n");
var ky = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/khunyoam");
var vs = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/winagsa");
var ms = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/measariang");

// Global variable to hold the current study area.
var currentSite = ud;  // Default site is 'ud'.

// Initialize UI
ui.root.clear();
var map = ui.Map();

var legendPanel = ui.Panel({
    style: {
        width: '180px',
        padding: '8px',
        backgroundColor: 'rgba(255, 255, 255, 0.7)'
    }
});
legendPanel.style().set({ position: 'bottom-left', margin: '0px 0px 30px 30px' });

var rightPanel = ui.Panel({ widgets: [ui.Label('สัญลักษณ์')], style: { width: '30%' } });
var leftPanel = ui.Panel({ style: { width: '20%' } });
var midPanel = ui.SplitPanel({ firstPanel: map, secondPanel: rightPanel, orientation: 'horizontal' });
var mainPanel = ui.SplitPanel({ firstPanel: leftPanel, secondPanel: ui.Panel(midPanel), orientation: 'horizontal' });

var chartPanel = ui.Panel({
    widgets: [ui.Label({
        value: 'Daily Charts',
        style: { fontSize: '20px', fontWeight: '800' }
    })],
    // style: { width: '30%' }
});

rightPanel.add(chartPanel);
map.add(legendPanel);
ui.root.add(mainPanel);

var test = ui.Label({
    value: 'test Daily Charts',
    style: { fontSize: '20px', fontWeight: '800' }
})

chartPanel.add(test);

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
    chartPanel.clear();
    function computeDailyMean(collection, bandName) {
        var collectionWithDate = collection.map(function (image) {
            var dateStr = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
            return image.set('date', dateStr);
        });
        var distinctDates = collectionWithDate.aggregate_array('date').distinct();
        var dailyFeatures = distinctDates.map(function (dateStr) {
            dateStr = ee.String(dateStr);
            var dailyImages = collectionWithDate.filter(ee.Filter.eq('date', dateStr));
            var dailyMean = ee.Number(
                dailyImages.mean().reduceRegion({
                    reducer: ee.Reducer.mean(),
                    geometry: currentSite,
                    scale: 10,
                    bestEffort: true
                }).get(bandName)
            );
            return ee.Feature(null, { date: dateStr, mean: dailyMean });
        });
        return ee.FeatureCollection(dailyFeatures).sort('date');
    }

    var ndmiDaily = computeDailyMean(ndmiCollection, 'NDMI');
    var ndwiDaily = computeDailyMean(ndwiCollection, 'NDWI');
    var ndviDaily = computeDailyMean(ndviCollection, 'NDVI');

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
    var legendTitle = ui.Label({
        value: indexName,
        style: {
            fontWeight: 'bold',
            backgroundColor: 'rgba(255, 255, 255, 0.1)'
        }
    });
    var colorBar = ui.Thumbnail({
        image: ee.Image.pixelLonLat().select(0).int(),
        params: makeColorBarParams(visPalette.palette),
        style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px' }
    });
    var legendLabels = ui.Panel({
        widgets: [
            ui.Label(visPalette.min.toFixed(1), {
                margin: '4px 8px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)'
            }),
            ui.Label(((visPalette.max - visPalette.min) / 2 + visPalette.min).toFixed(1), {
                margin: '4px 8px',
                textAlign: 'center',
                stretch: 'horizontal',
                backgroundColor: 'rgba(255, 255, 255, 0.1)'
            }),
            ui.Label(visPalette.max.toFixed(1), {
                margin: '4px 8px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)'
            })
        ],
        layout: ui.Panel.Layout.flow('horizontal'),
        style: {
            backgroundColor: 'rgba(255, 255, 255, 0.1)'
        }
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
        showLegend("ดัชนีความแตกต่างของพืชพรรณ: NDVI", visParams);
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
            palette: palette.ndmi
        };

        map.centerObject(currentSite, 11);
        map.addLayer(ndmi_imgs_sel.median(), visParams, 'NDMI');
        showLegend("ดัชนีความแตกต่างของความชื้น: NDMI", visParams);
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
            palette: palette.ndwi
        };

        map.centerObject(currentSite, 11);
        map.addLayer(ndwi_imgs_sel.median(), visParams, 'NDWI');
        showLegend("ดัชนีความแตกต่างของน้ำ: NDWI", visParams);
    });

    map.addLayer(currentSite.map(convertPolygonToLine), visPolygonBorder, "study area", true);
    createDateCharts(ndmi_imgs_sel, ndwi_imgs_sel, ndvi_imgs_sel)
}

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
        { label: "แม่ทาเหนือ เชียงใหม่", value: "mt" },
        { label: "ขุนยวม แม่ฮ่องสอน", value: "ky" },
        { label: "เวียงสา น่าน", value: "vs" },
        { label: "แม่สะเรียง แม่ฮ่องสอน", value: "ms" }
    ],
    value: 'ud',  // default value
    onChange: function (selected) {
        // Switch the current study area based on the selection.
        if (selected === 'ud') {
            currentSite = ud;
        } else if (selected === 'mt') {
            currentSite = mt;
        } else if (selected === 'ky') {
            currentSite = ky;
        } else if (selected === 'vs') {
            currentSite = vs;
        } else if (selected === 'ms') {
            currentSite = ms;
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
    value: 'ติดตาม NDVI, NDMI, NDWI',
    style: { margin: '4px 8px', fontSize: '20px', fontWeight: 'bold' }
});
leftPanel.add(txtTitle);

var txtSubTitle1 = ui.Label({
    value: 'ดัชนี้จากข้อมูล Sentinel-2',
    style: { margin: '4px 8px' }
});

var txtSubTitle2 = ui.Label({
    value: 'ดัชนีความแตกต่างของพืชพรรณ (Normalized Difference Vegetation Index: NDVI)',
    style: { margin: '4px 8px' }
});

var txtSubTitle3 = ui.Label({
    value: 'ดัชนีความแตกต่างของความชื้น (Normalized Difference Moisture Index: NDMI) และดัชนีความแตกต่างของน้ำ (Normalized Difference Water Index: NDWI) จากข้อมูล Sentinel-2',
    style: { margin: '4px 8px' }
});

var txtSubTitle4 = ui.Label({
    value: 'ดัชนีความแตกต่างของน้ำ (Normalized Difference Water Index: NDWI)',
    style: { margin: '4px 8px' }
});

leftPanel.add(txtSubTitle1);
leftPanel.add(txtSubTitle2);
leftPanel.add(txtSubTitle3);
leftPanel.add(txtSubTitle4);

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
