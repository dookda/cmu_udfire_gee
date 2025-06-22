// Data collections
var fbound = ee.FeatureCollection("projects/ee-sakda-451407/assets/forest_bound_sgpart");
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/ud_bound");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/meatha_n");

var currentSite = ud;

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

var rightPanel = ui.Panel({ widgets: [ui.Label('')], style: { width: '30%' } });
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

map.setOptions('SATELLITE');
map.setControlVisibility(true);
map.add(legendPanel);

rightPanel.add(chartPanel);
ui.root.add(mainPanel);

function reProject(image) {
    return image.reproject({ crs: "EPSG:32647", scale: 500 });
}

function computeNdvi(image) {
    var ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
        .rename('NDVI');
    return image.addBands(ndvi);
}

function computeBiomass(image) {
    var mcdData = ee.ImageCollection('MODIS/062/MCD18A1')
        .filter(ee.Filter.date('2023-11-01', '2024-03-30'))
        .filterBounds(currentSite)
        .select('GMT_0900_DSR')
        .median()
    // FPAR: Fraction of absorbed photosynthetically active radiation.
    // (Note: subtract(-0.1) is equivalent to adding 0.1)
    var fpar = image.select('NDVI').multiply(1.5).add(0.1).rename('FPAR');

    // dsr24hr: Converts daily solar radiation to a 24-hour value.
    var dsr24hr = mcdData.select('GMT_0900_DSR')
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

    var mdData = ee.ImageCollection('MODIS/061/MOD09GA')
        .filter(ee.Filter.date(dateStart, dateEnd))
        .filterBounds(currentSite)
        .map(reProject)
        .map(computeNdvi)
        .map(computeBiomass)
        .select(['sur_refl_b02', 'sur_refl_b01', 'NDVI', 'FPAR', 'DSR24hr', 'PAR', 'APAR', 'GPP', 'NPP', 'BM', 'BMT'])
    // .median()
    // .clip(currentSite);

    return mdData;
}


function createDateCharts(ndviCollection, bmCollection, bmtCollection) {
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

    var ndviDaily = computeDailyMean(ndviCollection, 'NDVI');
    var bmDaily = computeDailyMean(bmCollection, 'BM');
    var bmtDaily = computeDailyMean(bmtCollection, 'BMT');

    var ndviChart = ui.Chart.feature.byFeature(ndviDaily, 'date', 'mean')
        .setOptions({
            title: 'NDVI',
            hAxis: {
                title: 'Date',
                slantedText: true,
                slantedTextAngle: 90
            },
            vAxis: { title: 'NDVI' }
        });

    var bmChart = ui.Chart.feature.byFeature(bmDaily, 'date', 'mean')
        .setOptions({
            title: 'Biomass 3PGs',
            hAxis: {
                title: 'Date',
                slantedText: true,
                slantedTextAngle: 90
            },
            vAxis: { title: 'Biomass (kg/m²)' }
        });

    var bmtChart = ui.Chart.feature.byFeature(bmtDaily, 'date', 'mean')
        .setOptions({
            title: 'Biomas (Parinwat & Sakda Equation)',
            hAxis: {
                title: 'Date',
                slantedText: true,
                slantedTextAngle: 90
            },
            vAxis: { title: 'Biomass (kg/m²)' }
        });

    // Add charts to the chart panel.
    chartPanel.add(ndviChart);
    chartPanel.add(bmChart);
    chartPanel.add(bmtChart);
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

function showLegend(indexName, visPalette) {
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
        style: {
            stretch: 'horizontal',
            margin: '0px 8px',
            maxHeight: '24px'
        }
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
    ndmi: ['DCF2F1', '7FC7D9', '365486', '0F1035'],
    sr: ['F3EDC8', 'EAD196', 'BF3131', '7D0A0A'],
    bm: ['5e3c99', 'b2abd2', 'f7f7f7', 'fdb863', 'e66101']
};

var visPolygonBorder = { color: 'red', width: 2 };
function updateMap(dateEnd) {
    map.layers().reset();
    legendPanel.clear();

    var dataset = getDataset(dateEnd, 30);
    // print('Dataset:', dataset);

    var ndviCollection = dataset.select('NDVI');
    var ndviClipped = ndviCollection.median().clip(currentSite);
    var ndviStats = ndviClipped.reduceRegion({
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
        map.addLayer(ndviClipped, visParams, 'NDVI', true, 0.7);
        showLegend("NDVI", visParams);
    });

    var bmCollection = dataset.select('BM');
    var bmClipped = bmCollection.median().clip(currentSite);
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

        map.addLayer(bmClipped, bmParams, 'Biomass 3PGs', true, 0.7);
        showLegend("Biomass 3PGs (kg/m²)", bmParams);
    });

    var bmtCollection = dataset.select('BMT');
    var bmtClipped = bmtCollection.median().clip(currentSite);
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

        map.addLayer(bmtClipped, bmtParams, 'Biomass (Parinwat & Sakda Equation)', true, 0.7);
        showLegend("Biomass (Parinwat & Sakda Equation) (kg/m²)", bmtParams);
    });

    var fbund = siteSelect.getValue();

    if (fbund=="ud"){
        var styledFbound = fbound.style({
          color: 'FF0000',      
          fillColor: '00000000', 
          width: 2               
        });
        map.addLayer(styledFbound, {}, 'ขอบเขตป่าไม้', true);
    }

    map.addLayer(currentSite.map(convertPolygonToLine), visPolygonBorder, "study area", false);
    createDateCharts(ndviCollection, bmCollection, bmtCollection)
}

var dateSlider = ui.DateSlider({
    start: '2020-01-01',
    value: '2024-12-31',
    period: 1,
    onChange: function (date) {
        var selectedDate = (date.start) ? date.end() : date;
        var dateStr = ee.Date(selectedDate).format('yyyy-MM-dd').getInfo();
        updateMap(dateStr);
    },
    style: { width: '80%' }
});

var siteSelect = ui.Select({
    style: { margin: '4px 8px', fontSize: '18px', fontWeight: 'bold' },
    items: [
        { label: "อุตรดิตถ์", value: "ud" },
        { label: "แม่ทาเหนือ เชียงใหม่", value: "mt" }
    ],
    value: 'ud',
    onChange: function (selected) {
        if (selected === 'ud') {
            currentSite = ud;
        } else if (selected === 'mt') {
            currentSite = mt;
        }
        var currentDate = dateSlider.getValue();
        var selectedDate = (currentDate[0]) ? currentDate[0] : currentDate[1];
        var dateStr = ee.Date(selectedDate).format('yyyy-MM-dd').getInfo();
        updateMap(dateStr);
    }
});

var txtTitle = ui.Label({
    value: 'ติดตามปริมาณเชื้อเพลิง',
    style: { margin: '4px 8px', fontSize: '20px', fontWeight: 'bold' }
});
leftPanel.add(txtTitle);

var txtSubTitle0 = ui.Label({
    value: 'คำนวณปริมาณเชื้อเพลิงด้วย:',
    style: { margin: '4px 8px' }
});
leftPanel.add(txtSubTitle0);

var txtSubTitle1 = ui.Label({
    value: '1. จากวิธี 3PGs จากข้อมูลรายวันของดาวเทียม TERRA/AQUA MODIS ',
    style: { margin: '4px 8px' }
});
leftPanel.add(txtSubTitle1);

var txtSubTitle2 = ui.Label({
    value: '2. จากสมการที่พัฒนาขึ้นของโครงการ (Parinwat & Sakda Equation)',
    style: { margin: '4px 8px' }
});
leftPanel.add(txtSubTitle2);

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

var defaultValue = dateSlider.getValue();
var defaultDateStr = (defaultValue[0])
    ? ee.Date(defaultValue[0]).format('yyyy-MM-dd').getInfo()
    : ee.Date(defaultValue).format('yyyy-MM-dd').getInfo();

print('Default date selected:', defaultDateStr);
updateMap(defaultDateStr);
