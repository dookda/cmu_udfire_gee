ui.root.clear();

// Create map
var map = ui.Map();

var ud = ee.FeatureCollection("projects/earthengine-380405/assets/uttradit");
var mt = ee.FeatureCollection("projects/earthengine-380405/assets/meatha");
var st = ee.FeatureCollection("projects/earthengine-380405/assets/soubtea");


// สบเตี๊ยะ เชียงใหม่
var point1 = ee.Geometry.Point(98.6499538, 18.3732473);
var point2 = ee.Geometry.Point(98.6622527, 18.3662919);
var point3 = ee.Geometry.Point(98.6610433, 18.3711520);
var point4 = ee.Geometry.Point(98.6460676, 18.3823595);
var point5 = ee.Geometry.Point(98.6431057, 18.3788437);

var feature1 = ee.Feature(point1, { name: 'Point 1' });
var feature2 = ee.Feature(point2, { name: 'Point 2' });
var feature3 = ee.Feature(point3, { name: 'Point 3' });
var feature4 = ee.Feature(point4, { name: 'Point 4' });
var feature5 = ee.Feature(point5, { name: 'Point 5' });

// แม่ทาเหนือ
var point6 = ee.Geometry.Point(99.3126103, 18.6985221);
var point7 = ee.Geometry.Point(99.3222585, 18.6949385);
var point8 = ee.Geometry.Point(99.3186995, 18.7241396);
var point9 = ee.Geometry.Point(99.2722350, 18.7074061);
var point10 = ee.Geometry.Point(99.2667924, 18.6650878);
var point11 = ee.Geometry.Point(99.2531095, 18.6321305);
var point12 = ee.Geometry.Point(99.2891456, 18.6159649);

var feature6 = ee.Feature(point6, { name: 'Point 6' });
var feature7 = ee.Feature(point7, { name: 'Point 7' });
var feature8 = ee.Feature(point8, { name: 'Point 8' });
var feature9 = ee.Feature(point9, { name: 'Point 9' });
var feature10 = ee.Feature(point10, { name: 'Point 10' });
var feature11 = ee.Feature(point11, { name: 'Point 11' });
var feature12 = ee.Feature(point12, { name: 'Point 12' });

// Create a feature collection
var sb_fs = ee.FeatureCollection([feature6, feature7, feature8, feature9, feature10, feature11, feature12]);
var mt_fs = ee.FeatureCollection([feature1, feature2, feature3, feature4, feature5])
// map.addLayer(featureCollection)
// print(featureCollection);


// Panel area
var rightPanel = ui.Panel({
    widgets: [
        ui.Label({
            value: 'ระบบคำนวณดัชนีความแตกต่างพืชพรรณและความชื้น',
            style: { fontSize: '20px', fontWeight: 1000 }
        })
    ],
    style: { width: '30%', padding: '8px' }
});

var infoPanel = ui.Panel({
    // widgets: [ui.Label({ value: 'This is an informational panel' })],
    style: { height: '35%', padding: '0px' }
});

var leftPanel = ui.SplitPanel({
    firstPanel: map,
    secondPanel: infoPanel,
    orientation: 'vertical'
});

var mainPanel = ui.SplitPanel({
    firstPanel: ui.Panel({ widgets: [leftPanel] }),
    secondPanel: rightPanel,
    orientation: 'horizontal'
});

ui.root.add(mainPanel);

var index;
var selectedVi;
var studyArea;
var studyAreaText;

var siteLabelSelect = ui.Label({
    value: 'พื้นที่',
    style: { padding: '0px' }
});
rightPanel.add(siteLabelSelect);

var siteItems = [
    { label: 'อุตรดิตถ์', value: 'ud' },
    { label: 'แม่ทาเหนือ เชียงใหม่', value: 'mt' },
    { label: 'สบเตี๊ยะ เชียงใหม่', value: 'st' }
];

var siteSelectUi = ui.Select({
    items: siteItems,
    placeholder: 'Select an option',
    value: 'ud',
});
rightPanel.add(siteSelectUi);

var satelliteLabelSelect = ui.Label({
    value: 'ดาวเทียม',
    style: { padding: '0px' }
});
rightPanel.add(satelliteLabelSelect);

var satelliteItems = [
    { label: 'TERRA MODIS', value: 'TERRA' },
    { label: 'AQUA MODIS', value: 'AQUA' }
];

var satelliteSelectUi = ui.Select({
    items: satelliteItems,
    placeholder: 'Select an option',
    value: 'TERRA',
});
rightPanel.add(satelliteSelectUi);

var indexLabelSelect = ui.Label({
    value: 'ดัชนี',
    style: { padding: '0px' }
});
rightPanel.add(indexLabelSelect);

var indexItems = [
    { label: 'Normalized Difference Vegetation Index (NDVI)', value: 'NDVI' },
    { label: 'Normalized Difference Moisture Index (NDMI)', value: 'NDMI' }
];

var ndviCheckboxUi = ui.Checkbox({
    label: "Normalized Difference Vegetation Index (NDVI)",
    value: 'NDVI',
});
rightPanel.add(ndviCheckboxUi);

var ndmiCheckboxUi = ui.Checkbox({
    label: "Normalized Difference Moisture Index (NDMI)",
    value: 'NDMI',
});
rightPanel.add(ndmiCheckboxUi);

var labelEndDate = ui.Label({
    value: 'วันเริ่มต้น',
    style: { padding: '0px' }
});
rightPanel.add(labelEndDate);

var startDateUi = ui.DateSlider({
    start: '2020-01-01',
    period: 1,
    style: { padding: '0px' }
});
startDateUi.style().set('width', '100%');
rightPanel.add(startDateUi);

var labelEndDate = ui.Label({
    value: 'วันสิ้นสุด',
    style: { padding: '0px' }
});
rightPanel.add(labelEndDate);

var endDateUi = ui.DateSlider({
    start: '2020-01-01',
    period: 1,
    style: { padding: '0px' }
});
endDateUi.style().set('width', '100%');
rightPanel.add(endDateUi);

var currentDate = new Date();
startDateUi.setValue(ee.Date(currentDate).advance(-2, 'month'));
endDateUi.setValue(ee.Date(currentDate));


var legendPanel = ui.Panel();
rightPanel.add(legendPanel);

// Legend
var ndviPalette = ['db1e14', 'e1eb21', '34eb7d'];
var ndmiPalette = ['feda75', 'fa7e1e', 'd62976', '962fbf', '4f5bd5']

function calPalette(idx) {
    var palette
    if (idx == 'ndvi') {
        palette = {
            min: -1.0,
            max: 1.0,
            palette: ndviPalette
        }
    } else {
        palette = {
            min: -1.0,
            max: 1.0,
            palette: ndmiPalette
        }
    }
    return palette
}

// Dataset
var MOD09GQ = ee.ImageCollection("MODIS/061/MOD09GQ");
var MYD09GQ = ee.ImageCollection("MODIS/061/MYD09GQ");

var MOD09GA = ee.ImageCollection("MODIS/061/MOD09GA");
var MYD09GA = ee.ImageCollection("MODIS/061/MYD09GA");

function makeColorBarParams(palette) {
    var nSteps = 10;
    return {
        bbox: [0, 0, nSteps, 0.1],
        dimensions: '100x10',
        format: 'png',
        min: 0,
        max: nSteps,
        palette: palette,
    };
}

function showLegend(idx) {
    var indexName;
    var visPalette;
    if (idx == 'ndvi') {
        indexName = 'Normalized Difference Vegetation Index (NDVI)';
        visPalette = calPalette('ndvi');
    } else {
        indexName = 'Normalized Difference Moisture Index (NDMI)';
        visPalette = calPalette('ndmi');
    }

    var legendLabels = ui.Panel({
        widgets: [
            ui.Label(visPalette.min, { margin: '4px 8px' }),
            ui.Label(
                ((visPalette.max - visPalette.min) / 2 + visPalette.min),
                { margin: '4px 8px', textAlign: 'center', stretch: 'horizontal' }),
            ui.Label(visPalette.max, { margin: '4px 8px' })
        ],
        layout: ui.Panel.Layout.flow('horizontal')
    });

    var legendTitle = ui.Label({
        value: indexName,
        style: { fontWeight: 'normal' }
    });

    var colorBar = ui.Thumbnail({
        image: ee.Image.pixelLonLat().select(0).int(),
        params: makeColorBarParams(visPalette.palette),
        style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px' },
    });

    legendPanel.add(legendTitle);
    legendPanel.add(colorBar);
    legendPanel.add(legendLabels);
}

function loadDataset(startDate, endDate, satellite) {
    if (satellite == 'TERRA') {
        return MOD09GA.filterDate({
            start: startDate,
            end: endDate
        })
    } else {
        return MYD09GA.filterDate({
            start: startDate,
            end: endDate
        });
    }
}

function calNomalizedIndex(img) {
    var ndvi = img
        .clip(studyArea)
        .normalizedDifference({ bandNames: ['sur_refl_b01', 'sur_refl_b02'] })
        .rename('NDVI')

    var ndmi = img
        .clip(studyArea)
        .normalizedDifference({ bandNames: ['sur_refl_b02', 'sur_refl_b06'] })
        .rename('NDMI')

    var combined = ndvi.addBands(ndmi);

    var combinedWithProperties = combined.copyProperties({
        source: img,
        properties: ['system:time_start']
    });

    return combinedWithProperties;
}

function showChart(region) {
    var chartUi = ui.Chart.image.series({
        imageCollection: selectedVi,
        region: region,
        reducer: ee.Reducer.mean(),
        scale: 500,
        xProperty: 'system:time_start',
    });

    var chartOptions = {
        hAxis: {
            title: 'วันที่',
        },
        vAxis: {
            title: index,
        }
    };

    if (index == 'NDVI/NDMI') {
        chartOptions.series = {
            0: { curveType: 'function', color: '962fbf' },
            1: { curveType: 'function', color: '34eb7d' },
        }
    } else if (index == 'NDVI') {
        chartOptions.series = {
            0: { curveType: 'function', color: '34eb7d' }
        }
    } else if (index == 'NDMI') {
        chartOptions.series = {
            0: { curveType: 'function', color: '962fbf' }
        }
    }

    chartUi.setOptions(chartOptions);

    infoPanel.clear();
    infoPanel.add(chartUi);
}

function removeLayer(layerName) {
    var layers = map.layers();
    var numLayers = layers.length();
    for (var i = 0; i < numLayers; i++) {
        var layer = layers.get(i);
        if (layer.getName() === layerName) {
            map.layers().remove(layer);
            return;
        }
    }
}

function addMap(idx, dataset) {
    if (idx == 'ndvi') {
        map.addLayer({
            eeObject: dataset,
            visParams: calPalette('ndvi'),
            name: "ndvi",
            shown: true,
            opacity: 0.8
        })
    } else {
        map.addLayer({
            eeObject: dataset,
            visParams: calPalette('ndmi'),
            name: "ndmi",
            shown: true,
            opacity: 0.8
        })
    }
}

function showMap(dateStart, dateEnd, studyArea, satellite) {
    var ndviCheckbox = ndviCheckboxUi.getValue();
    var ndmiCheckbox = ndmiCheckboxUi.getValue();

    var dataset = loadDataset(dateStart, dateEnd, satellite);
    var vi = dataset.map(calNomalizedIndex);

    removeLayer('ndvi');
    removeLayer('ndmi');
    legendPanel.clear();

    if (ndviCheckbox === true && ndmiCheckbox === true) {
        selectedVi = vi;
        index = 'NDVI/NDMI';
        addMap('ndvi', vi.select("NDVI").median());
        addMap('ndmi', vi.select("NDMI").median());
        showChart(studyArea);
        showLegend('ndvi');
        showLegend('ndmi');
    } else if (ndviCheckbox === true && ndmiCheckbox === false) {
        selectedVi = vi.select("NDVI");
        index = 'NDVI';
        addMap('ndvi', vi.select("NDVI").median());
        showChart(studyArea);
        showLegend('ndvi');
    } else if (ndviCheckbox === false && ndmiCheckbox === true) {
        selectedVi = vi.select("NDMI");
        index = 'NDMI';
        addMap('ndmi', vi.select("NDMI").median());
        showChart(studyArea);
        showLegend('ndmi');
    } else {
        infoPanel.clear();
    }
}

function handleOnClick(e) {
    var point = ee.Geometry.Point(e.lon, e.lat);
    showChart(point);
}

function addFeature() {
    var markerColor = '00FF00';

    Map.addLayer(featureCollection.style({ pointSize: 8, color: markerColor }), {}, 'Markers');

    var markerClick = function (coords) {
        var features = featureCollection.filterBounds(ee.Geometry.Point(coords.lon, coords.lat));
        var label = features.first().get('label');
        print('Marker Label:', label);
    };

    Map.onClick(markerClick);
}

function loadData() {
    var startDate = ee.Date({ date: startDateUi.getValue()[0] });
    var endDate = ee.Date({ date: endDateUi.getValue()[0] });
    studyAreaText = siteSelectUi.getValue();

    var satellite = satelliteSelectUi.getValue();

    if (studyAreaText == 'ud') {
        studyArea = ud;
        map.centerObject(studyArea);
    } else if (studyAreaText == 'mt') {
        studyArea = mt;
        map.centerObject(studyArea);
    } else {
        studyArea = st;
        map.centerObject(studyArea);
    }

    showMap(startDate, endDate, studyArea, satellite);
}

siteSelectUi.onChange(loadData);
satelliteSelectUi.onChange(loadData);
ndviCheckboxUi.onChange(loadData);
ndmiCheckboxUi.onChange(loadData);
startDateUi.onChange(loadData);
endDateUi.onChange(loadData);
map.onClick(handleOnClick);
loadData();

map.setOptions({ mapTypeId: 'HYBRID' })