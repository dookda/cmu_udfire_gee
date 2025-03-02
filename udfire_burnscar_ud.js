ui.root.clear();

// Initialize maps
var map1 = ui.Map();
var map2 = ui.Map();

// Create a split panel for the maps
var mapPanel = ui.SplitPanel({
    firstPanel: map1,
    secondPanel: map2,
    orientation: 'horizontal'
});

// Layer panel for controls and labels
var layerPanel = ui.Panel({
    widgets: [ui.Label({
        value: 'ร่องรอยการเผาไหม้',
        style: { fontSize: '20px', fontWeight: '800' }
    })],
    style: { width: '20%' }
});

// Tool panel (currently unused)
var toolPanel = ui.Panel({
    widgets: [ui.Label({ value: 'Tool Panel' })],
    style: { width: '20%' }
});

// Main layout
var rightPanel = ui.SplitPanel({
    firstPanel: ui.Panel({ widgets: [mapPanel] }),
    secondPanel: layerPanel,
    orientation: 'horizontal'
});

var mainPanel = ui.SplitPanel({
    firstPanel: toolPanel,
    secondPanel: ui.Panel({ widgets: [rightPanel] }),
    orientation: 'horizontal'
});

ui.root.add(rightPanel);

// Add UI elements to the layer panel
function addLabel(text, fontSize, fontWeight) {
    return ui.Label({
        value: text,
        style: { fontSize: fontSize, fontWeight: fontWeight }
    });
}

layerPanel.add(addLabel('เลือกพื้นที่', '16px', '800'));

var siteItems = [
    { label: 'ป่าชุมชน อุตรดิตถ์', value: 'ud' },
    { label: 'แม่ทาเหนือ เชียงใหม่', value: 'mt' },
    { label: 'สบเตี๊ยะ เชียงใหม่', value: 'st' }
];

var siteSelectUi = ui.Select({
    items: siteItems,
    placeholder: 'Select an option',
    value: 'mt',
});
layerPanel.add(siteSelectUi);

layerPanel.add(addLabel('ระบุช่วงเวลาที่ต้องการ', '16px', '800'));
layerPanel.add(addLabel('(ระบุวันที่  Year-Month-Day เช่น 2023-01-16)', '12px', 'normal'));

var startDateUi = ui.Textbox({
    placeholder: 'Insert start date',
    value: '2023-01-01'
});
layerPanel.add(startDateUi);

layerPanel.add(addLabel('ถึง', '16px', 'normal'));

var endDateUi = ui.Textbox({
    placeholder: 'Insert end date',
    value: '2023-01-31'
});
layerPanel.add(endDateUi);

layerPanel.add(addLabel('ระบุ % ปกคลุมของเมฆ', '16px', '800'));

var cloudSliderUi = ui.Slider({
    min: 0,
    max: 100,
    value: 30,
    style: { width: '70%' }
});
layerPanel.add(cloudSliderUi);

// Feature collections
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/meatha_n");
var st = ee.FeatureCollection("projects/ee-sakda-451407/assets/soubtea");
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");

// Visualization parameters
var vis_true = { min: 0.0, max: 0.4, bands: ['B4', 'B3', 'B2'] };
var vis_false = { min: 0.1, max: 0.4, bands: ['B11', 'B5', 'B4'] };
var visPalette = { min: -0.3, max: 0.5, palette: ['4f5bd5', 'd62976', 'feda75', 'feda75'] };

map1.setOptions('TERRAIN');
map2.setOptions('HYBRID');
ui.Map.Linker([map1, map2]);

// Function to remove layers by name
function removeLayer(layerName) {
    [map1, map2].forEach(function (map) {
        var layers = map.layers();
        var numLayers = layers.length();
        for (var i = 0; i < numLayers; i++) {
            var layer = layers.get(i);
            if (layer.getName() === layerName) {
                map.layers().remove(layer);
                break; // Exit the loop once the layer is found and removed
            }
        }
    });
}

// Function to load and display Sentinel-2 data
function showMap(startDate, endDate, studyArea, cloudCover) {
    removeLayer('S2');
    removeLayer('Burn Scars');

    var S2_SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudCover))
        .filter(ee.Filter.date(startDate, endDate));

    var prepare = function (img) {
        return img.clip(studyArea).divide(10000);
    };

    var s2 = S2_SR.map(prepare).median();

    var band12 = s2.select('B12');
    var band11 = s2.select('B11');

    var expression = '10 * B12 - 9.8 * B11 + 2';
    var NIRBI = s2.expression(expression, { 'B12': band12, 'B11': band11 }).rename('NIRBI');

    var threshold = 0.8;
    var burnScars = NIRBI.lt(threshold);

    map1.addLayer(s2, vis_false, 'S2', 1, 0.9);
    map2.addLayer(burnScars, { palette: ['white', 'red'] }, 'Burn Scars', 1, 0.7);
}

// Function to load data based on user inputs
function loadData() {
    var startDate = ee.Date(startDateUi.getValue());
    var endDate = ee.Date(endDateUi.getValue());
    var cloudCover = cloudSliderUi.getValue();
    var studyAreaText = siteSelectUi.getValue();

    var studyArea;
    switch (studyAreaText) {
        case 'mt':
            studyArea = mt;
            break;
        case 'st':
            studyArea = st;
            break;
        default:
            studyArea = ud;
    }

    map1.centerObject(studyArea, 10);
    map2.centerObject(studyArea, 10);
    showMap(startDate, endDate, studyArea, cloudCover);
}

// Add event listeners
startDateUi.onChange(loadData);
endDateUi.onChange(loadData);
cloudSliderUi.onChange(loadData);
siteSelectUi.onChange(loadData);

// Initial load
loadData();

// Function to show legend
function showLegend() {
    var colors = ['white', 'red'];
    var labels = ['No Data', 'ร่องรอยการเผาไหม้'];

    var colorEntry = ui.Panel({
        layout: ui.Panel.Layout.flow('horizontal'),
        style: { margin: '0 0 4px 0' }
    });

    var colorBox = ui.Label({
        style: {
            backgroundColor: colors[1],
            padding: '6px',
            margin: '0 5px 4px 10px',
        }
    });

    var colorLabel = ui.Label({
        value: labels[1],
        style: { fontSize: '16px', margin: '0 0 0 6px' }
    });

    colorEntry.add(colorBox);
    colorEntry.add(colorLabel);
    layerPanel.add(colorEntry);

    layerPanel.add(addLabel('ดัชนีพื้นที่เปิดโล่ง', '16px', 'normal'));
}

showLegend();