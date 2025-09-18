ui.root.clear();

var map1 = ui.Map();
var map2 = ui.Map();

var mapPanel = ui.SplitPanel({
    firstPanel: map1,
    secondPanel: map2,
    orientation: 'horizontal'
});

var layerPanel = ui.Panel({
    widgets: [ui.Label({
        value: 'ร่องรอยการเผาไหม้',
        style: {
            fontSize: '20px',
            fontWeight: '800'
        }
    })],
    style: { width: '20%' }
});

var toolPanel = ui.Panel({
    widgets: [ui.Label({ value: 'tool Panel' })],
    style: { width: '20%' }
})

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

var labelDirection = ui.Label({
    value: 'เลือกพื้นที่',
    style: {
        fontSize: '16px',
        fontWeight: '800'
    },
});
layerPanel.add(labelDirection);

var siteItems = [
    { label: 'แม่ทาเหนือ เชียงใหม่', value: 'mt' },
    { label: 'สบเตี๊ยะ เชียงใหม่', value: 'st' },
    { label: 'ป่าชุมชน อุตรดิตถ์', value: 'ud' },
    { label: 'ขุนยวม แม่ฮ่องสอน', value: 'ky' },
    { label: 'เวียงสา น่าน', value: 'vs' }
];
var siteSelectUi = ui.Select({
    items: siteItems,
    placeholder: 'Select an option',
    value: 'ud',
});
layerPanel.add(siteSelectUi);

var labelDirection = ui.Label({
    value: 'ระบุช่วงเวลาที่ต้องการ',
    style: {
        fontSize: '16px',
        fontWeight: '800'
    },
});
layerPanel.add(labelDirection);

var labelDirection = ui.Label({
    value: '(ระบุวันที่  Year-Month-Day เช่น 2023-01-16)',
    style: {
        fontSize: '12px'
    }
});
layerPanel.add(labelDirection);

var startDateUi = ui.Textbox({
    placeholder: 'insert date start',
    value: '2023-01-01'
});
layerPanel.add(startDateUi);

var labelDirection = ui.Label({ value: 'ถึง ' });
layerPanel.add(labelDirection);

var endDateUi = ui.Textbox({
    placeholder: 'insert date start',
    value: '2023-01-31'
});
layerPanel.add(endDateUi);

var labelDirection = ui.Label({
    value: 'ระบุ % ปกคลุมของเมฆ ',
    style: {
        fontSize: '16px',
        fontWeight: '800'
    },
});
layerPanel.add(labelDirection);

var cloudSliderUi = ui.Slider({
    min: 0,
    max: 100,
    value: 30,
    style: {
        width: '70%'
    }
})
layerPanel.add(cloudSliderUi);


var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/meatha_n");
var st = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/soubtea");
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/ud_bound");
var ky = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/khunyoam");
var vs = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/winagsa");

var vis_true = {
    min: 0.0,
    max: 0.4,
    bands: ['B4', 'B3', 'B2'],
};

var visPalette = {
    min: -0.3,
    max: 0.5,
    palette: ['4f5bd5', 'd62976', 'feda75', 'feda75'],
};

map1.setOptions('TERRAIN');
map2.setOptions('HYBRID');

ui.Map.Linker([map1, map2]);

function removeLayer(layerName) {
    var layers1 = map1.layers();
    var numLayers1 = layers1.length();
    for (var i = 0; i < numLayers1; i++) {
        var layer1 = layers1.get(i);
        if (layer1.getName() === layerName) {
            map1.layers().remove(layer1);
            return;
        }
    }

    var layers2 = map2.layers();
    var numLayers2 = layers2.length();
    for (var j = 0; j < numLayers2; j++) {
        var layer2 = layers2.get(i);
        if (layer2.getName() === layerName) {
            map2.layers().remove(layer2);
            return;
        }
    }
}

function showMap(startDate, endDate, studyArea, cloudCover) {
    removeLayer('NBR');
    removeLayer('true color');
    removeLayer('Burn Scars');

    var S2_SR = ee.ImageCollection('COPERNICUS/S2_SR')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudCover))
        .filter(ee.Filter.date(startDate, endDate));

    var prepare = function (img) {
        return img.clip(studyArea).divide(10000);
    }

    var s2 = S2_SR.map(prepare);

    // var nir = s2.select('B12').median(); // NIR
    // var red = s2.select('B8').median(); // Red
    var band12 = s2.median().select('B12');
    var band11 = s2.median().select('B11');

    var expression = '10 * B12 - 9.8 * B11 + 2';

    var NIRBI = s2.median().expression(expression, {
        'B12': band12,
        'B11': band11
    }).rename('NIRBI');

    var NBR = s2.median().normalizedDifference(['B8A', 'B12']);

    var threshold = 0.8;
    var burnScars = NIRBI.lt(threshold);

    map1.addLayer(NBR, visPalette, 'NBR', 1, 0.6)
    // map1.addLayer(NIRBI, visPalette, 'NIRBI',1, 0.8);
    map2.addLayer(s2, vis_true, 'true color', 0, 0.8)
    map2.addLayer(burnScars, { palette: ['white', 'red'] }, 'Burn Scars', 1, 0.7);

}

function loadData() {
    var startDate = ee.Date({ date: startDateUi.getValue() });
    var endDate = ee.Date({ date: endDateUi.getValue() });
    var cloudCover = cloudSliderUi.getValue();
    var studyAreaText = siteSelectUi.getValue();
    var studyArea;

    if (studyAreaText == 'mt') {
        studyArea = mt;
        map1.centerObject(studyArea);
        map2.centerObject(studyArea);
        showMap(startDate, endDate, studyArea, cloudCover);
    } else if (studyAreaText == 'st') {
        studyArea = st;
        map1.centerObject(studyArea);
        map2.centerObject(studyArea);
        showMap(startDate, endDate, studyArea, cloudCover);
    } else if (studyAreaText == 'ud') {
        studyArea = ud;
        map1.centerObject(studyArea);
        map2.centerObject(studyArea);
        showMap(startDate, endDate, studyArea, cloudCover);
    } else if (studyAreaText == 'ky') {
        studyArea = ky;
        map1.centerObject(studyArea);
        map2.centerObject(studyArea);
        showMap(startDate, endDate, studyArea, cloudCover);
    } else if (studyAreaText == 'vs') {
        studyArea = vs;
        map1.centerObject(studyArea);
        map2.centerObject(studyArea);
        showMap(startDate, endDate, studyArea, cloudCover);
    }
}

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

function showLegend() {
    var indexName;

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

    var labelDirection = ui.Label({
        value: 'สัญลักษณ์',
        style: {
            fontSize: '16px',
            fontWeight: '800'
        },
    });
    layerPanel.add(labelDirection);



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
        style: {
            fontSize: '16px',
            margin: '0 0 0 6px'
        }
    });

    colorEntry.add(colorBox);
    colorEntry.add(colorLabel);
    layerPanel.add(colorEntry);

    var labelBri = ui.Label({
        value: 'ดัชนีพื้นที่เปิดโล่ง',
        style: {
            fontSize: '16px',
            margin: '0 0 0 5px'
        },
    });
    layerPanel.add(labelBri);

    layerPanel.add(legendTitle);
    layerPanel.add(colorBar);
    layerPanel.add(legendLabels);
}

// btnSubmit.onClick(loadData);
startDateUi.onChange(loadData);
endDateUi.onChange(loadData);
cloudSliderUi.onChange(loadData);
siteSelectUi.onChange(loadData);
loadData();
showLegend()
