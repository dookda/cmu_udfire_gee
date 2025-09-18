/***** 1. Load and prepare data *****/

// Load MOD09GA surface reflectance data and filter by date
var mod09ga = ee.ImageCollection("MODIS/061/MOD09GA");

// Function to compute NDMI using NIR and SWIR bands from MOD09GA
var addNDMI = function (image) {
    var nir = image.select('sur_refl_b02').multiply(0.0001);
    var swir = image.select('sur_refl_b06').multiply(0.0001);
    var ndmi = nir.subtract(swir).divide(nir.add(swir)).rename('NDMI');
    return image.addBands(ndmi);
};

// Map the NDMI function over the collection
var mod09gaWithNDMI = mod09ga.map(addNDMI);

// Function to compute NDVI from MOD09GA bands (Red and NIR)
function computeNDVI(image) {
    var nir = image.select('sur_refl_b02').multiply(0.0001);
    var red = image.select('sur_refl_b01').multiply(0.0001);
    return nir.subtract(red).divide(nir.add(red)).rename('NDVI');
}

// Biomass equation using NDVI and NDMI:
// biomass = 0.2 + (0.1 * NDMI) - (0.4 * NDVI^2) + (0.3 * NDVI^3) - (0.05 * NDVI * NDMI)
function biomassEquation(image) {
    var ndvi = computeNDVI(image);
    var ndmi = image.select('NDMI');

    var biomass = ee.Image(0.2)
        .add(ndmi.multiply(0.1))
        .subtract(ndvi.pow(2).multiply(0.4))
        .add(ndvi.pow(3).multiply(0.3))
        .subtract(ndvi.multiply(ndmi).multiply(0.05))
        .rename('biomass');

    return image.addBands(biomass);
}

// Create a new collection that includes NDMI and biomass (NDVI computed on the fly)
var modCollection = mod09gaWithNDMI.map(biomassEquation);

// Define visualization parameters for NDVI and biomass
var ndviVis = {
    min: -0.2,
    max: 1,
    palette: ['red', 'yellow', 'green']
};
var biomassVis = {
    min: 0,
    max: 20,
    palette: ['#43766C', '#F8FAE5', '#B19470', '#76453B']
};

// Load study area feature collections
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/meatha_n");

/***** 2. Create UI components *****/
// 2.2 Date slider
var startDate = new Date('2020-01-01');
var endDate = new Date();
var dateSlider = ui.DateSlider({
    start: startDate,
    end: endDate,
    value: [startDate, endDate],
    period: 16, // MODIS 16-day composite
    onChange: updateMapAndCharts  // Re-calc when slider changes
});

// 2.3 Study area selector
var studyAreaSelector = ui.Select({
    items: ["ปากทับ อุตรดิตถ์", "แม่ทาเหนือ เชียงใหม่"],
    value: "ปากทับ อุตรดิตถ์", // Default
    placeholder: 'เลือกพื้นที่ศึกษา',
    onChange: updateMapAndCharts
});

// 2.4 Left panel (holding date slider and area selector)
var leftPanel = ui.Panel({
    widgets: [
        ui.Label('เลือกวันที่และพื้นที่ศึกษา', { fontWeight: 'bold' }),
        dateSlider,
        studyAreaSelector
    ],
    layout: ui.Panel.Layout.flow('vertical'),
    style: { width: '300px', padding: '8px' }
});

// 2.5 Map
var map = ui.Map();
map.setOptions('ROADMAP'); // default base map style

// 2.6 Right panel (for charts)
var ndviChartPanel = ui.Panel();
var biomassChartPanel = ui.Panel();
var rightPanel = ui.Panel({
    widgets: [
        ui.Label('Charts', { fontWeight: 'bold' }),
        ndviChartPanel,
        biomassChartPanel
    ],
    layout: ui.Panel.Layout.flow('vertical'),
    style: { width: '300px', padding: '8px' }
});

/***** 3. Legend panel on the map’s left side *****/

// Adapted legend functions

function makeColorBarParams(palette, minVal, maxVal) {
    return {
        bbox: [minVal, 0, maxVal, 0.1],
        dimensions: '100x10',
        format: 'png',
        min: minVal,
        max: maxVal,
        palette: palette
    };
}

function showLegend(legendPanel, indexName, visParams) {
    var legendTitle = ui.Label({
        value: indexName,
        style: { fontWeight: 'normal' }
    });

    var colorBar = ui.Thumbnail({
        image: ee.Image.pixelLonLat().select(0).int(),
        params: makeColorBarParams(visParams.palette, visParams.min, visParams.max),
        style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px' }
    });

    var legendLabels = ui.Panel({
        widgets: [
            ui.Label(visParams.min.toFixed(1), { margin: '4px 8px' }),
            ui.Label(
                (((visParams.max - visParams.min) / 2) + visParams.min).toFixed(1),
                { margin: '4px 8px', textAlign: 'center', stretch: 'horizontal' }
            ),
            ui.Label(visParams.max.toFixed(1), { margin: '4px 8px' })
        ],
        layout: ui.Panel.Layout.flow('horizontal')
    });

    legendPanel.add(legendTitle);
    legendPanel.add(colorBar);
    legendPanel.add(legendLabels);
}

// Create a legend panel and add it to the map.
var legendPanel = ui.Panel({
    style: { position: 'top-left', padding: '8px', backgroundColor: 'white' }
});
map.add(legendPanel);
showLegend(legendPanel, 'NDVI', ndviVis);
showLegend(legendPanel, 'มวลชีวภาพ (kg/m2)', biomassVis);

/***** 4. Combine panels into a final layout *****/

var middlePanel = ui.Panel({
    widgets: [leftPanel, map, rightPanel],
    layout: ui.Panel.Layout.flow('horizontal'),
    style: { stretch: 'horizontal' }
});

var rootPanel = ui.Panel({
    widgets: [middlePanel],
    layout: ui.Panel.Layout.flow('vertical'),
    style: { stretch: 'both' }
});

ui.root.clear();
ui.root.add(rootPanel);

/***** 5. Main update function *****/

function updateMapAndCharts() {
    // Get the current date range from the slider
    var range = dateSlider.getValue();
    var start = ee.Date(range[0]);
    var end = ee.Date(range[1]);

    // Select study area
    var selectedOption = studyAreaSelector.getValue();
    var studyArea = (selectedOption === "แม่ทาเหนือ เชียงใหม่") ? mt : ud;

    // Compute the mean image from modCollection over the selected date range
    var filtered = modCollection.filterDate(start, end);
    var meanImage = filtered.mean();

    // Compute mean NDVI from the mean image and select the biomass band
    var meanNDVI = computeNDVI(meanImage).clipToCollection(studyArea);
    var biomassImage = meanImage.select('biomass').clipToCollection(studyArea);

    // Display NDVI and biomass layers on the map
    map.layers().set(0, ui.Map.Layer(meanNDVI, ndviVis, 'NDVI'));
    map.layers().set(1, ui.Map.Layer(biomassImage, biomassVis, 'มวลชีวภาพ (kg/m2)'));
    map.centerObject(studyArea);

    // Create NDVI time series chart
    var ndviTimeSeries = ui.Chart.image.series({
        imageCollection: modCollection.filterDate(start, end).map(function (img) {
            return computeNDVI(img)
                .set('system:time_start', img.get('system:time_start'));
        }),
        region: studyArea.geometry(),
        reducer: ee.Reducer.mean(),
        scale: 250
    }).setOptions({
        title: 'NDVI Time Series',
        hAxis: { title: 'วันที่' },
        vAxis: { title: 'NDVI' }
    });

    // Create Biomass time series chart using the computed biomass band
    var biomassTimeSeries = ui.Chart.image.series({
        imageCollection: modCollection.filterDate(start, end).map(function (img) {
            return img.select('biomass')
                .set('system:time_start', img.get('system:time_start'));
        }),
        region: studyArea.geometry(),
        reducer: ee.Reducer.mean(),
        scale: 250
    }).setOptions({
        title: 'Biomass Time Series',
        hAxis: { title: 'วันที่' },
        vAxis: { title: 'มวลชีวภาพ (kg/m2)' }
    });

    ndviChartPanel.clear().add(ndviTimeSeries);
    biomassChartPanel.clear().add(biomassTimeSeries);
}

/***** 6. Define computeMean (optional helper) *****/

function computeMean(startDate, endDate) {
    var filtered = modCollection.filterDate(startDate, endDate);
    var meanImage = filtered.mean();
    var meanNDVI = computeNDVI(meanImage);
    var biomassImage = meanImage.select('biomass');
    return { ndvi: meanNDVI, biomass: biomassImage };
}

// 7. Initialize the map and charts
updateMapAndCharts();
