// Load the feature collection and get its bounds
var ud_paktub = ee.FeatureCollection("projects/ee-sakda-451407/assets/paktab");
var bound = ud_paktub.geometry().bounds();

// Clear the UI and initialize the map
ui.root.clear();
var mapPanel = ui.Map();
mapPanel.setOptions('SATELLITE');

// Create a panel for layers and controls
var layerPanel = ui.Panel({
    widgets: [ui.Label({
        value: 'ข้อมูล NDMI, NDWI และ NDVI',
        style: { fontSize: '20px', fontWeight: '800' }
    })],
    style: { width: '20%' }
});

// Create a panel for charts
var chartPanel = ui.Panel({
    widgets: [ui.Label({
        value: 'Daily Charts',
        style: { fontSize: '20px', fontWeight: '800' }
    })],
    style: { width: '30%' }
});

// Create the main panel layout
var mainPanel = ui.Panel({
    widgets: [layerPanel, mapPanel, chartPanel],
    layout: ui.Panel.Layout.flow('horizontal'),
    style: { width: "100%", height: "100%" }
});
ui.root.add(mainPanel);

// Function to get the image collection and calculate indices
function getCollection(param, year) {
    var startDate = year + '-01-01';
    var endDate = year + '-04-30';

    // Filter the Sentinel-2 collection
    var s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bound)
        .filterDate(startDate, endDate)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));

    // Calculate the selected index and copy the time property.
    var calculateIndex;
    if (param === 'NDWI') {
        // NDWI: (Green - NIR) / (Green + NIR)
        calculateIndex = s2Collection.map(function (image) {
            return image.normalizedDifference(['B3', 'B8'])
                .rename('NDWI')
                .clip(bound)
                .copyProperties(image, ['system:time_start']);
        });
    } else if (param === 'NDMI') {
        // NDMI: (NIR - SWIR) / (NIR + SWIR)
        calculateIndex = s2Collection.map(function (image) {
            return image.normalizedDifference(['B8', 'B11'])
                .rename('NDMI')
                .clip(bound)
                .copyProperties(image, ['system:time_start']);
        });
    } else if (param === 'NDVI') {
        // NDVI: (NIR - Red) / (NIR + Red)
        calculateIndex = s2Collection.map(function (image) {
            return image.normalizedDifference(['B8', 'B4'])
                .rename('NDVI')
                .clip(bound)
                .copyProperties(image, ['system:time_start']);
        });
    }

    // Compute the median of the collection to get a single image for visualization.
    var medianImage = calculateIndex.median();

    // Get the min and max values for visualization.
    var minMax = medianImage.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: bound,
        scale: 10,
        bestEffort: true
    });

    var paramMin = ee.Number(minMax.get(param + '_min')).getInfo();
    var paramMax = ee.Number(minMax.get(param + '_max')).getInfo();

    // Visualization parameters.
    var visParam = {
        min: paramMin,
        max: paramMax,
        palette: ['#7b3294', '#f7f7f7', '#008837']
    };

    // Add the layer to the map.
    mapPanel.addLayer(medianImage.select(param), visParam, param, true, 0.8);

    return calculateIndex;
}

// Function to create daily charts based on the available date.
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

// Add a dropdown for year selection.
var years = [
    { label: 'ปี 2019', value: '2019' },
    { label: 'ปี 2020', value: '2020' },
    { label: 'ปี 2021', value: '2021' },
    { label: 'ปี 2022', value: '2022' },
    { label: 'ปี 2023', value: '2023' },
    { label: 'ปี 2024', value: '2024' },
];
var selectYear = ui.Select({
    items: years,
    value: '2022'
});
layerPanel.add(ui.Label({ value: "เลือกปี" }));
layerPanel.add(selectYear);

// Create a legend.
var legend = ui.Panel({
    style: {
        position: 'bottom-left',
        padding: '8px 15px'
    }
});
var legendTitle = ui.Label({
    value: 'สัญลักษณ์ NDMI, NDWI และ NDVI ',
    style: { fontWeight: 'bold', fontSize: '18px', margin: '0 0 4px 0', padding: '0' }
});
legend.add(legendTitle);

function makeLegendRow(color, name) {
    var colorBox = ui.Label({
        style: {
            backgroundColor: color,
            padding: '10px',
            margin: '0 0 4px 0'
        }
    });
    var description = ui.Label({
        value: name,
        style: { margin: '0 0 4px 6px' }
    });
    return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
    });
}

legend.add(makeLegendRow('#008837', 'Low'));
legend.add(makeLegendRow('#ffffbf', 'Medium'));
legend.add(makeLegendRow('#7b3294', 'High'));
layerPanel.add(legend);

// Style for the boundary.
var boundStyle = ud_paktub.style({
    color: 'blue',
    fillColor: '00000000',
    width: 2
});

// Function to update the map and charts based on the selected year.
function updateMap() {
    mapPanel.clear();
    var year = selectYear.getValue();
    var ndmiCollection = getCollection('NDMI', year);
    var ndwiCollection = getCollection('NDWI', year);
    var ndviCollection = getCollection('NDVI', year);
    mapPanel.addLayer(boundStyle, {}, 'ป่าชุมชนบ้านปากทับ');

    mapPanel.centerObject(bound);

    // Create daily charts.
    createDateCharts(ndmiCollection, ndwiCollection, ndviCollection);
}

// Add an event listener for the year dropdown.
selectYear.onChange(updateMap);

// Initialize the map.
updateMap();
