

var site = ee.Geometry.Polygon(
    [[[100.01441440702139, 19.708677575751913],
    [100.01441440702139, 16.492859375194183],
    [101.42066440702139, 16.492859375194183],
    [101.42066440702139, 19.708677575751913]]]);

ui.root.clear()
var map = ui.Map();

var legendPanel = ui.Panel({
    style: {
        width: '180px',
        padding: '8px',
        backgroundColor: 'rgba(255, 255, 255, 0.9)'
    }
})
legendPanel.style().set({
    position: 'bottom-left',
    margin: '0px 0px 30px 30px'
});


var bottomPanel = ui.Panel({
    widgets: [ui.Label('bottomPanel')],
    style: { height: '30%' }
});

var leftPanel = ui.Panel({
    style: { width: '20%' }
});

var midPanel = ui.SplitPanel({
    firstPanel: map,
    secondPanel: bottomPanel,
    orientation: 'vertical',
})

var mainPanel = ui.SplitPanel({
    firstPanel: leftPanel,
    secondPanel: ui.Panel(midPanel),
    orientation: 'horizontal'
})

ui.root.add(mainPanel);

function getDataset() {
    var dateStart = '2023-01-01';
    var dateEnd = '2023-12-31'
    var dataset = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .filter(ee.Filter.date(dateStart, dateEnd));

    var mod16 = ee.ImageCollection('MODIS/006/MOD16A2')
        .filter(ee.Filter.date(dateStart, dateEnd));
    print(mod16);
    return dataset
}

function getMin(dataset) {
    var min = dataset.reduceRegion({
        reducer: ee.Reducer.min(),
        geometry: site,
        scale: 5000,
        maxPixels: 1e9
    });
    return min
}

function getMax(dataset) {
    var max = dataset.reduceRegion({
        reducer: ee.Reducer.max(),
        geometry: site,
        scale: 5000,
        maxPixels: 1e9
    });
    return max
}

function resize(image) {
    return image.clip(site).reproject({ crs: 'EPSG:32647', scale: 500 })
}

function showChart(dataset, region) {
    var chartUi = ui.Chart.image.series({
        imageCollection: dataset,
        region: region,
        reducer: ee.Reducer.mean(),
        scale: 500,
        xProperty: 'system:time_start'
    })

    var chartOptions = {
        hAxis: { title: 'วันที่' },
        vAxis: { title: 'index' },
        curveType: 'function',
    };

    chartUi.setOptions(chartOptions);

    bottomPanel.add(chartUi)

}

function showMap(dataset, band) {
    // var precipitation = dataset.select('precipitation');
    // var precipitationVis = {
    //     min: 1,
    //     max: 1500,
    //     palette: ['001137', '0aab1e', 'e7eb05', 'ff4a2d', 'e90000'],
    // };

    var rainSum = dataset.sum();
    var min = getMin(rainSum);
    var max = getMax(rainSum);


    var visParam = {
        min: min.get(band).getInfo(),
        max: max.get(band).getInfo(),
        palette: ['001137', '0aab1e', 'e7eb05', 'ff4a2d', 'e90000'],
    }

    var lib = require('users/sakdahomhuan/gg_engine:cmu_grayscale');
    map.setOptions('Map Grayscale', { 'Map Grayscale': lib.mapGrayscale });
    map.centerObject(site);
    map.addLayer(rainSum, visParam, 'Precipitation');


}

function init() {
    var dataset = getDataset();
    var band = 'precipitation';
    var rainClip = dataset.select(band).map(resize);

    showMap(rainClip, band);
    showChart(rainClip, site)
}

init()

var startYear = 2010;
var endYear = 2020;

// Create two date objects for start and end years.
var startDate = ee.Date.fromYMD(startYear, 1, 1);
var endDate = ee.Date.fromYMD(endYear + 1, 1, 1);

// Make a list with years.
var years = ee.List.sequence(startYear, endYear);

// Make a list with months.
var months = ee.List.sequence(1, 12);

// Import the CHIRPS dataset.
var CHIRPS = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD');

// Filter for the relevant time period.
CHIRPS = CHIRPS.filterDate(startDate, endDate);
var monthlyPrecip = ee.ImageCollection.fromImages(
    years.map(function (y) {
        return months.map(function (m) {
            var w = CHIRPS.filter(ee.Filter
                .calendarRange(y, y, 'year'))
                .filter(ee.Filter.calendarRange(m, m,
                    'month'))
                .sum();
            return w.set('year', y)
                .set('month', m)
                .set('system:time_start', ee.Date
                    .fromYMD(y, m, 1));

        });
    }).flatten()
);

print(monthlyPrecip);