ui.root.clear();
var site = ee.Geometry.Polygon(
    [[[100.01441440702139, 19.708677575751913],
    [100.01441440702139, 16.492859375194183],
    [101.42066440702139, 16.492859375194183],
    [101.42066440702139, 19.708677575751913]]]);


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


var chartPanel = ui.Panel({
    // widgets: [ui.Label('bottomPanel')],
    style: {
        height: '50%',
        margin: '0px 0px 0px 0px'
    }
});

var leftPanel = ui.Panel({
    style: { width: '20%' }
});

var midPanel = ui.SplitPanel({
    firstPanel: map,
    secondPanel: chartPanel,
    orientation: 'vertical',
})

var mainPanel = ui.SplitPanel({
    firstPanel: leftPanel,
    secondPanel: ui.Panel(midPanel),
    orientation: 'horizontal'
})

ui.root.add(mainPanel);

function getMonthlySum(dataset, year, month, bandName) {
    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month');

    var mntCollection = dataset.filterDate(start, end);
    // print(mntCollection)
    var mntSum = mntCollection.reduce(ee.Reducer.sum());
    var a = mntSum.rename(bandName)
    var b = a.set('system:time_start', start.millis());

    return b;
}

function getDataset(yearStart, yearEnd) {
    print(ee.Number(yearStart).getInfo(), yearEnd)
    var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation')
    var mod16a2 = ee.ImageCollection('MODIS/006/MOD16A2').select('ET')

    var years = ee.List.sequence(ee.Number(yearStart), ee.Number(yearEnd));
    var months = ee.List.sequence(1, 12);

    var rainMonth = years.map(function (y) {
        return months.map(function (m) {
            return getMonthlySum(chirps, y, m, 'precipitation');
        });
    }).flatten();

    var evapMonth = years.map(function (y) {
        return months.map(function (m) {
            return getMonthlySum(mod16a2, y, m, 'ET');
        });
    }).flatten();

    var rain = ee.ImageCollection.fromImages(rainMonth);
    var evap = ee.ImageCollection.fromImages(evapMonth);

    return { rain: rain, evap: evap }
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
        vAxis: { title: 'ปริมาณน้ำฝน (ม.ม.)' },
        curveType: 'function',
    };

    chartUi.setOptions(chartOptions);
    chartPanel.clear();
    chartPanel.add(chartUi);
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

    // var lib = require('users/sakdahomhuan/gg_engine:cmu_grayscale');
    // map.setOptions('Map Grayscale', { 'Map Grayscale': lib.mapGrayscale });
    map.clear();
    map.centerObject(site);
    map.addLayer(rainSum, visParam, 'Precipitation');

}

function resize(image) {
    return image.clip(site).reproject({ crs: 'EPSG:32647', scale: 500 })
}

function mergeBands(feature) {
    var image1 = ee.Image(feature.get('primary'));
    var image2 = ee.Image(feature.get('secondary'));
    var mergedImage = image1.addBands(image2);
    return mergedImage;
}

function combineImage(rain, evap) {
    var filter = ee.Filter.equals({
        leftField: 'system:time_start',
        rightField: 'system:time_start'
    });

    var join = ee.Join.inner();
    var joinSet = join.apply(rain, evap, filter);
    print(joinSet);
    var merge = joinSet.map(mergeBands);
    print(merge);
}

function init() {
    var yearStart = dateStartUi.getValue();
    var yearEnd = dateEndUi.getValue();

    var dataset = getDataset(parseInt(yearStart), parseInt(yearEnd));

    var band = 'precipitation';
    var rainClip = dataset.rain.select(band).map(resize);

    var rain = dataset.rain.map(resize);
    var evap = dataset.evap.map(resize);

    combineImage(rain, evap)

    showMap(rainClip, band);
    showChart(rainClip, site)
}

var dateStartUi = ui.Textbox({
    placeholder: '2020',
    value: 2020
});
leftPanel.add(dateStartUi);

var dateEndUi = ui.Textbox({
    placeholder: '2024',
    value: 2024
});
leftPanel.add(dateEndUi);

var submitBtn = ui.Button({
    label: 'ตกลง',
})
leftPanel.add(submitBtn)

init();
submitBtn.onClick(init)


