ui.root.clear();
var site = ee.Geometry.Polygon(
    [[[100.01441440702139, 19.708677575751913],
    [100.01441440702139, 16.492859375194183],
    [101.42066440702139, 16.492859375194183],
    [101.42066440702139, 19.708677575751913]]]);


var mapGrayscale = [
    {
        "featureType": "administrative",
        "elementType": "all",
        "stylers": [{
            "saturation": "-100"
        }]
    }, {
        "featureType": "administrative.province",
        "elementType": "all",
        "stylers": [{
            "visibility": "off"
        }]
    }, {
        "featureType": "landscape",
        "elementType": "all",
        "stylers": [{
            "saturation": -100
        }, {
            "lightness": 65
        }, {
            "visibility": "on"
        }]
    }, {
        "featureType": "poi",
        "elementType": "all",
        "stylers": [{
            "saturation": -100
        }, {
            "lightness": "50"
        }, {
            "visibility": "simplified"
        }]
    }, {
        "featureType": "road",
        "elementType": "all",
        "stylers": [{
            "saturation": "-100"
        }]
    }, {
        "featureType": "road.highway",
        "elementType": "all",
        "stylers": [{
            "visibility": "simplified"
        }]
    }, {
        "featureType": "road.arterial",
        "elementType": "all",
        "stylers": [{
            "lightness": "30"
        }]
    }, {
        "featureType": "road.local",
        "elementType": "all",
        "stylers": [{
            "lightness": "40"
        }]
    }, {
        "featureType": "transit",
        "elementType": "all",
        "stylers": [{
            "saturation": -100
        }, {
            "visibility": "simplified"
        }]
    }, {
        "featureType": "water",
        "elementType": "geometry",
        "stylers": [{
            "hue": "#ffff00"
        }, {
            "lightness": -25
        }, {
            "saturation": -97
        }]
    }, {
        "featureType": "water",
        "elementType": "labels",
        "stylers": [{
            "lightness": -25
        }, {
            "saturation": -100
        }]
    }
]

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

function monthlySum(dataset, year, month, bandName) {
    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month');

    var mntCollection = dataset.filter(ee.Filter.date(start, end));
    var mntSum = mntCollection.reduce(ee.Reducer.sum());
    var a = mntSum.rename(bandName)
    var b = a.set('system:time_start', start.millis());

    return b;
}

function getByMonth(collection, bandName, yStart, yEnd) {
    var startDate = ee.Date.fromYMD(yStart, 1, 1);
    var endDate = ee.Date.fromYMD(yEnd, 1, 1);

    var collectionFiltered = collection.filter(ee.Filter.date(startDate, endDate));

    var years = ee.List.sequence(yStart, yEnd);
    var months = ee.List.sequence(1, 12);

    var dataMonth = years.map(function (y) {
        return months.map(function (m) {
            return getMonthlySum(collectionFiltered, y, m, bandName);
        });
    }).flatten();

    return ee.ImageCollection.fromImages(dataMonth);
}

function getBy8DaySum(collection, bandName, yStart, yEnd) {
    var startDate = ee.Date.fromYMD(yStart, 1, 1);
    var endDate = ee.Date.fromYMD(yEnd, 1, 1);

    var collectionFiltered = collection.filter(ee.Filter.date(startDate, endDate))

    var nPeriods = endDate.difference(startDate, 'day').divide(8).ceil();

    var startDates = ee.List.sequence(0, nPeriods.subtract(1)).map(function (n) {
        return startDate.advance(ee.Number(n).multiply(8), 'day');
    });

    print(startDates)

    var data8Day = startDates.map(function (date) {
        date = ee.Date(date);
        var periodEnd = date.advance(8, 'day');
        var periodCollection = collectionFiltered.filterDate(date, periodEnd);
        return periodCollection.reduce(ee.Reducer.sum()).rename(bandName).set('system:time_start', date.millis());
    })

    return ee.ImageCollection.fromImages(data8Day);
}

function getBy8DayModisSum(collection, bandName, dateList) {

    var startDates = ee.List(dateList).map(function (dateStr) {
        return ee.Date(dateStr);
    });

    var data8Day = startDates.map(function (date) {
        date = ee.Date(date);
        var periodEnd = date.advance(8, 'day');
        var periodCollection = collection.filterDate(date, periodEnd);
        return periodCollection.reduce(ee.Reducer.sum()).rename(bandName).set('system:time_start', date.millis());
    })

    return ee.ImageCollection.fromImages(data8Day);
}

function resize(image) {
    return image.clip(site).reproject({ crs: 'EPSG:32647', scale: 500 })
}

function getDataset(yearStart, yearEnd) {
    var yStart = ee.Number(yearStart).getInfo();
    var yEnd = ee.Number(yearEnd).getInfo();

    var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation');
    var mod16a2 = ee.ImageCollection('MODIS/061/MOD16A2').select('ET');

    var startDate = ee.Date.fromYMD(yStart, 1, 1);
    var endDate = ee.Date.fromYMD(yEnd, 1, 1);

    var collectionFiltered = mod16a2.filter(ee.Filter.date(startDate, endDate));

    var dateList = collectionFiltered.map(function (image) {
        return ee.Feature(null, { 'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd') });
    });

    var dates = dateList.aggregate_array('date');

    var GCN250_Average = ee.Image("users/jaafarhadi/GCN250/GCN250Average").select('b1').rename('average');
    var GCN250_Dry = ee.Image("users/jaafarhadi/GCN250/GCN250Dry").select('b1').rename('dry');
    var GCN250_Wet = ee.Image("users/jaafarhadi/GCN250/GCN250Wet").select('b1').rename('wet');

    // visualize the Dry GCN dataset
    var vis = {
        min: 40,
        max: 75,
        palette: ['Red', 'SandyBrown', 'Yellow', 'LimeGreen', 'Blue', 'DarkBlue']
    };

    // var rainCollection = getByMonth(chirps, 'precipitation', yStart, yEnd);
    // var evapCollection = getByMonth(mod16a2, 'ET', yStart, yEnd);

    // var rainCollection = get8DaySum(chirps, 'precipitation', yStart, yEnd);
    // var evapCollection = get8DaySum(mod16a2, 'ET', yStart, yEnd);

    var rainCollection = getBy8DayModisSum(chirps, 'precipitation', dates);
    var evapCollection = getBy8DayModisSum(mod16a2, 'ET', dates);

    var rain = rainCollection.map(resize);
    var evap = evapCollection.map(resize);

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

function safeSelectBand(image) {
    var selected = image.select([band]);
    var bandPresent = image.bandNames().contains(band);
    return ee.Image(ee.Algorithms.If(
        bandPresent,
        selected,
        ee.Image(0).rename(band).updateMask(0)
    ));
}

function showMap(ds, band) {
    var sumCollection = ds.select(band).sum();
    var min = getMin(sumCollection);
    var max = getMax(sumCollection);

    var visParam = {
        min: min.get(band).getInfo(),
        max: max.get(band).getInfo(),
        palette: ['001137', '0aab1e', 'e7eb05', 'ff4a2d', 'e90000'],
    }

    map.addLayer(sumCollection, visParam, band);
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

    var mergedCollection = joinSet.map(function (feature) {
        return mergeBands(feature);
    });

    return ee.ImageCollection(mergedCollection);
}

function init() {
    var yearStart = dateStartUi.getValue();
    var yearEnd = dateEndUi.getValue();

    var rawDataset = getDataset(parseInt(yearStart), parseInt(yearEnd));
    var collection = combineImage(rawDataset.rain, rawDataset.evap);

    map.clear();
    // var lib = require('users/sakdahomhuan/gg_engine:cmu_grayscale');
    // map.setOptions('Map Grayscale', { 'Map Grayscale': lib.mapGrayscale });

    map.centerObject(site, 8);
    map.setOptions('Map Grayscale', { 'Map Grayscale': mapGrayscale });

    showMap(collection, 'precipitation');
    showMap(collection, 'ET');
    showChart(collection, site)
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




