var site = ee.FeatureCollection("projects/earthengine-380405/assets/thapla");
var points = ee.FeatureCollection("projects/earthengine-380405/assets/paktab_sampling");
var poly = ee.Geometry.Polygon(
    [[[100.41393344491567, 17.741727506427726],
    [100.41393344491567, 17.739438531998285],
    [100.4200274237975, 17.739438531998285],
    [100.4200274237975, 17.741727506427726]]], null, false);

// ui
ui.root.clear();
// var map = ui.Map();

var mapPanel = ui.Map();

var rightPanel = ui.Panel({
    widgets: [ui.Label('rightPanel')],
    style: { width: '30%' }
});

var leftPanel = ui.Panel({
    widgets: [ui.Label('leftPanel')],
    style: { width: '20%' }
});

var midPanel = ui.SplitPanel({
    firstPanel: mapPanel,
    secondPanel: rightPanel,
    orientation: 'horizontal',
})

var mainPanel = ui.SplitPanel({
    firstPanel: leftPanel,
    secondPanel: ui.Panel(midPanel),
    orientation: 'horizontal'
})

// var midPanel = ui.SplitPanel({
//   firstPanel: mapPanel,
//   secondPanel: bottomPanel,
//   orientation: 'vertical'
// });

// var rightPanel = ui.SplitPanel({
//   firstPanel: ui.Panel(midPanel),
//   secondPanel: infoPanel,
//   orientation: 'horizontal',
// })

// var mainPanel = ui.SplitPanel({
//   firstPanel: panelOne,
//   secondPanel: ui.Panel(panelTwo),
//   orientation: 'horizontal'
// })

ui.root.add(mainPanel);

function getDataset(dateEnd) {
    var d = ee.Date(dateEnd);
    var dateStart = d.advance(-5, 'day').format('yyyy-MM-dd');

    var mdData = ee.ImageCollection('MODIS/061/MOD09GA')
        .filterDate(dateStart, dateEnd);

    var mcdData = ee.ImageCollection('MODIS/061/MCD18A1')
        .filterDate(dateStart, dateEnd)
        .select('GMT_0900_DSR');

    return { md: mdData, mcd: mcdData }
}

function calIndex(image) {
    var ndvi = image
        .normalizedDifference({ bandNames: ['sur_refl_b02', 'sur_refl_b01'] })
        .rename('NDVI')
    var ndmi = image
        .normalizedDifference({ bandNames: ['sur_refl_b02', 'sur_refl_b06'] })
        .rename('NDMI')
    var combined = ndvi.addBands(ndmi);
    var combinedWithProperties = combined.copyProperties({
        source: image,
        properties: ['system:time_start']
    });
    return combinedWithProperties;
}

function reProject(image) {
    return image.clip(site).reproject({ crs: "EPSG:32647", scale: 500 })
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

function calDiffNdvi(image) {
    var imagePair = ee.List(image);
    var currentImage = ee.Image(imagePair.get(0));
    var previousImage = ee.Image(imagePair.get(1));
    var ndviDiff = currentImage.subtract(previousImage).rename('NDVIdiff');
    var ndviWithNdvidiff = currentImage.addBands(ndviDiff)
    return ndviWithNdvidiff.select('NDVIdiff');
}

function calFpar(image) {
    var fpar = image.select('NDVI').multiply(1.5).subtract(-0.1).rename('FPAR');
    return image.addBands(fpar)
}

function calPar(image) {
    var dsr24hr = image.select('GMT_0900_DSR').multiply(18000).divide(1000000)
    var par = dsr24hr.multiply(0.45).rename('PAR');
    var parWithProperties = par.copyProperties({
        source: image,
        properties: ['system:time_start']
    });
    return image.addBands(parWithProperties);
}

function calApar(image) {
    var apar = image.select('FPAR').multiply(image.select('PAR')).rename('APAR');
    var gpp = apar.multiply(1.8).rename('GPP');
    var npp = gpp.multiply(0.45).rename('NPP');
    return image.addBands(apar).addBands(gpp).addBands(npp);
}

function mergeBands(feature) {
    var image1 = ee.Image(feature.get('primary'));
    var image2 = ee.Image(feature.get('secondary'));
    var mergedImage = image1.addBands(image2);
    return mergedImage;
}

function showChart(mdCollection, site) {

    var chartUi = ui.Chart.image.series({
        imageCollection: mdCollection,
        region: site,
        reducer: ee.Reducer.mean(),
        scale: 500,
        xProperty: 'system:time_start'
    });

    rightPanel.clear()
    rightPanel.add(chartUi)
}

function showMap(mdCollection, dateEnd) {
    var visBand = {
        min: 1000,
        max: 100,
        bands: ['sur_refl_b04', 'sur_refl_b03', 'sur_refl_b02'],
    }

    var visPalette = {
        min: -1,
        max: 1,
        palette: ['red', 'yellow', 'green']
    }

    var visBiomass = {
        min: 0,
        max: 5,
        palette: ['red', 'yellow', 'green']
    }

    mapPanel.centerObject(site);
    // Map.addLayer(mdProj.median(), visBand , "true color", false, 0.8);
    // Map.addLayer(mdCollection.select('NDVI').median(), visPalette , "NDVI", true, 0.8);
    // Map.addLayer(mdCollection.select('NDVIdiff').median(), visPalette , "NDVIdiff", true, 0.8);
    // Map.addLayer(mdCollection.select('NDMI').median(), visPalette , "NDMI", true, 0.8);
    // Map.addLayer(mdCollection.select('FPAR').median(), visPalette , "FPAR", true, 0.8);
    // Map.addLayer(mdCollection.select('PAR').median(), visPalette , "PAR", true, 0.8);
    // Map.addLayer(mdCollection.select('APAR').median(), visPalette , "APAR", true, 0.8);
    // Map.addLayer(mdCollection.select('GPP').median(), visBiomass , "GPP", true, 0.8);
    mapPanel.addLayer(mdCollection.select('NPP').median(), visBiomass, "NPP " + dateEnd, true, 0.8);
}

function exportToCSV(sampledValues, endDate) {
    Export.table.toDrive({
        collection: sampledValues,
        description: 'sampling_point_5d_' + endDate,
        fileFormat: 'CSV'
    });
}

function zonalStat(mdCollection, feature, dateEnd) {
    var sampledValues = mdCollection.median()
        .sampleRegions({
            collection: feature,
            scale: 500,
            properties: ['id'],
            geometries: true
        });

    exportToCSV(sampledValues, dateEnd);
    return sampledValues;
}

function init(dateEnd) {
    var dataset = getDataset(dateEnd);
    var filter = ee.Filter.equals({
        leftField: 'system:time_start',
        rightField: 'system:time_start'
    });

    var join = ee.Join.inner();

    // convert to 32647
    var mdProj = dataset.md.map(reProject);
    var mcdProj = dataset.mcd.map(reProject);

    // NDVI, NDMI calculation
    var mdIndex = mdProj.map(calIndex);

    // FPAR calculation
    var mdIndexFpar = mdIndex.map(calFpar);

    // PAR calculation
    var mcdPar = mcdProj.map(calPar);
    var joinPar = join.apply(mdIndexFpar, mcdPar, filter);
    var mdIndexFparPar = joinPar.map(mergeBands);

    // NDVIdiff
    var mdNdvi = mdIndex.select('NDVI');
    var ndviList = mdNdvi.toList(mdNdvi.size());
    var ndviDiff = ndviList.slice(1).zip(ndviList.slice(0, -1)).map(calDiffNdvi);
    var joinNdvidiff = join.apply(mdIndexFparPar, ndviDiff, filter);
    var mdIndexFparParNdvidiff = joinNdvidiff.map(mergeBands);

    // APAR, GPP, NPP calculation
    var listIndexFparParNdvidiff = mdIndexFparParNdvidiff.toList(mdIndexFparParNdvidiff.size());
    var allCollection = ee.ImageCollection.fromImages(listIndexFparParNdvidiff);
    var mdCollection = allCollection.map(calApar);

    showChart(mdCollection, site);
    showMap(mdCollection, dateEnd);

    // var zStat = zonalStat(mdCollection, points, dateEnd);
}

function changeDate() {
    var date = dateSliderUi.getValue()
    print(date);
}

var txtSlideUi = ui.Label({ value: 'เลือก % การปกคลุมของเมฆ', style: { margin: '4px 8px' } });
leftPanel.add(txtSlideUi);

var sliderUi = ui.Slider({
    min: 0,
    max: 100,
    value: 50,
    style: { width: '90%' }
});
leftPanel.add(sliderUi);

var txtDateUi = ui.Label({ value: 'เลือกวันที่', style: { margin: '4px 8px' } });
leftPanel.add(txtDateUi);
var dateSliderUi = ui.DateSlider({
    start: '2010-01-01',
    style: { width: '80%' }
});
leftPanel.add(dateSliderUi);

var txtDateCompositeUi = ui.Label({ value: 'เลือกจำนวนวัน ที่ต้องการ composite', style: { margin: '4px 8px' } });
leftPanel.add(txtDateCompositeUi);

var dateItems = [
    { label: '3 วัน', value: 3 },
    { label: '5 วัน', value: 5 },
    { label: '7 วัน', value: 7 },
    { label: '15 วัน', value: 15 },
    { label: '30 วัน', value: 30 },
];
var dateCompositeUi = ui.Select({
    item: dateItems,
    style: { width: '80%' }
});
leftPanel.add(dateCompositeUi);


dateSliderUi.onChange(changeDate)
// set date of data
var dateArray = ['2023-11-15', '2023-11-20', '2023-11-25',
    '2023-11-30', '2023-12-05', '2023-12-10',
    '2023-12-15', '2023-12-20', '2023-12-25',
    '2023-12-30', '2024-01-05']

init(dateArray[0]);
// dateArray.forEach(function (i) {
//     init(i);
// });



// add statics feature
var visPolygonBorder = {
    color: 'red',
    width: 2,
}

var siteLine = site.map(convertPolygonToLine);
mapPanel.addLayer(poly, visPolygonBorder, "site", true);
mapPanel.addLayer(siteLine, visPolygonBorder, "site", true);
// Map.addLayer(points, { color: 'blue' }, "sampling point", true);
