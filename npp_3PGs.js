var site = ee.FeatureCollection("projects/earthengine-380405/assets/thapla");
var points = ee.FeatureCollection("projects/earthengine-380405/assets/paktab_sampling");
var poly = ee.Geometry.Polygon(
    [[[100.41393344491567, 17.741727506427726],
    [100.41393344491567, 17.739438531998285],
    [100.4200274237975, 17.739438531998285],
    [100.4200274237975, 17.741727506427726]]], null, false);

// init ui
ui.root.clear();
var map = ui.Map();

var rightPanel = ui.Panel({
    widgets: [ui.Label('rightPanel')],
    style: { width: '30%' }
});

var leftPanel = ui.Panel({
    widgets: [ui.Label('leftPanel')],
    style: { width: '20%' }
});

var midPanel = ui.SplitPanel({
    firstPanel: map,
    secondPanel: rightPanel,
    orientation: 'horizontal',
})

var mainPanel = ui.SplitPanel({
    firstPanel: leftPanel,
    secondPanel: ui.Panel(midPanel),
    orientation: 'horizontal'
})

ui.root.add(mainPanel);

function getDataset(dateEnd, dateComposite) {
    var d = ee.Date(dateEnd);
    var dateStart = d.advance(dateComposite, 'day').format('yyyy-MM-dd');

    var mdData = ee.ImageCollection('MODIS/061/MOD09GA')
        .filter(ee.Filter.date(dateStart, dateEnd))

    var mcdData = ee.ImageCollection('MODIS/061/MCD18A1')
        .filter(ee.Filter.date(dateStart, dateEnd))
        .select('GMT_0900_DSR');

    var firms = ee.ImageCollection("FIRMS")
        .filter(ee.Filter.date(dateStart, dateEnd))
        // .filter(ee.Filter.bounds(site))
        .select('T21');

    return { md: mdData, mcd: mcdData, firms: firms }
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

function showChart(mdCollection, bandArr, site) {
    var chartUi = ui.Chart.image.series({
        imageCollection: mdCollection.select(bandArr),
        region: site,
        reducer: ee.Reducer.mean(),
        scale: 500,
        xProperty: 'system:time_start'
    });

    rightPanel.clear()
    rightPanel.add(chartUi)
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

function showLegend(indexName, visPalette) {
    var legendTitle = ui.Label({
        value: indexName,
        style: { fontWeight: 'normal' }
    });

    var colorBar = ui.Thumbnail({
        image: ee.Image.pixelLonLat().select(0).int(),
        params: makeColorBarParams(visPalette.palette),
        style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px' },
    });

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

    rightPanel.add(legendTitle);
    rightPanel.add(colorBar);
    rightPanel.add(legendLabels);
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

    var visNdmi = {
        min: -1,
        max: 1,
        palette: ['DCF2F1', '7FC7D9', '365486', '0F1035']
    }

    var visSr = {
        min: 0,
        max: 500,
        palette: ['F3EDC8', 'EAD196', 'BF3131', '7D0A0A']
    }

    var visBiomass = {
        min: 0,
        max: 5,
        palette: ['43766C', 'F8FAE5', 'B19470', '76453B']
    }

    var visFirms = {
        min: 0,
        max: 1000,
        palette: ['red']
    }

    var visPolygonBorder = {
        color: 'red',
        width: 2,
    }

    var cbNdvi = chkbNdvi.getValue();
    var cbNdviDiff = chkbNdviDiff.getValue();
    var cbNdmi = chkbNdmi.getValue();
    var cbSr = chkbSr.getValue();
    var cbFpar = chkbFpar.getValue();
    var cbPar = chkbPar.getValue();
    var cbApar = chkbApar.getValue();
    var cbGpp = chkbGpp.getValue();
    var cbNpp = chkbNpp.getValue();

    var bandArr = [];

    map.clear()
    map.centerObject(site);

    if (cbNdvi) {
        map.addLayer(mdCollection.select('NDVI').median(), visPalette, "NDVI", true, 0.8);
        showLegend('cbNdvi', visPalette);
        bandArr.push('NDVI');
    }

    if (cbNdviDiff) {
        map.addLayer(mdCollection.select('NDVIdiff').median(), visPalette, "NDVIdiff", true, 0.8);
        showLegend('cbNdviDiff', visPalette);
        bandArr.push('NDVIdiff');
    }

    if (cbNdmi) {
        map.addLayer(mdCollection.select('NDMI').median(), visNdmi, "NDMI", true, 0.8);
        showLegend('cbNdmi', visNdmi);
        bandArr.push('NDMI');
    }

    if (cbFpar) {
        map.addLayer(mdCollection.select('FPAR').median(), visPalette, "FPAR", true, 0.8);
        showLegend('cbFpar', visPalette);
        bandArr.push('FPAR');
    }

    if (cbSr) {
        map.addLayer(mdCollection.select('GMT_0900_DSR').median(), visSr, "SR", true, 0.8);
        showLegend('cbSr', visSr);
        bandArr.push('GMT_0900_DSR');
    }

    if (cbPar) {
        map.addLayer(mdCollection.select('PAR').median(), visPalette, "PAR", true, 0.8);
        showLegend('cbPar', visPalette);
        bandArr.push('PAR');
    }

    if (cbApar) {
        map.addLayer(mdCollection.select('APAR').median(), visPalette, "APAR", true, 0.8);
        showLegend('cbApar', visPalette);
        bandArr.push('APAR');
    }

    if (cbGpp) {
        map.addLayer(mdCollection.select('GPP').median(), visBiomass, "GPP", true, 0.8);
        showLegend('cbGpp (Kg/m^2)', visBiomass);
        bandArr.push('GPP');
    }

    if (cbNpp) {
        map.addLayer(mdCollection.select('NPP').median(), visBiomass, "NPP ", true, 0.8);
        showLegend('cbNpp (Kg/m^2)', visBiomass);
        bandArr.push('NPP');
    }

    var siteLine = site.map(convertPolygonToLine);
    map.addLayer(siteLine, visPolygonBorder, "site", true);

    showChart(mdCollection, bandArr, site);
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

function loadData() {
    var dd = dateSliderUi.getValue();
    var dateEnd = ee.Date(dd[1]).format('YYYY-MM-dd');

    var dateComposite = dateCompositeUi.getValue() * -1;

    // get imageCollection
    var dataset = getDataset(dateEnd, dateComposite);

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

    showMap(mdCollection, dateEnd.getInfo());
    // var zStat = zonalStat(mdCollection, points, dateEnd);
}

var txtCloudSlideUi = ui.Label({
    value: 'เลือก % การปกคลุมของเมฆ',
    style: {
        margin: '4px 8px',
        fontSize: '18px',
        fontWeight: 1000
    }
});

leftPanel.add(txtCloudSlideUi);

var cloudSliderUi = ui.Slider({
    min: 0,
    max: 100,
    value: 50,
    style: { width: '90%' }
});
leftPanel.add(cloudSliderUi);

var txtDateUi = ui.Label({
    value: 'เลือกวันที่',
    style: {
        margin: '4px 8px',
        fontSize: '18px',
        fontWeight: 1000
    }
});
leftPanel.add(txtDateUi);

var dateSliderUi = ui.DateSlider({
    start: '2010-01-01',
    // end: '2023-12-31',
    value: '2023-11-15',
    style: { width: '80%' }
});
leftPanel.add(dateSliderUi);

var txtDateCompositeUi = ui.Label({
    value: 'เลือกจำนวนวันย้อนหลัง',
    style: {
        margin: '4px 8px',
        fontSize: '18px',
        fontWeight: 1000
    }
});
leftPanel.add(txtDateCompositeUi);

var dateItems = [
    { label: '3 วัน', value: 3 },
    { label: '5 วัน', value: 5 },
    { label: '7 วัน', value: 7 },
    { label: '15 วัน', value: 15 },
    { label: '30 วัน', value: 30 },
    { label: '60 วัน', value: 30 },
    { label: '120 วัน', value: 120 },
    { label: '180 วัน', value: 180 },
    { label: '360 วัน', value: 360 },
];
var dateCompositeUi = ui.Select({
    items: dateItems,
    value: 7,
    style: { width: '80%' }
});
leftPanel.add(dateCompositeUi);

var txtDateCompositeUi = ui.Label({
    value: 'เลือกชั้นข้อมูลที่ต้องการแสดงผล',
    style: {
        margin: '4px 8px',
        fontSize: '18px',
        fontWeight: 1000
    }
});
leftPanel.add(txtDateCompositeUi);

var chkbNdvi = ui.Checkbox({
    label: 'Normalized Difference Vegetation Index: NDVI',
    value: true
})
leftPanel.add(chkbNdvi);

var chkbNdviDiff = ui.Checkbox({
    label: 'NDVI diff',
    value: false
})
leftPanel.add(chkbNdviDiff);

var chkbNdmi = ui.Checkbox({
    label: 'Normalized Difference Moisture Index: NDMI',
    value: false
})
leftPanel.add(chkbNdmi);

var chkbFpar = ui.Checkbox({
    label: 'Fraction of Photosynthetically Active Radiation: FPAR',
    value: false
})
leftPanel.add(chkbFpar);

var chkbSr = ui.Checkbox({
    label: 'Surface Radiation: SR',
    value: false
})
leftPanel.add(chkbSr);

var chkbPar = ui.Checkbox({
    label: 'Photosynthetically Active Radiation: PAR',
    value: false
})
leftPanel.add(chkbPar);

var chkbApar = ui.Checkbox({
    label: 'Absorption Photosynthetically Active Radiation: APAR',
    value: false
})
leftPanel.add(chkbApar);

var chkbGpp = ui.Checkbox({
    label: 'Gross Primary Productivity: GPP',
    value: false
})
leftPanel.add(chkbGpp);

var chkbNpp = ui.Checkbox({
    label: 'Net Primary Productivity: NPP',
    value: true
})
leftPanel.add(chkbNpp);

cloudSliderUi.onChange(loadData);
dateSliderUi.onChange(loadData);
dateCompositeUi.onChange(loadData);

chkbNdvi.onChange(loadData);
chkbNdviDiff.onChange(loadData);
chkbNdmi.onChange(loadData);
chkbSr.onChange(loadData);
chkbFpar.onChange(loadData);
chkbPar.onChange(loadData);
chkbApar.onChange(loadData);
chkbGpp.onChange(loadData);
chkbNpp.onChange(loadData);

// field collection date
var dateArray = ['2023-11-15', '2023-11-20', '2023-11-25',
    '2023-11-30', '2023-12-05', '2023-12-10',
    '2023-12-15', '2023-12-20', '2023-12-25',
    '2023-12-30', '2024-01-05']

loadData();

// dateArray.forEach(function (i) {
//     init(i);
// });

