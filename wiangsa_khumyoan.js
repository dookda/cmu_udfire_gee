// -----------------------------
// 1. Load AOIs and LULC base image
// -----------------------------
var aoiDict = {
    'เวียงสา': ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/winagsa"),
    'ขุนยวม': ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/khunyoam") // สมมุติชื่อแตกต่างกัน
};

var lulcBase = ee.Image('ESA/WorldCover/v200/2021').select('Map');

var lulcVisParams = {
    min: 10,
    max: 95,
    palette: [
        '006400', 'ffbb22', 'ffff4c', 'f096ff',
        'fa0000', 'b4b4b4', '0064c8', '0096a0'
    ]
};

// -----------------------------
// 2. Map Panel
// -----------------------------
var mapPanel = ui.Map({
    style: {
        border: '1px solid black',
        borderRadius: '10px',
    }
});
mapPanel.setOptions('SATELLITE');

// -----------------------------
// 3. Legend Panel (bottom-left of map)
// -----------------------------
var legend = ui.Panel({
    style: {
        position: 'bottom-left',
        padding: '8px 15px',
        backgroundColor: 'rgba(255,255,255,0.85)',
        border: '1px solid black',
        borderRadius: '10px',
        maxWidth: '200px'
    }
});

legend.add(ui.Label({
    value: 'คำอธิบาย LULC และ Hotspot',
    style: { fontWeight: 'bold', fontSize: '16px', margin: '0 0 8px 0' }
}));

[
    { name: 'ป่าไม้', color: '006400' },
    { name: 'พุ่มไม้', color: 'ffbb22' },
    { name: 'ทุ่งหญ้า', color: 'ffff4c' },
    { name: 'พื้นที่เกษตรกรรม', color: 'f096ff' },
    { name: 'พื้นที่เมือง', color: 'fa0000' },
    { name: 'พื้นที่ดินโล่ง', color: 'b4b4b4' },
    { name: 'แหล่งน้ำ', color: '0064c8' },
    { name: 'พื้นที่ชุ่มน้ำ', color: '0096a0' },
    { name: 'Hotspot (FIRMS)', color: 'ff0000' }
].forEach(function (c) {
    var box = ui.Label({
        style: {
            backgroundColor: '#' + c.color,
            padding: '8px',
            margin: '0 0 4px 0',
            border: '1px solid #000'
        }
    });
    var label = ui.Label({
        value: c.name,
        style: { margin: '0 0 4px 6px' }
    });
    legend.add(ui.Panel([box, label], ui.Panel.Layout.Flow('horizontal')));
});
mapPanel.add(legend);

// -----------------------------
// 4. Control Panel (left)
// -----------------------------
var controlPanel = ui.Panel({
    style: {
        padding: '8px 15px',
        backgroundColor: 'rgba(255,255,255,0.85)',
        border: '1px solid black',
        borderRadius: '10px',
        width: '20%'
    }
});

// AOI select
var aoiSelect = ui.Select({
    items: Object.keys(aoiDict),
    value: 'เวียงสา',
    style: { width: '100%' },
    onChange: function () {
        updateMap();
    }
});
controlPanel.add(ui.Label('เลือกพื้นที่ศึกษา:', { margin: '0 0 4px 0' }));
controlPanel.add(aoiSelect);

// Year select
var yearSelect = ui.Select({
    items: ['2020', '2021', '2022', '2023', '2024'],
    value: '2024',
    style: { width: '100%' },
    onChange: function () {
        updateMap();
    }
});
controlPanel.add(ui.Label('เลือกปี (FIRMS):', { margin: '10px 0 4px 0' }));
controlPanel.add(yearSelect);

// -----------------------------
// 5. Update Function
// -----------------------------
var lulcLayer, hotspotLayer, outlineLayer;

function updateMap() {
    var selectedAOI = aoiDict[aoiSelect.getValue()];
    var selectedYear = parseInt(yearSelect.getValue());

    // ล้างชั้นเก่า
    if (lulcLayer) mapPanel.layers().remove(lulcLayer);
    if (hotspotLayer) mapPanel.layers().remove(hotspotLayer);
    if (outlineLayer) mapPanel.layers().remove(outlineLayer);

    // แสดงขอบเขต AOI
    var outline = selectedAOI.style({
        color: 'yellow',
        width: 2,
        fillColor: '00000000'  // transparent
    });
    outlineLayer = ui.Map.Layer(outline, {}, 'ขอบเขตพื้นที่ศึกษา');
    mapPanel.layers().add(outlineLayer);

    // เพิ่ม LULC
    var lulc = lulcBase.clip(selectedAOI);
    lulcLayer = ui.Map.Layer(lulc, lulcVisParams, 'LULC');
    mapPanel.layers().add(lulcLayer);

    // เพิ่ม Hotspot FIRMS
    var start = ee.Date.fromYMD(selectedYear, 1, 1);
    var end = ee.Date.fromYMD(selectedYear, 12, 31);
    var firms = ee.ImageCollection('FIRMS')
        .filterDate(start, end)
        .filterBounds(selectedAOI)
        .select('T21')
        .mean()
        .clip(selectedAOI);
    hotspotLayer = ui.Map.Layer(firms, { palette: ['red'] }, 'Hotspot FIRMS ' + selectedYear);
    mapPanel.layers().add(hotspotLayer);

    // ขยายแผนที่ไปยัง AOI
    mapPanel.centerObject(selectedAOI, 9);
}


// Initial load
updateMap();

// -----------------------------
// 6. Combine Panels
// -----------------------------
var mainPanel = ui.SplitPanel({
    firstPanel: controlPanel,
    secondPanel: mapPanel,
    orientation: 'horizontal',
    wipe: false,
    style: { stretch: 'both' }
});

ui.root.clear();
ui.root.add(mainPanel);
