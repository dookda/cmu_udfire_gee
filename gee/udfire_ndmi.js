Map.setOptions('SATELLITE');
// 1. Define study areas (can be customized)
var ud = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/paktab");
var mt = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/meatha_n");
var vs = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/winagsa");
// var ms = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/measariang");

var provinceName = 'Mae Hong Son';

// --- โหลดขอบเขตอำเภอ (ADM2) ของประเทศไทย ---
var gaulAdm2 = ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM0_NAME', 'Thailand'))
    .filter(ee.Filter.eq('ADM1_NAME', provinceName));

var ms = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Mae Sariang'));
var ky = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Khun Yuam'));
var hs = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Muang Mae Hong Son'));
var mln = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Mae La Noi'));
var pth = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Pai'));
var sng = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Sop Moei'));
var pmp = gaulAdm2.filter(ee.Filter.eq('ADM2_NAME', 'Pang Mapha'));


var studyAreas = {
    'ปากทับ จ.อุตรดิตถ์': ud,
    'แม่ทา จ.เชียงใหม่': mt,
    'อ.เวียงสา จ.น่าน': vs,
    'อ.แม่สะเรียง แม่ฮ่องสอน': ms,
    'อ.เมือง แม่ฮ่องสอน': hs,
    'อ.แม่ลาน้อย แม่ฮ่องสอน': mln,
    'อ.ปาย แม่ฮ่องสอน': pth,
    'อ.สบเมย แม่ฮ่องสอน': sng,
    'อ.ปางมะผ้า แม่ฮ่องสอน': pmp,
    'อ.ขุนยวม แม่ฮ่องสอน': ky
};

// 2. NDMI calculation function
function getNDMI(image) {
    return image.normalizedDifference(['B8', 'B11']).rename('NDMI');
}

// 3. Create UI widgets
var areaSelect = ui.Select({
    items: Object.keys(studyAreas),
    placeholder: 'เลือกพื้นที่ศึกษา',
    value: 'ปากทับ จ.อุตรดิตถ์',
    onChange: function (selected) {
        var region = studyAreas[selected];
        Map.centerObject(region, 8);
        var selectedRegion = region;
    }
});

var startDateInput = ui.Textbox({ placeholder: 'YYYY-MM-DD', value: '2022-01-01' });
var endDateInput = ui.Textbox({ placeholder: 'YYYY-MM-DD', value: '2022-01-31' });

var applyButton = ui.Button({
    label: 'คำนวณ NDMI',
    style: { stretch: 'horizontal' },
    onClick: function () {
        var startDate = startDateInput.getValue();
        var endDate = endDateInput.getValue();
        var selectedArea = areaSelect.getValue();

        if (!startDate || !endDate || !selectedArea) {
            ui.Label('Please select area and dates');
            return;
        }

        Map.clear();
        generateNDMI(startDate, endDate, studyAreas[selectedArea]);
        Map.setOptions('SATELLITE');
    }
});

// 4. Add widgets to the UI panel
var panel = ui.Panel({
    widgets: [
        ui.Label('ดัชนีความแตกต่างของความชื้น (NDMI)', { fontWeight: 'bold', fontSize: '20px' }),
        ui.Label('NDMI (Normalized Difference Moisture Index) เป็นดัชนีที่ใช้ประเมินปริมาณน้ำในพืช (vegetation water content) โดยใช้ข้อมูลจากแถบคลื่นอินฟราเรดใกล้ (NIR) และคลื่นอินฟราเรดช่วงสั้น (SWIR) (การศึกษานี้ใช้ข้อมูลจากดาวเทียม Sentinel-2) ค่า NDMI จะอยู่ระหว่าง -1 ถึง 1 โดยค่าสูงบ่งชี้ปริมาณน้ำในพืชสูง และค่าต่ำบ่งชี้ถึงความแห้งแล้งหรือความเครียดจากน้ำในพืช', { fontSize: '12px' }),
        ui.Label('1. เลือกพื้นที่ศึกษา:', { fontWeight: 'bold' }), areaSelect,
        ui.Label('2. เลือกช่วงเวลาที่ต้องการคำนวณ:', { fontWeight: 'bold' }),
        ui.Label('วันที่เริ่มต้น (ปี ค.ศ.-เดือน-วัน)'), startDateInput,
        ui.Label('วันที่สิ้นสุด (ปี ค.ศ.-เดือน-วัน)'), endDateInput,
        ui.Label(''),
        ui.Label('3. กดปุ่ม "คำนวณ NDMI" เพื่อแสดงผลลัพธ์บนแผนที่', { fontWeight: 'bold' }),
        applyButton
    ],
    style: { width: '300px' }
});
ui.root.insert(0, panel);

// 5. Add legend
var colors = ['#e66101', '#fdb863', '#f7f7f7', '#b2abd2', '#5e3c99'];
function addNDMILegend() {
    var legend = ui.Panel({
        style: {
            position: 'bottom-right',
            padding: '8px 15px',
            backgroundColor: 'white',
            borderRadius: '8px',
        }
    });

    // Title
    var title = ui.Label({
        value: 'NDMI Legend',
        style: { fontWeight: 'bold', fontSize: '14px', margin: '0 0 4px 0' }
    });

    // Create color bar
    var makeColorBar = function (palette) {
        return ui.Thumbnail({
            image: ee.Image.pixelLonLat().select(0),
            params: {
                bbox: [0, 0, 1, 0.1],
                dimensions: '100x10',
                format: 'png',
                min: 0,
                max: 1,
                palette: palette
            },
            style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '20px' }
        });
    };

    var colorBar = makeColorBar(colors);

    // Create labels
    var legendLabels = ui.Panel({
        layout: ui.Panel.Layout.flow('horizontal'),
        widgets: [
            ui.Label('-1 (แห้งแล้ง)', { margin: '4px 8px' }),
            ui.Label('0 (ปกติ)', { margin: '4px 28px' }),
            ui.Label('+1 (ชื้น)', { margin: '4px 8px' })
        ]
    });

    // Add all to the legend panel
    legend.add(title);
    legend.add(colorBar);
    legend.add(legendLabels);

    // Add to map
    Map.add(legend);
}

// 5. Function to compute and display NDMI
function generateNDMI(startDate, endDate, region) {
    var collection = ee.ImageCollection('COPERNICUS/S2_SR')
        .filterDate(startDate, endDate)
        .filterBounds(region)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(function (img) {
            return getNDMI(img).copyProperties(img, ['system:time_start']);
        });

    var ndmiComposite = collection.mean().clip(region);

    var visParams = {
        min: -0.3,
        max: 0.5,
        palette: colors
    };

    Map.centerObject(region)
    Map.addLayer(ndmiComposite, visParams, 'NDMI', true, 1.0);

    var border = region.style({
        color: 'red',  // Red border
        width: 2,  // Border width
        fillColor: '00000000'  // Transparent fill
    });

    Map.addLayer(border, {}, 'ขอบเขตตำบล', true);
    addNDMILegend();

    // Export NDMI as PNG
    // Export.image.toDrive({
    //   image: ndmiComposite,
    //   description: 'NDMI_' + startDate + '_to_' + endDate,
    //   folder: 'GEE_NDMI',
    //   region: region,
    //   scale: 10,
    //   fileFormat: 'PNG',
    //   maxPixels: 1e13
    // });

    print('Export task created. Open the "Tasks" tab to start the download.');
}

var startDate = startDateInput.getValue();
var endDate = endDateInput.getValue();
var selectedArea = areaSelect.getValue();

if (!startDate || !endDate || !selectedArea) {
    ui.Label('Please select area and dates');
}

generateNDMI(startDate, endDate, studyAreas[selectedArea]);
