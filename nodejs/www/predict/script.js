// MapLibre GL JS Map Configuration
let map;
let hexagonLayerLoaded = false;
let hotspotLayerLoaded = false;

// CartoDB Basemap URLs
const basemaps = {
    light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
};

// Initialize the map
function initMap() {
    map = new maplibregl.Map({
        container: 'map',
        style: {
            version: 8,
            sources: {
                'carto-light': {
                    type: 'raster',
                    tiles: [
                        'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                        'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                        'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                        'https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
                    ],
                    tileSize: 256,
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                },
                'carto-dark': {
                    type: 'raster',
                    tiles: [
                        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                        'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                    ],
                    tileSize: 256,
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                }
            },
            layers: [
                {
                    id: 'carto-light-layer',
                    type: 'raster',
                    source: 'carto-light',
                    layout: {
                        visibility: 'visible'
                    }
                },
                {
                    id: 'carto-dark-layer',
                    type: 'raster',
                    source: 'carto-dark',
                    layout: {
                        visibility: 'none'
                    }
                }
            ]
        },
        center: [100.0, 18.5], // Thailand center coordinates
        zoom: 7,
        maxZoom: 18,
        minZoom: 3,
        pitch: 45,
        projection: 'globe'
    });

    // Add navigation controls
    map.addControl(new maplibregl.NavigationControl(), 'bottom-left');

    // Add scale control
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left');

    // Add fullscreen control
    map.addControl(new maplibregl.FullscreenControl(), 'bottom-left');

    // Map event listeners
    map.on('load', onMapLoad);
    map.on('move', updateMapInfo);
    map.on('zoom', updateMapInfo);

    map.on('style.load', () => {
        map.setProjection({ type: 'globe' });
    });
}

// Map load event handler
function onMapLoad() {
    console.log('Map loaded successfully');

    // Load hexagon layer
    loadHexagonLayer();

    // Load FIRMS hotspot layer
    loadHotspotLayer();

    // Hide loading overlay
    hideLoading();

    // Update map info
    updateMapInfo();
}

// Load hexagon GeoJSON layer
async function loadHexagonLayer() {
    try {
        console.log('Loading hexagon layer...');

        // Fetch and process the GeoJSON data
        const response = await fetch('./hex_forest_pro_4326_predict.geojson');
        const geojsonData = await response.json();

        // Process features to add simplified prediction properties
        geojsonData.features.forEach(feature => {
            if (feature.properties.predictions) {
                let predictions;
                try {
                    // Handle both string and object predictions
                    predictions = typeof feature.properties.predictions === 'string'
                        ? JSON.parse(feature.properties.predictions)
                        : feature.properties.predictions;
                } catch (e) {
                    predictions = feature.properties.predictions;
                }

                // Add simplified properties for each month
                if (Array.isArray(predictions)) {
                    predictions.forEach(pred => {
                        const monthKey = `pred_${pred.date.replace(/-/g, '_')}`;
                        feature.properties[monthKey] = pred.predicted_hotspot_count;
                    });
                }
            }
        });

        // Add the hexagon source with processed data
        map.addSource('hexagon-source', {
            type: 'geojson',
            data: geojsonData
        });

        // Add hexagon fill layer
        map.addLayer({
            id: 'hexagon-fill',
            type: 'fill',
            source: 'hexagon-source',
            paint: {
                'fill-color': [
                    'case',
                    ['has', 'Shape_Area'],
                    [
                        'interpolate',
                        ['linear'],
                        ['get', 'Shape_Area'],
                        0, '#feedde',
                        1000, '#fdd49e',
                        5000, '#fdbb84',
                        10000, '#fc8d59',
                        20000, '#ef6548',
                        50000, '#d7301f',
                        100000, '#990000'
                    ],
                    '#cccccc'
                ],
                'fill-opacity': 0.7,
                'fill-outline-color': '#ffffff'
            }
        });

        // Add hexagon border layer
        map.addLayer({
            id: 'hexagon-border',
            type: 'line',
            source: 'hexagon-source',
            paint: {
                'line-color': '#ffffff',
                'line-width': [
                    'interpolate',
                    ['linear'],
                    ['zoom'],
                    5, 0.5,
                    10, 1,
                    15, 2
                ],
                'line-opacity': 0.8
            }
        });

        // Add click event for hexagons
        map.on('click', 'hexagon-fill', onHexagonClick);

        // Change cursor on hover
        map.on('mouseenter', 'hexagon-fill', () => {
            map.getCanvas().style.cursor = 'pointer';
        });

        map.on('mouseleave', 'hexagon-fill', () => {
            map.getCanvas().style.cursor = '';
        });

        hexagonLayerLoaded = true;
        console.log('Hexagon layer loaded successfully');

    } catch (error) {
        console.error('Error loading hexagon layer:', error);
        showNotification('Error loading hexagon layer', 'error');
    }
}

// Load FIRMS hotspot MVT layer
async function loadHotspotLayer() {
    try {
        console.log('Loading FIRMS thermal anomalies from WFS...');

        // Use FIRMS WFS GeoJSON endpoint for Southeast Asia (24 hours)
        const firmsUrl = 'https://firms.modaps.eosdis.nasa.gov/mapserver/wfs/SouthEast_Asia/7a16aa667fe01b181ffebcf83c022e34/?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0&TYPENAME=ms:fires_modis_24hrs&STARTINDEX=0&COUNT=1000&SRSNAME=urn:ogc:def:crs:EPSG::4326&BBOX=-90,-180,90,180,urn:ogc:def:crs:EPSG::4326&outputformat=geojson';

        console.log('Fetching FIRMS GeoJSON data...');
        const response = await fetch(firmsUrl);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const geojsonData = await response.json();

        console.log(`Loaded ${geojsonData.features?.length || 0} thermal anomalies from FIRMS WFS`);

        // Add source
        map.addSource('viirs-hotspots', {
            type: 'geojson',
            data: geojsonData
        });

        // Add hotspot circle layer
        map.addLayer({
            id: 'hotspot-points',
            type: 'circle',
            source: 'viirs-hotspots',
            paint: {
                'circle-radius': [
                    'interpolate',
                    ['linear'],
                    ['zoom'],
                    5, 2,
                    10, 5,
                    15, 8
                ],
                'circle-color': [
                    'case',
                    ['has', 'confidence'],
                    [
                        'interpolate',
                        ['linear'],
                        ['to-number', ['get', 'confidence']],
                        0, '#ffeb3b',    // Low confidence - yellow
                        50, '#ff9800',   // Medium confidence - orange  
                        80, '#f44336',   // High confidence - red
                        100, '#b71c1c'   // Very high confidence - dark red
                    ],
                    '#ff4444' // Default red for FIRMS data
                ],
                'circle-opacity': 0.9,
                'circle-stroke-width': 0.8,
                'circle-stroke-color': '#fff',
                'circle-stroke-opacity': 0.8
            }
        });

        // Add click event for hotspots
        map.on('click', 'hotspot-points', onHotspotClick);

        // Change cursor on hover
        map.on('mouseenter', 'hotspot-points', () => {
            map.getCanvas().style.cursor = 'pointer';
        });

        map.on('mouseleave', 'hotspot-points', () => {
            map.getCanvas().style.cursor = '';
        });

        hotspotLayerLoaded = true;
        console.log(`FIRMS thermal anomalies loaded successfully: ${geojsonData.features?.length || 0} points`);
        showNotification(`${geojsonData.features?.length || 0} thermal anomalies loaded successfully`, 'info');

    } catch (error) {
        console.error('Error loading VIIRS thermal anomalies:', error);
        showNotification('Error loading thermal anomalies. Using fallback...', 'error');

        // Fallback to alternative source
        loadHotspotLayerFallback();
    }
}

// Fallback hotspot layer using GeoJSON from FIRMS
async function loadHotspotLayerFallback() {
    try {
        console.log('Loading MODIS hotspot data (fallback)...');

        // Use FIRMS API for MODIS data (last 24 hours)
        const firmsUrl = 'https://firms.modaps.eosdis.nasa.gov/api/country/json/7a16aa667fe01b181ffebcf83c022e34/MODIS_NRT/THA/1';

        const response = await fetch(firmsUrl);
        const data = await response.json();

        console.log(`Fallback: Found ${data.length} MODIS hotspots`);

        // Convert to GeoJSON
        const geojsonData = {
            type: 'FeatureCollection',
            features: data.map(hotspot => ({
                type: 'Feature',
                geometry: {
                    type: 'Point',
                    coordinates: [parseFloat(hotspot.longitude), parseFloat(hotspot.latitude)]
                },
                properties: {
                    confidence: hotspot.confidence,
                    bright_ti4: hotspot.bright_ti4,
                    bright_ti5: hotspot.bright_ti5,
                    scan: hotspot.scan,
                    track: hotspot.track,
                    acq_date: hotspot.acq_date,
                    acq_time: hotspot.acq_time,
                    satellite: hotspot.satellite,
                    instrument: hotspot.instrument,
                    version: hotspot.version
                }
            }))
        };

        // Update the existing source instead of creating a new one
        if (map.getSource('viirs-hotspots')) {
            map.getSource('viirs-hotspots').setData(geojsonData);
        } else {
            map.addSource('viirs-hotspots', {
                type: 'geojson',
                data: geojsonData
            });

            // Add the layers if they don't exist
            if (!map.getLayer('hotspot-points')) {
                map.addLayer({
                    id: 'hotspot-points',
                    type: 'circle',
                    source: 'viirs-hotspots',
                    paint: {
                        'circle-radius': [
                            'interpolate',
                            ['linear'],
                            ['zoom'],
                            5, 1.5,
                            10, 3,
                            15, 6
                        ],
                        'circle-color': '#ff4444',
                        'circle-opacity': 0.9,
                        'circle-stroke-width': 1,
                        'circle-stroke-color': '#fff',
                        'circle-stroke-opacity': 0.8
                    }
                });

                // Add events
                map.on('click', 'hotspot-points', onHotspotClick);
                map.on('mouseenter', 'hotspot-points', () => {
                    map.getCanvas().style.cursor = 'pointer';
                });
                map.on('mouseleave', 'hotspot-points', () => {
                    map.getCanvas().style.cursor = '';
                });
            }
        }

        hotspotLayerLoaded = true;
        console.log(`MODIS hotspot fallback loaded: ${data.length} points`);
        showNotification(`${data.length} MODIS hotspots loaded (fallback)`, 'info');

    } catch (error) {
        console.error('Error loading hotspot fallback:', error);
        showNotification('Error loading thermal anomalies', 'error');
    }
}

// Update hexagon colors based on selected month
function updateHexagonColors(selectedMonth) {
    if (!map.getLayer('hexagon-fill')) return;

    let fillColorExpression;

    if (!selectedMonth) {
        // Default coloring based on Shape_Area
        fillColorExpression = [
            'case',
            ['has', 'Shape_Area'],
            [
                'interpolate',
                ['linear'],
                ['get', 'Shape_Area'],
                0, '#feedde',
                1000, '#fdd49e',
                5000, '#fdbb84',
                10000, '#fc8d59',
                20000, '#ef6548',
                50000, '#d7301f',
                100000, '#990000'
            ],
            '#cccccc'
        ];
    } else {
        // Color based on predictions for selected month
        fillColorExpression = [
            'case',
            // Check if predictions exist and extract value for selected month
            [
                'any',
                [
                    'all',
                    ['has', 'predictions'],
                    [
                        '>',
                        [
                            'length',
                            [
                                'filter',
                                ['literal', JSON.parse('[]')], // This will be dynamically built
                                ['==', ['get', 'date'], selectedMonth]
                            ]
                        ],
                        0
                    ]
                ]
            ],
            // Color interpolation based on predicted hotspot count
            [
                'interpolate',
                ['linear'],
                // Extract prediction value for the selected month
                [
                    'get',
                    'predicted_hotspot_count',
                    [
                        'at',
                        0,
                        [
                            'filter',
                            ['get', 'predictions'],
                            ['==', ['get', 'date'], selectedMonth]
                        ]
                    ]
                ],
                0, '#d7f4d7',    // Very low (green)
                10, '#b8e6b8',   // Low
                25, '#ffe066',   // Medium-low (yellow)
                50, '#ffb366',   // Medium (orange)
                75, '#ff8566',   // Medium-high
                100, '#ff5566',  // High (red)
                150, '#cc0000',  // Very high (dark red)
                200, '#990000'   // Extreme (very dark red)
            ],
            '#cccccc' // Default gray for no predictions
        ];

        // Since MapLibre expressions have limitations with complex JSON parsing,
        // let's use a simpler approach with multiple case statements
        fillColorExpression = [
            'case',
            ['has', 'predictions'],
            [
                'interpolate',
                ['linear'],
                // Use a custom property that we'll set when processing the data
                ['coalesce', ['get', `pred_${selectedMonth.replace(/-/g, '_')}`], 0],
                0, '#d7f4d7',    // Very low (green)
                10, '#b8e6b8',   // Low
                25, '#ffe066',   // Medium-low (yellow)
                50, '#ffb366',   // Medium (orange)
                75, '#ff8566',   // Medium-high
                100, '#ff5566',  // High (red)
                150, '#cc0000',  // Very high (dark red)
                200, '#990000'   // Extreme (very dark red)
            ],
            '#cccccc' // Default gray for no predictions
        ];
    }

    // Update the layer paint property
    map.setPaintProperty('hexagon-fill', 'fill-color', fillColorExpression);

    console.log(`Updated hexagon colors for month: ${selectedMonth || 'default'}`);
}

// Hexagon click event handler
function onHexagonClick(e) {
    const features = map.queryRenderedFeatures(e.point, {
        layers: ['hexagon-fill']
    });

    if (features.length > 0) {
        const feature = features[0];
        const properties = feature.properties;

        // Create popup content
        let popupContent = '<div class="popup-content">';
        popupContent += '<h4>🌲 Forest Hexagon Prediction</h4>';

        if (properties.PROV_NAM_E) {
            popupContent += `<p><strong>Province:</strong> ${properties.PROV_NAM_E}</p>`;
        }

        if (properties.PROV_NAM_T) {
            popupContent += `<p><strong>จังหวัด:</strong> ${properties.PROV_NAM_T}</p>`;
        }

        if (properties.Shape_Area) {
            popupContent += `<p><strong>Shape Area:</strong> ${properties.Shape_Area.toLocaleString()} sq meters</p>`;
        }

        if (properties.id) {
            popupContent += `<p><strong>Hexagon ID:</strong> ${properties.id}</p>`;
        }

        // Display predictions if available
        if (properties.predictions) {
            let predictions;
            try {
                // Handle both string and object predictions
                predictions = typeof properties.predictions === 'string'
                    ? JSON.parse(properties.predictions)
                    : properties.predictions;
            } catch (e) {
                predictions = properties.predictions;
            }

            if (Array.isArray(predictions) && predictions.length > 0) {
                popupContent += '<hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">';
                popupContent += '<h5>🔥 Hotspot Predictions 2026</h5>';
                popupContent += '<div style="max-height: 200px; overflow-y: auto;">';

                predictions.forEach(pred => {
                    const date = new Date(pred.date).toLocaleDateString('th-TH', {
                        year: 'numeric',
                        month: 'long'
                    });
                    const count = Math.round(pred.predicted_hotspot_count);
                    const intensity = count > 100 ? '🔴' : count > 50 ? '🟠' : count > 20 ? '🟡' : '🟢';
                    popupContent += `<p style="margin: 5px 0;">${intensity} <strong>${date}:</strong> ${count} hotspots</p>`;
                });

                popupContent += '</div>';

                // Add summary statistics
                const totalPredicted = predictions.reduce((sum, p) => sum + p.predicted_hotspot_count, 0);
                const avgMonthly = Math.round(totalPredicted / predictions.length);
                const maxMonth = predictions.reduce((max, p) => p.predicted_hotspot_count > max.predicted_hotspot_count ? p : max);

                popupContent += '<hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">';
                popupContent += '<h6>📊 Summary</h6>';
                popupContent += `<p><strong>Total Year:</strong> ${Math.round(totalPredicted)} hotspots</p>`;
                popupContent += `<p><strong>Monthly Avg:</strong> ${avgMonthly} hotspots</p>`;
                popupContent += `<p><strong>Peak Month:</strong> ${new Date(maxMonth.date).toLocaleDateString('th-TH', { month: 'long' })} (${Math.round(maxMonth.predicted_hotspot_count)})</p>`;
            }
        }

        // Add other key properties (excluding predictions which we handled above)
        Object.keys(properties).forEach(key => {
            if (!['Shape_Area', 'id', 'PROV_NAM_E', 'PROV_NAM_T', 'geometry', 'predictions'].includes(key) && properties[key] !== null && properties[key] !== undefined) {
                const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                popupContent += `<p><strong>${displayKey}:</strong> ${properties[key]}</p>`;
            }
        });

        popupContent += '</div>';

        // Show popup
        new maplibregl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(popupContent)
            .addTo(map);
    }
}

// Hotspot click event handler
function onHotspotClick(e) {
    const features = map.queryRenderedFeatures(e.point, {
        layers: ['hotspot-points']
    });

    if (features.length > 0) {
        const feature = features[0];
        const properties = feature.properties;

        // Create popup content
        let popupContent = '<div class="popup-content hotspot-popup">';
        popupContent += '<h4>🔥 FIRMS Fire Hotspot</h4>';

        if (properties.confidence) {
            const confidence = parseFloat(properties.confidence);
            let confidenceClass = '';
            let confidenceText = '';

            if (confidence >= 80) {
                confidenceClass = 'high-confidence';
                confidenceText = 'High';
            } else if (confidence >= 50) {
                confidenceClass = 'medium-confidence';
                confidenceText = 'Medium';
            } else {
                confidenceClass = 'low-confidence';
                confidenceText = 'Low';
            }

            popupContent += `<p><strong>Confidence:</strong> <span class="${confidenceClass}">${confidence}% (${confidenceText})</span></p>`;
        }

        if (properties.bright_ti4) {
            popupContent += `<p><strong>Brightness (4μm):</strong> ${properties.bright_ti4}K</p>`;
        }

        if (properties.bright_ti5) {
            popupContent += `<p><strong>Brightness (11μm):</strong> ${properties.bright_ti5}K</p>`;
        }

        if (properties.acq_date) {
            popupContent += `<p><strong>Date:</strong> ${properties.acq_date}</p>`;
        }

        if (properties.acq_time) {
            const time = properties.acq_time.toString().padStart(4, '0');
            const formattedTime = time.slice(0, 2) + ':' + time.slice(2);
            popupContent += `<p><strong>Time:</strong> ${formattedTime} UTC</p>`;
        }

        if (properties.satellite) {
            popupContent += `<p><strong>Satellite:</strong> ${properties.satellite}</p>`;
        }

        if (properties.instrument) {
            popupContent += `<p><strong>Instrument:</strong> ${properties.instrument}</p>`;
        }

        if (properties.scan && properties.track) {
            popupContent += `<p><strong>Scan/Track:</strong> ${properties.scan}/${properties.track}</p>`;
        }

        popupContent += '</div>';

        // Show popup
        new maplibregl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(popupContent)
            .addTo(map);
    }
}

// Basemap toggle functionality
function setupBasemapToggle() {
    const toggleSwitch = document.getElementById('basemap-toggle');

    toggleSwitch.addEventListener('change', function () {
        if (this.checked) {
            // Switch to dark basemap
            map.setLayoutProperty('carto-light-layer', 'visibility', 'none');
            map.setLayoutProperty('carto-dark-layer', 'visibility', 'visible');
        } else {
            // Switch to light basemap
            map.setLayoutProperty('carto-light-layer', 'visibility', 'visible');
            map.setLayoutProperty('carto-dark-layer', 'visibility', 'none');
        }
    });
}

// Layer toggle functionality
function setupLayerToggle() {
    const hexagonToggle = document.getElementById('hexagon-layer');
    const hotspotToggle = document.getElementById('hotspot-layer');
    const monthSelector = document.getElementById('month-selector');

    hexagonToggle.addEventListener('change', function () {
        if (hexagonLayerLoaded) {
            const visibility = this.checked ? 'visible' : 'none';
            map.setLayoutProperty('hexagon-fill', 'visibility', visibility);
            map.setLayoutProperty('hexagon-border', 'visibility', visibility);
        }
    });

    hotspotToggle.addEventListener('change', function () {
        if (hotspotLayerLoaded) {
            const visibility = this.checked ? 'visible' : 'none';
            map.setLayoutProperty('hotspot-points', 'visibility', visibility);
        }
    });

    // Month selector for dynamic coloring
    monthSelector.addEventListener('change', function () {
        if (hexagonLayerLoaded) {
            updateHexagonColors(this.value);
        }
    });
}

// Update map information display
function updateMapInfo() {
    const center = map.getCenter();
    const zoom = map.getZoom();

    document.getElementById('zoom-level').textContent = zoom.toFixed(1);
    document.getElementById('map-center').textContent =
        `${center.lng.toFixed(3)}, ${center.lat.toFixed(3)}`;
}

// Show/hide loading overlay
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    setTimeout(() => {
        document.getElementById('loading').classList.add('hidden');
    }, 1000);
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;

    // Add styles for notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'error' ? '#ff4444' : '#2196F3'};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: slideInTop 0.3s ease;
    `;

    // Add notification to body
    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Responsive map resize
function handleResize() {
    if (map) {
        map.resize();
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM loaded, initializing map...');

    // Show loading
    showLoading();

    // Initialize map
    initMap();

    // Setup controls
    setupBasemapToggle();
    setupLayerToggle();

    // Handle window resize
    window.addEventListener('resize', handleResize);

    // Add CSS animations
    addCustomStyles();
});

// Add custom CSS animations and styles
function addCustomStyles() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInTop {
            from {
                transform: translate(-50%, -100%);
                opacity: 0;
            }
            to {
                transform: translate(-50%, 0);
                opacity: 1;
            }
        }
        
        .popup-content {
            font-family: 'Prompt', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .popup-content h4 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 16px;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 5px;
        }
        
        .popup-content p {
            margin: 5px 0;
            color: #666;
        }
        
        .popup-content strong {
            color: #333;
        }
        
        .notification button {
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: background-color 0.3s ease;
        }
        
        .notification button:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Hotspot popup specific styles */
        .hotspot-popup h4 {
            border-bottom: 2px solid #ff4444 !important;
        }
        
        .high-confidence {
            color: #b71c1c;
            font-weight: bold;
        }
        
        .medium-confidence {
            color: #ff9800;
            font-weight: bold;
        }
        
        .low-confidence {
            color: #ffeb3b;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }
    `;
    document.head.appendChild(style);
}

// Error handling for map
window.addEventListener('error', function (e) {
    console.error('Global error:', e.error);
    if (e.error && e.error.message && e.error.message.includes('map')) {
        showNotification('Map initialization error. Please refresh the page.', 'error');
    }
});

// Export functions for potential external use
window.mapApp = {
    map: () => map,
    showNotification,
    updateMapInfo,
    handleResize
};