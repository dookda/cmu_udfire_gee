import ee
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

ee.Authenticate()
try:
    ee.Initialize(project="ee-sakda-451407")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-sakda-451407")

study_area = ee.FeatureCollection("projects/ee-sakda-451407/assets/fire/khunyoam_sub") \
               .geometry().bounds()
date_start = ee.Date('2020-01-01')
date_end = ee.Date('2024-12-31')

# 1) Hotspot points as before
firms = ee.ImageCollection('FIRMS')\
    .select('confidence')\
    .filterDate(date_start, date_end)\
    .filterBounds(study_area)


def create_fire_points(image):
    fire_mask = image.select('confidence').gt(80)\
        .set('system:time_start', image.get('system:time_start'))
    vectors = fire_mask.reduceToVectors(
        geometry=study_area,
        scale=375,
        geometryType='centroid',
        labelProperty='fire',
        maxPixels=1e9
    ).filterBounds(study_area)
    return vectors.map(lambda feature: feature.set('system:time_start', image.get('system:time_start')))


fire_points = firms.map(create_fire_points)
fire_pts = fire_points.flatten()

# 2) CHIRPS daily rainfall (unchanged)
rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')\
    .select('precipitation')\
    .filterDate(date_start, date_end)\
    .filterBounds(study_area)

# 3) MODIS 8-day surface reflectance → compute NDVI
sr8 = ee.ImageCollection('MODIS/061/MOD09A1')\
    .select(['sur_refl_b02', 'sur_refl_b01'])\
    .filterDate(date_start, date_end)\
    .filterBounds(study_area)


def compute_ndvi(img):
    ndvi = img.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']) \
              .rename('NDVI') \
              .copyProperties(img, ['system:time_start'])
    return ndvi


ndvi8 = sr8.map(compute_ndvi)

# 4) Build 8-day periods instead of weeks
#    total number of 8-day blocks in your range:
numPeriods = date_end.difference(date_start, 'day').divide(8).floor()
periods = ee.List.sequence(0, numPeriods)


def count_period(i):
    i = ee.Number(i)
    start = date_start.advance(i.multiply(8), 'day')
    end = start.advance(8, 'day')

    # … hotspot as before …
    hs = fire_pts.filterDate(start, end).size()

    # rainfall: use dict.get('precipitation', 0)
    rain_dict = rainfall.filterDate(start, end).sum().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=study_area,
        scale=1000,
        bestEffort=True
    )
    rain_val = ee.Number(rain_dict.get('precipitation', 0))

    # NDVI: same pattern, default to 0 if missing
    nd_dict = ndvi8.filterDate(start, end).mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=study_area,
        scale=500,
        bestEffort=True
    )
    nd_val = ee.Number(nd_dict.get('NDVI', 0))

    return ee.Feature(None, {
        'period_start': start.format('YYYY-MM-dd'),
        'hotspot':      hs,
        'rainfall':     rain_val,
        'ndvi':         nd_val
    })


fc8 = ee.FeatureCollection(periods.map(count_period))

# 5) Bring into Python and build DataFrame
features = fc8.getInfo()['features']
df = pd.DataFrame([{
    'week':     f['properties']['period_start'],
    'hotspot':  f['properties']['hotspot'],
    'rainfall': f['properties']['rainfall'],
    'ndvi':     f['properties']['ndvi']
} for f in features])

df['week'] = pd.to_datetime(df['week'])
df = df.set_index('week').sort_index().astype(float).fillna(0)

# 6) Add your month_sin/month_cos in pandas
df['month'] = df.index.month
df['month_sin'] = np.sin(2*np.pi*df.month/12)
df['month_cos'] = np.cos(2*np.pi*df.month/12)


fire_pts = fire_points.flatten()
features = ['hotspot', 'month_sin', 'month_cos', 'rainfall', 'ndvi']
print(df[features].head(15))

# 1) Prepare multivariate sequences
data = df[features].values
# data = df['hotspot'].values

# scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# 2) Create sequences with a longer look-back


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


sequence_length = 30
X, y = create_sequences(scaled, sequence_length)
n_features = X.shape[-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# 8) Define and train LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(
        sequence_length, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Predict only hotspot
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=0)
