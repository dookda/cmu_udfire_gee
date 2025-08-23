# ================== 53-Week Forecast for Next Year ==================
# Assumptions:
# - You trained on WEEKLY data (7-day windows). If your current code uses 8-day windows,
#   change the two places where you have 'advance(i*8, "day")' and 'advance(8, "day")' to 7.
# - 'scaled' is the MinMax-scaled full series of your feature(s) (here only 'hotspot').
# - 'sequence_length' and 'n_features' are defined as in your training section.
# - 'scaler' was fit on df[features] and contains min/max for inverse scaling.

import numpy as np
import pandas as pd

# 1) Choose next year and number of weeks
next_year = (pd.to_datetime(df['date']).dt.year.max() + 1) if 'date' in df.columns else 2026
n_weeks = 53

# 2) Seed sequence: last 'sequence_length' timesteps from your scaled series
last_seq = scaled[-sequence_length:, :].copy()  # shape: (seq_len, n_features)

# 3) Iterative one-step-ahead predictions (feed previous prediction back as input)
preds_norm = []
seq = last_seq.copy()
for _ in range(n_weeks):
    x = seq.reshape(1, sequence_length, n_features)
    yhat = model.predict(x, verbose=0)[0, 0]  # normalized space
    preds_norm.append(yhat)
    # roll window: drop first row, append the new prediction as next row
    seq = np.vstack([seq[1:], [yhat] if n_features == 1 else [np.r_[yhat, seq[-1,1:]]]])

preds_norm = np.array(preds_norm)

# 4) Inverse scale back to hotspot counts
hot_min = scaler.data_min_[0]
hot_max = scaler.data_max_[0]
hot_rng = hot_max - hot_min
preds = preds_norm * hot_rng + hot_min

# 5) Build weekly dates for the next year (choose anchor weekday as needed)
start_date = pd.Timestamp(f"{int(next_year)}-01-01")
# Align to Mondays; adjust to 'W-SUN' or 'W-THU' if you prefer another anchor
dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

forecast_53w = pd.DataFrame({
    "date": dates,
    "pred_hotspot": preds.astype(float)
})

# Save
out_csv = "/mnt/data/forecast_next_year_53w.csv"
forecast_53w.to_csv(out_csv, index=False)
print("Saved 53-week forecast to:", out_csv)

# (Optional) Quick plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,3))
plt.plot(forecast_53w['date'], forecast_53w['pred_hotspot'])
plt.title(f"53-week Forecast for {int(next_year)}")
plt.xlabel("Week")
plt.ylabel("Predicted Hotspot Count")
plt.grid(True)
plt.tight_layout()
plt.show()
# ================== End Forecast Patch ==================