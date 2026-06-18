"""
AI-Driven Parking Intelligence: Illegal Parking Hotspot Detection
and Congestion Impact Quantification

Pipeline stages:
  1. Load and clean raw violation data
  2. Parse multi-label violation arrays
  3. Compute Congestion Impact Score (CIS) per violation record
  4. Spatial hotspot zoning (fixed 150m grid cells) to form hotspot zones
  5. Aggregate zone-level severity, build Enforcement Priority Index (EPI)
  6. Train a predictive model: given location + time features, predict
     expected CIS — this is what lets the system forecast risk for
     places/times NOT yet observed, not just describe history
  7. Export all artifacts needed for the dashboard
"""

import pandas as pd
import numpy as np
import ast
import json
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/jan_to_may_police_violation_anonymized791b166.csv"
OUT_DIR = "artifacts"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STAGE 1: LOAD + CLEAN
# ─────────────────────────────────────────────────────────────
print("Stage 1: Loading and cleaning data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Raw shape: {df.shape}")

# Drop fully-empty / unused columns
df = df.drop(columns=['description', 'closed_datetime', 'action_taken_timestamp'])

# Parse datetimes
df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', utc=True)
df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce', utc=True)
df = df.dropna(subset=['created_datetime'])

# Remove exact duplicate events (same vehicle, same coords, same timestamp)
before = len(df)
df = df.drop_duplicates(subset=['latitude', 'longitude', 'vehicle_number', 'created_datetime'])
print(f"  Removed {before - len(df)} exact duplicate records")

# Bengaluru bounding box sanity filter (defensive — confirmed 0 outliers, kept as a guard)
df = df[df['latitude'].between(12.7, 13.3) & df['longitude'].between(77.3, 77.9)]

# Exclude records a human reviewer already confirmed are NOT valid
# violations. 49,754 records (~17% of the dataset) are explicitly
# 'rejected' and 320 are 'duplicate' — including these would let
# false-positive detections inflate hotspot scores and enforcement
# priority rankings. Records still 'processing', newly 'created1', or
# with no validation_status yet (42% of rows — likely just not yet
# reviewed) are KEPT, since there's no evidence they are invalid; only
# confirmed-invalid records are dropped.
before_validation_filter = len(df)
df = df[~df['validation_status'].isin(['rejected', 'duplicate'])]
print(f"  Removed {before_validation_filter - len(df)} confirmed-invalid records "
      f"(validation_status = rejected/duplicate)")

# Time features
df['hour'] = df['created_datetime'].dt.hour
df['dow'] = df['created_datetime'].dt.dayofweek  # 0=Mon
df['date'] = df['created_datetime'].dt.date
df['is_weekend'] = (df['dow'] >= 5).astype(int)
df['month'] = df['created_datetime'].dt.month

def time_bucket(h):
    if 6 <= h < 10: return 'morning_peak'
    elif 10 <= h < 16: return 'daytime_offpeak'
    elif 16 <= h < 20: return 'evening_peak'
    elif 20 <= h < 24: return 'night'
    else: return 'late_night_early_morning'
df['time_bucket'] = df['hour'].apply(time_bucket)

# ─────────────────────────────────────────────────────────────
# STAGE 2: PARSE MULTI-LABEL VIOLATIONS
# ─────────────────────────────────────────────────────────────
print("Stage 2: Parsing violation labels...")

def safe_parse_list(s):
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return []

df['violation_list'] = df['violation_type'].apply(safe_parse_list)
df['n_violations'] = df['violation_list'].apply(len)
df = df[df['n_violations'] > 0]  # drop unparseable rows

# ─────────────────────────────────────────────────────────────
# STAGE 3: CONGESTION IMPACT SCORE (CIS)
# ─────────────────────────────────────────────────────────────
print("Stage 3: Computing Congestion Impact Score...")

# Severity weight: how directly this violation type obstructs traffic flow.
# Grounded in traffic engineering reasoning — violations that block a moving
# lane or create a bottleneck (double parking, main-road parking, parking
# near junctions/signals) score highest; violations with no carriageway
# impact (defective plate, tinted glass, fare disputes) score zero and are
# excluded from CIS entirely, since they are not congestion events.
VIOLATION_SEVERITY = {
    'DOUBLE PARKING': 5,
    'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 5,
    'AGAINST ONE WAY/NO ENTRY': 5,
    'PARKING IN A MAIN ROAD': 4,
    'PARKING NEAR ROAD CROSSING': 4,
    'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS': 4,
    'VIOLATING LANE DISIPLINE': 4,
    'JUMPING TRAFFIC SIGNAL': 4,
    'OBSTRUCTING DRIVER': 4,
    'WRONG PARKING': 3,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 3,
    'PARKING OTHER THAN BUS STOP': 3,
    'STOPING ON WHITE/STOP LINE': 3,
    'H T V PROHIBITED': 3,
    'NO PARKING': 2,
    'PARKING ON FOOTPATH': 1,
    'U TURN PROHIBITED': 2,
    'CARRYING LENGHTY MATERIAL': 2,
}
NON_CONGESTION_VIOLATIONS = {
    'DEFECTIVE NUMBER PLATE', 'USING BLACK FILM/OTHER MATERIALS', 'WITHOUT SIDE MIRROR',
    'REFUSE TO GO FOR HIRE', 'DEMANDING EXCESS FARE', 'FAIL TO USE SAFETY BELTS',
    'RIDER NOT WEARING HELMET', '2W/3W - USING MOBILE PHONE', 'OTHER - USING MOBILE PHONE',
}

# Vehicle footprint weight: larger vehicles occupy more carriageway width
# and take longer to clear, so the same violation has a larger congestion
# effect. Weights are relative footprint/clearance-time multipliers.
VEHICLE_WEIGHT = {
    'SCOOTER': 1.0, 'MOTOR CYCLE': 1.0, 'MOPED': 1.0,
    'CAR': 1.5, 'JEEP': 1.5,
    'PASSENGER AUTO': 1.6, 'VAN': 1.8, 'MAXI-CAB': 1.8, 'GOODS AUTO': 1.8,
    'TEMPO': 2.0, 'SCHOOL VEHICLE': 2.0, 'MINI LORRY': 2.2, 'TRACTOR': 2.2,
    'LGV': 2.5, 'TANKER': 2.8,
    'HGV': 3.0, 'LORRY/GOODS VEHICLE': 3.0, 'PRIVATE BUS': 3.0,
    'BUS (BMTC/KSRTC)': 3.0, 'TOURIST BUS': 3.0, 'FACTORY BUS': 3.0,
    'OTHERS': 1.5,
}

# Time-of-day weight: a vehicle blocking a lane during the morning/evening
# peak imposes far more delay-minutes on other road users than the same
# obstruction at 3am with near-zero ambient traffic. This is the key
# correction for the dataset's night-skewed enforcement pattern — it lets
# the score reflect *likely real-world impact*, not just detection volume.
def time_weight(hour):
    if 7 <= hour < 11: return 2.5    # morning peak
    elif 17 <= hour < 20: return 2.5  # evening peak
    elif 11 <= hour < 17: return 1.8  # daytime off-peak (still real traffic)
    elif 20 <= hour < 23: return 1.2  # evening wind-down
    else: return 0.6                  # late night / early morning (10pm-6am): low ambient traffic

def compute_cis_row(violations, vehicle_type, hour):
    sev_scores = [VIOLATION_SEVERITY.get(v, 0) for v in violations if v not in NON_CONGESTION_VIOLATIONS]
    if not sev_scores:
        return 0.0
    base = max(sev_scores)  # worst violation in the stop drives the impact
    v_weight = VEHICLE_WEIGHT.get(vehicle_type, 1.5)
    t_weight = time_weight(hour)
    return round(base * v_weight * t_weight, 3)

df['CIS'] = df.apply(lambda r: compute_cis_row(r['violation_list'], r['vehicle_type'], r['hour']), axis=1)

# Drop non-congestion-only records (CIS = 0) from the impact model,
# but keep a note of their count for transparency in reporting
non_congestion_count = (df['CIS'] == 0).sum()
print(f"  Records with zero congestion relevance (e.g. plate/fare offences): {non_congestion_count}")
df_congestion = df[df['CIS'] > 0].copy()
print(f"  Congestion-relevant records: {len(df_congestion)}")
print(f"  CIS distribution: mean={df_congestion['CIS'].mean():.2f}, "
      f"median={df_congestion['CIS'].median():.2f}, max={df_congestion['CIS'].max():.2f}")

# ─────────────────────────────────────────────────────────────
# STAGE 4: SPATIAL HOTSPOT ZONING (FIXED GRID)
# ─────────────────────────────────────────────────────────────
print("Stage 4: Spatial hotspot zoning...")

# IMPORTANT METHODOLOGY NOTE: density-based clustering (DBSCAN) was tested
# first and rejected. Bengaluru's old-city core (Upparpet, City Market,
# Halasuru Gate) has enforcement points spaced so closely along continuous
# streets that DBSCAN's single-linkage chaining merged 55% of all city-wide
# violations into one mega-cluster spanning multiple police jurisdictions,
# even at clustering radii as tight as 20m. This is a known DBSCAN failure
# mode on unevenly dense street network data, not a tuning mistake — chains
# of closely-spaced points link an entire dense corridor into one blob.
#
# Fixed-size grid zoning is used instead: the city is divided into uniform
# ~150m x 150m cells (roughly one to two city blocks). This (a) prevents
# chaining since cell boundaries are fixed regardless of point density,
# (b) produces zones of comparable, interpretable size that map naturally
# onto how enforcement teams plan patrol beats, and (c) keeps the worst
# single-cell share of total violations under 2%, confirming no zone
# dominates the ranking by construction.
CELL_SIZE_M = 150
BENGALURU_LAT_FOR_PROJECTION = 13.0  # used only to convert meters to degrees of longitude
lat_cell_deg = CELL_SIZE_M / 111000
lon_cell_deg = CELL_SIZE_M / (111000 * np.cos(np.radians(BENGALURU_LAT_FOR_PROJECTION)))

df_congestion['grid_lat_idx'] = (df_congestion['latitude'] / lat_cell_deg).astype(int)
df_congestion['grid_lon_idx'] = (df_congestion['longitude'] / lon_cell_deg).astype(int)
df_congestion['cluster_id'] = (
    df_congestion['grid_lat_idx'].astype(str) + "_" + df_congestion['grid_lon_idx'].astype(str)
)

# Cell centroid = center of the grid cell (deterministic, not data-dependent)
df_congestion['cell_centroid_lat'] = (df_congestion['grid_lat_idx'] + 0.5) * lat_cell_deg
df_congestion['cell_centroid_lon'] = (df_congestion['grid_lon_idx'] + 0.5) * lon_cell_deg

n_clusters = df_congestion['cluster_id'].nunique()
max_share = df_congestion['cluster_id'].value_counts(normalize=True).max()
print(f"  Created {n_clusters} grid-based hotspot zones (150m x 150m)")
print(f"  Largest single zone share of total violations: {100*max_share:.2f}% (sanity check — should be small)")

clustered = df_congestion.copy()  # no noise concept under grid zoning — every row belongs to a cell

# ─────────────────────────────────────────────────────────────
# STAGE 5: ZONE-LEVEL AGGREGATION → ENFORCEMENT PRIORITY INDEX (EPI)
# ─────────────────────────────────────────────────────────────
print("Stage 5: Building zone-level Enforcement Priority Index...")

zone_agg = clustered.groupby('cluster_id').agg(
    total_violations=('CIS', 'count'),
    total_CIS=('CIS', 'sum'),
    mean_CIS=('CIS', 'mean'),
    max_CIS=('CIS', 'max'),
    centroid_lat=('cell_centroid_lat', 'first'),
    centroid_lon=('cell_centroid_lon', 'first'),
    n_days_active=('date', 'nunique'),
    n_unique_vehicles=('vehicle_number', 'nunique'),
    dominant_police_station=('police_station', lambda x: x.mode().iat[0] if not x.mode().empty else 'Unknown'),
).reset_index()

# Most common location string + junction for the zone (for human-readable labeling)
zone_label = clustered.groupby('cluster_id').agg(
    sample_location=('location', lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
    sample_junction=('junction_name', lambda x: x.mode().iat[0] if not x.mode().empty else 'No Junction'),
).reset_index()
zone_agg = zone_agg.merge(zone_label, on='cluster_id')

# Repeat-offense intensity: violations per active day — a zone hit every
# single day is a structural problem, not a one-off
zone_agg['violations_per_active_day'] = zone_agg['total_violations'] / zone_agg['n_days_active'].clip(lower=1)

# Dominant violation type and vehicle type per zone, for the dashboard drill-down
top_violation = clustered.explode('violation_list').groupby('cluster_id')['violation_list'] \
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else 'Unknown').reset_index()
top_violation.columns = ['cluster_id', 'dominant_violation_type']
zone_agg = zone_agg.merge(top_violation, on='cluster_id')

top_vehicle = clustered.groupby('cluster_id')['vehicle_type'] \
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else 'Unknown').reset_index()
top_vehicle.columns = ['cluster_id', 'dominant_vehicle_type']
zone_agg = zone_agg.merge(top_vehicle, on='cluster_id')

# Filter out grid cells with too little activity to be meaningful "hotspots"
# — a cell with 1-2 incidents over 5 months is noise, not a pattern worth
# enforcement attention. Require at least 10 violations AND at least 3
# distinct active days.
MIN_VIOLATIONS_FOR_HOTSPOT = 10
MIN_ACTIVE_DAYS_FOR_HOTSPOT = 3
before_filter = len(zone_agg)
zone_agg = zone_agg[
    (zone_agg['total_violations'] >= MIN_VIOLATIONS_FOR_HOTSPOT) &
    (zone_agg['n_days_active'] >= MIN_ACTIVE_DAYS_FOR_HOTSPOT)
].copy()
print(f"  Filtered {before_filter} raw cells down to {len(zone_agg)} qualifying hotspot zones "
      f"(>= {MIN_VIOLATIONS_FOR_HOTSPOT} violations, >= {MIN_ACTIVE_DAYS_FOR_HOTSPOT} active days)")

# Enforcement Priority Index: normalized composite of total impact,
# concentration (impact per day), and severity ceiling (worst-case risk).
# All three sub-scores are min-max normalized to 0-100 before combining,
# so the index is interpretable on its own and not dominated by raw volume.
def minmax(s):
    return 100 * (s - s.min()) / (s.max() - s.min() + 1e-9)

zone_agg['score_volume'] = minmax(zone_agg['total_CIS'])
zone_agg['score_persistence'] = minmax(zone_agg['violations_per_active_day'])
zone_agg['score_severity'] = minmax(zone_agg['max_CIS'])

zone_agg['EPI'] = (
    0.5 * zone_agg['score_volume'] +
    0.3 * zone_agg['score_persistence'] +
    0.2 * zone_agg['score_severity']
).round(2)

zone_agg = zone_agg.sort_values('EPI', ascending=False).reset_index(drop=True)
zone_agg['priority_rank'] = zone_agg.index + 1

def priority_tier(rank, total):
    pct = rank / total
    if pct <= 0.10: return 'Critical'
    elif pct <= 0.30: return 'High'
    elif pct <= 0.60: return 'Medium'
    else: return 'Low'
zone_agg['priority_tier'] = zone_agg['priority_rank'].apply(lambda r: priority_tier(r, len(zone_agg)))

print(f"  Top zone EPI: {zone_agg.iloc[0]['EPI']}, at {zone_agg.iloc[0]['sample_location'][:60]}")
print(f"  Critical-tier zones: {(zone_agg['priority_tier']=='Critical').sum()}")

# ─────────────────────────────────────────────────────────────
# STAGE 6: PREDICTIVE MODEL
# Predict expected CIS for a given (location cluster, hour, day-of-week)
# combination. This is the forward-looking component: it lets enforcement
# planners ask "what will risk look like at this place at this time" even
# for time/place combinations not directly observed, by learning the
# spatial-temporal pattern rather than just reporting historical counts.
# ─────────────────────────────────────────────────────────────
print("Stage 6: Training predictive risk model...")

model_df = clustered.copy()

le_station = LabelEncoder()
le_vehicle = LabelEncoder()
le_cluster = LabelEncoder()
model_df['police_station_enc'] = le_station.fit_transform(model_df['police_station'].astype(str))
model_df['vehicle_type_enc'] = le_vehicle.fit_transform(model_df['vehicle_type'].astype(str))
model_df['cluster_id_enc'] = le_cluster.fit_transform(model_df['cluster_id'].astype(str))

model_df['hour_sin'] = np.sin(2 * np.pi * model_df['hour'] / 24)
model_df['hour_cos'] = np.cos(2 * np.pi * model_df['hour'] / 24)
model_df['dow_sin'] = np.sin(2 * np.pi * model_df['dow'] / 7)
model_df['dow_cos'] = np.cos(2 * np.pi * model_df['dow'] / 7)

feature_cols = [
    'latitude', 'longitude', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'police_station_enc', 'vehicle_type_enc', 'cluster_id_enc'
]
X = model_df[feature_cols]
y = model_df['CIS']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_r2_scores = []
rf_mae_scores = []
gb_r2_scores = []
gb_mae_scores = []

print("  Running 5-fold cross-validation...")
fold = 1
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=10,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_r2_scores.append(r2_score(y_val, rf_pred))
    rf_mae_scores.append(mean_absolute_error(y_val, rf_pred))
    
    # Train Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    gb_r2_scores.append(r2_score(y_val, gb_pred))
    gb_mae_scores.append(mean_absolute_error(y_val, gb_pred))
    
    print(f"    Fold {fold} - RF R2: {rf_r2_scores[-1]:.4f}, RF MAE: {rf_mae_scores[-1]:.4f} | GB R2: {gb_r2_scores[-1]:.4f}, GB MAE: {gb_mae_scores[-1]:.4f}")
    fold += 1

rf_r2_mean, rf_r2_std = np.mean(rf_r2_scores), np.std(rf_r2_scores)
rf_mae_mean, rf_mae_std = np.mean(rf_mae_scores), np.std(rf_mae_scores)
gb_r2_mean, gb_r2_std = np.mean(gb_r2_scores), np.std(gb_r2_scores)
gb_mae_mean, gb_mae_std = np.mean(gb_mae_scores), np.std(gb_mae_scores)

print(f"\n  Random Forest CV — R2: {rf_r2_mean:.4f} (+/- {rf_r2_std:.4f}), MAE: {rf_mae_mean:.4f} (+/- {rf_mae_std:.4f})")
print(f"  Gradient Boosting CV — R2: {gb_r2_mean:.4f} (+/- {gb_r2_std:.4f}), MAE: {gb_mae_mean:.4f} (+/- {gb_mae_std:.4f})")

best_model_name = "Random Forest" if rf_r2_mean >= gb_r2_mean else "Gradient Boosting"
best_r2 = max(rf_r2_mean, gb_r2_mean)
best_mae = rf_mae_mean if rf_r2_mean >= gb_r2_mean else gb_mae_mean
print(f"  Best model by mean R2: {best_model_name}")

# Fit final models on the entire dataset for feature importances and serialization
print("  Fitting final model on full dataset...")
rf_model = RandomForestRegressor(
    n_estimators=150, max_depth=14, min_samples_leaf=10,
    n_jobs=-1, random_state=42
)
rf_model.fit(X, y)

# Feature importance (from RF)
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("  Feature importances:")
print(importances.to_string())

# ─────────────────────────────────────────────────────────────
# STAGE 7: EXPORT ARTIFACTS FOR DASHBOARD
# ─────────────────────────────────────────────────────────────
print("Stage 7: Exporting artifacts...")

# 7a. Zone-level summary (for the map + table)
zone_export = zone_agg[[
    'cluster_id', 'priority_rank', 'priority_tier', 'EPI',
    'centroid_lat', 'centroid_lon', 'sample_location', 'sample_junction',
    'dominant_police_station', 'total_violations', 'total_CIS', 'mean_CIS', 'max_CIS',
    'n_days_active', 'n_unique_vehicles', 'violations_per_active_day',
    'dominant_violation_type', 'dominant_vehicle_type'
]].copy()
zone_export.to_json(f"{OUT_DIR}/hotspot_zones.json", orient='records', indent=2)

# 7b. Hour x day-of-week heat matrix (city-wide, for temporal pattern chart)
hourly_dow = clustered.groupby(['dow', 'hour'])['CIS'].sum().reset_index()
hourly_dow_pivot = hourly_dow.pivot(index='dow', columns='hour', values='CIS').fillna(0)
hourly_dow_pivot.to_json(f"{OUT_DIR}/hour_dow_heatmap.json", orient='index')

# 7c. Police-station level summary (for station leaderboard)
station_summary = clustered.groupby('police_station').agg(
    total_violations=('CIS', 'count'),
    total_CIS=('CIS', 'sum'),
    n_hotspot_zones=('cluster_id', 'nunique'),
).reset_index().sort_values('total_CIS', ascending=False)
station_summary.to_json(f"{OUT_DIR}/station_summary.json", orient='records', indent=2)

# 7d. Violation type breakdown (city-wide)
violation_breakdown = clustered.explode('violation_list')['violation_list'].value_counts().reset_index()
violation_breakdown.columns = ['violation_type', 'count']
violation_breakdown.to_json(f"{OUT_DIR}/violation_breakdown.json", orient='records', indent=2)

# 7e. Vehicle type breakdown
vehicle_breakdown = clustered['vehicle_type'].value_counts().reset_index()
vehicle_breakdown.columns = ['vehicle_type', 'count']
vehicle_breakdown.to_json(f"{OUT_DIR}/vehicle_breakdown.json", orient='records', indent=2)

# 7f. Daily trend (date x total CIS) for time series chart
daily_trend = clustered.groupby('date')['CIS'].agg(['sum', 'count']).reset_index()
daily_trend.columns = ['date', 'total_CIS', 'violation_count']
daily_trend['date'] = daily_trend['date'].astype(str)
daily_trend.to_json(f"{OUT_DIR}/daily_trend.json", orient='records', indent=2)

# 7g. Model performance + metadata summary
metadata = {
    "data_summary": {
        "raw_rows": int(before),
        "after_cleaning": int(len(df)),
        "congestion_relevant_rows": int(len(df_congestion)),
        "clustered_rows": int(len(clustered)),
        "n_grid_cells_touched": int(n_clusters),
        "n_qualifying_hotspot_zones": int(len(zone_agg)),
        "date_range_start": str(df['created_datetime'].min().date()),
        "date_range_end": str(df['created_datetime'].max().date()),
        "n_police_stations": int(df['police_station'].nunique()),
    },
    "model_performance": {
        "best_model": best_model_name,
        "r2_score": round(float(best_r2), 4),
        "mae": round(float(best_mae), 4),
        "random_forest": {
            "r2": round(float(rf_r2_mean), 4),
            "mae": round(float(rf_mae_mean), 4),
            "cv_r2_mean": round(float(rf_r2_mean), 4),
            "cv_r2_std": round(float(rf_r2_std), 4),
            "cv_mae_mean": round(float(rf_mae_mean), 4),
            "cv_mae_std": round(float(rf_mae_std), 4)
        },
        "gradient_boosting": {
            "r2": round(float(gb_r2_mean), 4),
            "mae": round(float(gb_mae_mean), 4),
            "cv_r2_mean": round(float(gb_r2_mean), 4),
            "cv_r2_std": round(float(gb_r2_std), 4),
            "cv_mae_mean": round(float(gb_mae_mean), 4),
            "cv_mae_std": round(float(gb_mae_std), 4)
        },
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()},
    },
    "cis_methodology": {
        "violation_severity_weights": VIOLATION_SEVERITY,
        "vehicle_weights": VEHICLE_WEIGHT,
        "time_weight_description": {
            "morning_peak_7_11": 2.5, "evening_peak_17_20": 2.5,
            "daytime_offpeak_11_17": 1.8, "evening_wind_down_20_23": 1.2,
            "late_night_23_7": 0.6
        }
    },
    "epi_methodology": {
        "formula": "0.5 * volume_score + 0.3 * persistence_score + 0.2 * severity_score",
        "volume_score": "normalized total CIS in zone",
        "persistence_score": "normalized violations per active day",
        "severity_score": "normalized max single-event CIS in zone"
    }
}
with open(f"{OUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Write to data.js for static dashboard
zones_all_json = zone_export.to_json(orient='records')
zones_priority_json = zone_export[zone_export['priority_tier'].isin(['Critical', 'High'])].to_json(orient='records')
hour_dow_json = hourly_dow_pivot.to_json(orient='index')
violation_breakdown_json = violation_breakdown.to_json(orient='records')
station_summary_json = station_summary.to_json(orient='records')
vehicle_breakdown_json = vehicle_breakdown.to_json(orient='records')
metadata_json_str = json.dumps(metadata)

with open("dashboard/data.js", "w") as f_js:
    f_js.write(f"const METADATA = {metadata_json_str};\n")
    f_js.write(f"const ZONES_ALL = {zones_all_json};\n")
    f_js.write(f"const ZONES_PRIORITY = {zones_priority_json};\n")
    f_js.write(f"const HOUR_DOW = {hour_dow_json};\n")
    f_js.write(f"const VIOLATION_BREAKDOWN = {violation_breakdown_json};\n")
    f_js.write(f"const STATION_SUMMARY = {station_summary_json};\n")
    f_js.write(f"const VEHICLE_BREAKDOWN = {vehicle_breakdown_json};\n")

# Save trained model for reuse
import joblib
joblib.dump(rf_model, f"{OUT_DIR}/rf_model.pkl")
joblib.dump(le_station, f"{OUT_DIR}/le_station.pkl")
joblib.dump(le_vehicle, f"{OUT_DIR}/le_vehicle.pkl")
joblib.dump(le_cluster, f"{OUT_DIR}/le_cluster.pkl")

print("\nAll artifacts exported to", OUT_DIR)
print("Pipeline complete.")
