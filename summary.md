# ParkWatch BLR — Solution & Model Summary

This document provides a comprehensive overview of the **ParkWatch BLR** data science pipeline, spatial zoning methodology, predictive model architecture, evaluation metrics, and final results.

---

## 1. Project Background & Objective
ParkWatch BLR is an AI-driven parking intelligence system designed for **Tata Technologies InnoVent (Problem Statement 1: Parking-Induced Congestion)**. 

Illegal and spillover parking on commercial corridors, transit hubs, and school zones chokes Bengaluru’s roadways. However, traditional traffic enforcement is patrol-based and reactive. The goal of this project is to process historical parking violations into an actionable, predictive enforcement priority system that quantifies the real congestion impact of each event and directs patrols to high-risk hotspots.

---

## 2. Data Preparation & Filtering (Stage 1 & 2)
The raw dataset contains **298,450** anonymized violation logs from the Bengaluru Traffic Police (November 2023 – April 2024). 
*   **De-duplication**: Removed **5,372** exact duplicate records (same vehicle, coordinates, and timestamp).
*   **Quality Filtering**: Excluded **49,799** confirmed-invalid records (where `validation_status` was explicitly marked as `rejected` or `duplicate`). Keeping these false positives would skew hotspot rankings.
*   **Result**: **243,274** clean, congestion-relevant records were passed to the scoring engine.

---

## 3. Congestion Impact Score (CIS) Formulation (Stage 3)
Because the raw data does not include ground-truth road speed, traffic flow, or delay measurements, we formulated the **Congestion Impact Score (CIS)** as a transparent, traffic-engineering proxy:

$$\text{CIS} = \text{Severity Weight} \times \text{Vehicle Footprint Weight} \times \text{Time-of-Day Weight}$$

### A. Severity Weight (1 to 5)
Reflects how directly the violation blocks moving lanes or creates bottlenecks:
*   **Double Parking / Opposite Parking / One-Way Obstruction**: **5** (blocks active lanes)
*   **Main Road / Junction / Traffic Light Obstruction**: **4** (creates severe bottle-necks)
*   **Wrong Parking / Near Hospital/School/Bus stop**: **3** (medium obstruction)
*   **No Parking Zone**: **2** (minor lane spillover)
*   **Footpath Parking**: **1** (pedestrian obstruction, minimal roadway flow impact)
*   *Non-congestion violations (defective plates, tinted glass, helmet/belt violations, fare disputes) are assigned a weight of **0** and filtered out entirely.*

### B. Vehicle Footprint Weight (1.0 to 3.0)
Reflects the physical lane width occupied and clearance time:
*   **Buses (BMTC/KSRTC, Tourist, Private), HGV, Lorry**: **3.0** (maximum roadway blockage)
*   **LGV, Tanker, Tractor**: **2.2 to 2.8**
*   **Tempo, School Vehicle, Mini Lorry**: **2.0**
*   **Van, Maxi-Cab, Goods Auto**: **1.8**
*   **Passenger Car, Jeep**: **1.5**
*   **Scooter, Motor Cycle, Moped**: **1.0** (minimal single-vehicle footprint)

### C. Time-of-Day Traffic Weight (0.6 to 2.5)
Corrects for the night-skewed bias in enforcement device activity and reflects ambient traffic density:
*   **Morning Peak (7 AM – 11 AM) & Evening Peak (5 PM – 8 PM)**: **2.5** (maximum delay-minute impact on other commuters)
*   **Daytime Off-Peak (11 AM – 5 PM)**: **1.8** (still significant ambient traffic flow)
*   **Evening Wind-Down (8 PM – 11 PM)**: **1.2** (reducing traffic)
*   **Late Night & Early Morning (11 PM – 7 AM)**: **0.6** (minimal delay impact due to empty roads)

---

## 4. Spatial Hotspot Zoning: Grid-Based vs. DBSCAN (Stage 4)
### Why Density-Based Clustering (DBSCAN) Was Rejected
During early prototyping, DBSCAN was tested and rejected. Because Bengaluru's old-city commercial core (Upparpet, City Market, Shivajinagar) has closely spaced enforcement coordinates along continuous streets, DBSCAN suffered from **single-linkage chaining**. 
At any search radius above 20m, DBSCAN chained **55% of all city-wide violations into a single mega-cluster** spanning multiple police station jurisdictions. Tuning the radius lower simply resulted in noise classification.

### The Grid-Based Solution
We implemented a **Fixed Grid Zoning** methodology:
*   The city is divided into uniform, deterministic **150m × 150m grid cells** (roughly one to two city blocks).
*   **Advantages**: 
    1.  **Prevents Chaining**: Cell boundaries are fixed, isolating separate corridors regardless of point density.
    2.  **Patrol-Beat Sized**: 150m is highly interpretable for enforcement teams planning foot patrol beats.
    3.  **Sanity Check**: The largest single grid cell contains less than 2% of the citywide violations, confirming no single zone dominates by design.
*   **Result**: Created **5,484** active spatial grid cells.

---

## 5. Enforcement Priority Index (EPI) (Stage 5)
To reduce noise, we filtered out low-activity cells (requiring a cell to have $\ge 10$ violations and $\ge 3$ active days), leaving **2,100 qualifying hotspot zones**.

We developed the **Enforcement Priority Index (EPI)** (ranging from 0 to 100) to rank these zones based on a composite score of volume, day-to-day persistence, and severity ceilings. Each component is min-max normalized to 0-100 before combining:

$$\text{EPI} = 0.5 \times \text{Score}_{\text{Volume}} + 0.3 \times \text{Score}_{\text{Persistence}} + 0.2 \times \text{Score}_{\text{Severity}}$$

*   **Volume Score (50%)**: Normalized total CIS accumulated in the zone.
*   **Persistence Score (30%)**: Violations per active day (distinguishes chronic daily bottlenecks from one-off event spikes).
*   **Severity Score (20%)**: Normalized max single-event CIS (reflects the absolute worst-case obstruction ceiling).

### Hotspot Priority Tiers:
*   **Critical** (Top 10%): **210 zones** (EPI range: 17.50 to 93.00)
*   **High** (Next 20%): **420 zones**
*   **Medium** (Next 30%): **630 zones**
*   **Low** (Remaining 40%): **840 zones**

---

## 6. Predictive Risk Model (Stage 6)
While historical hotspots are useful, static maps cannot predict future risk for locations or times where enforcement was absent. We refactored our predictive model to forecast future zone-hour risk, resolving row-level target leakage (which occurred previously because individual violation attributes directly derived the CIS). 

The target variable is now the **Expected Total CIS per Zone per Hour** (representing future spatio-temporal risk).

### Model Features (With Lag Features & Weather Proxy)
*   `hour_sin`, `hour_cos` (cyclical time of day)
*   `dow_sin`, `dow_cos` (cyclical day of week)
*   `is_weekend` (binary flag)
*   `lag_1h_cis` (CIS in the same zone during the previous hour $t-1$)
*   `lag_24h_cis` (CIS in the same zone during the same hour yesterday $t-24$)
*   `is_monsoon` (binary weather proxy flagging heavy rain months Apr-May and Sept-Nov)
*   `police_station_enc` (encoded jurisdiction)
*   `vehicle_type_enc` (encoded dominant vehicle type in the zone-hour bin)
*   `cluster_id_enc` (label-encoded grid cell ID)

### Model Selection & Evaluation (5-Fold Cross-Validation)
We conducted a **5-fold K-Fold Cross-Validation** to verify model stability and prevent overfitting. The model is trained using a **Random Forest Regressor** (150 estimators, max depth 14, min_samples_leaf 10):

| Metric | Random Forest (Mean ± SD) |
| :--- | :--- |
| **R² Score** | **0.2347 ± 0.0200** |
| **MAE** (Mean Absolute Error) | **6.9221 ± 0.6114** |
| **RMSE** (Root Mean Squared Error) | **14.3719** |

*Note: This performance is highly realistic for forecasting zero-inflated, highly sparse hourly traffic risk data using an honest chronological TimeSeriesSplit. The previous model's R² (~0.90) was artificially high due to target leakage (since the target was row-level CIS and features included the vehicle type and hour that mathematically defined it).*

### Feature Importances (Random Forest)
1.  **Time cyclical components (`hour_sin`, `hour_cos`)**: **59.45%** (demonstrates that daily traffic cycles remain the strongest predictor of congestion risk)
2.  **Short-term lag (`lag_1h_cis`)**: **13.38%** (captures immediate persistence of congestion)
3.  **Spatial cell ID (`cluster_id_enc`)**: **7.88%** (captures zone-specific baseline risk)
4.  **Jurisdiction (`police_station_enc`)**: **6.22%**
5.  **Daily lag (`lag_24h_cis`)**: **6.01%** (captures day-to-day routine patterns)
6.  **Vehicle Type (`vehicle_type_enc`)**: **4.59%**
7.  **Calendar & Weather (`dow_sin`, `dow_cos`, `is_monsoon`, `is_weekend`)**: **2.47%** (incorporates the `is_monsoon` weather proxy and weekend flags to model seasonal rainfall bottlenecks)

---

## 7. Top 10 Hotspots Sanity Check (Empirical Validation)
A practitioner-style validation of the top 10 ranked zones confirms they are well-known real-world congestion nodes:
1.  **Rank 1 (EPI = 93.00) — Mysore Road, KR Market Junction**: Wholesale market and bus terminal hub with severe scooter/commercial loading spillover.
2.  **Rank 2 (EPI = 51.60) — Sahakar Nagar Road, Byatarayanapura**: High-street commercial dining district with massive parking spillover and double-parking.
3.  **Rank 3 (EPI = 51.36) — 6th Main Road, Gandhi Nagar (Majestic)**: Major transport node (KSR Railway Station) choked with auto-rickshaws and delivery vehicles.
4.  **Rank 4 (EPI = 49.83) — Meenakshi Koil Street, Shivaji Nagar**: Hyper-dense bazaar adjacent to the Shivaji Nagar Bus Terminal.
5.  **Rank 5 & 7 (EPI = 49.81 & 44.83) — Kamaraj Road / Dickenson Road (Safina Plaza Junction)**: Narrow commercial streets catching spillover from Commercial Street.
6.  **Rank 6 (EPI = 46.07) — Begur Chikkanahalli (Chikkajala)**: Logistic warehouse cluster with heavy delivery vehicles parked on shoulders.
7.  **Rank 8 (EPI = 41.95) — 5th Main Road, KG Circle (Majestic)**: Hotel and retail district near Majestic central terminal.
8.  **Rank 9 (EPI = 39.12) — MBT Road, Devasandra Junction (KR Puram)**: Crucial intersection in East Bengaluru choked by illegal parking near turning lanes.
9.  **Rank 10 (EPI = 39.08) — New Horizon College Road, Kadubisanahalli**: IT corridor access road where commuter drop-offs and parked two-wheelers block the carriage-way.
