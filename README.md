# Pleiades Digital Twin -- Block B

A digital twin built on [PLEIAData](https://zenodo.org/records/7620136) (Pleiades building,
University of Murcia, PHOENIX EU project): predictive simulation + anomaly/efficiency detection
for Block B, scoped to an 8-week winter window (2021-02-01 to 2021-03-28).

See `C:\Users\hasan\.claude\plans\i-want-to-create-zazzy-hamster.md` for the full design rationale
(why Block B, why 2D not 3D, why XGBoost over ARIMA/Prophet, etc.).

## Architecture

1. **Structural model** (`src/structure.py`) -- block -> room -> sensor/HVAC/presence/CO2 device
   mapping, built from the dataset's `relations-*.csv` files. Output: `data/processed/structure.json`.
2. **State store** (`src/ingest.py`) -- tidy per-room-per-timestamp table (temp, HVAC, weather,
   presence, CO2) for the 8 fully-instrumented Block B rooms, plus a per-block consumption table.
   Builds on the dataset authors' own cleaned/resampled files rather than redoing their work.
3. **The brain** (`src/features.py`, `src/models.py`):
   - Model A (XGBoost): next-step indoor temperature per room.
   - Model B (XGBoost): next-hour block energy consumption.
   - Prophet baseline: univariate energy forecast (trend/seasonality only), shown for comparison --
     not used for simulation or anomaly detection (see plan for why).
   - Anomaly detection: residual-based (actual vs. model-expected) plus two rule-based checks
     (HVAC on with no detected motion and no CO2 rise (>20 ppm over the room baseline) in 2 hours; indoor temp far from setpoint while HVAC is on).
4. **Interface** (`app/app.py`) -- Streamlit app, 6 tabs: Overview, Rooms & Diagnosis, Compare Rooms,
   Weather What-if, Model & Evidence, About / Real Data.

## Running it

```
.venv\Scripts\python.exe src\structure.py B      # rebuild structural model (already done)
.venv\Scripts\python.exe src\ingest.py           # rebuild data/processed/*.parquet (already done)
.venv\Scripts\python.exe src\models.py           # retrain models + anomalies (already done)
.venv\Scripts\python.exe -m streamlit run app\app.py
```

Artifacts are already built under `data/processed/`, so just running the Streamlit
command is enough to see the app. Install with `pip install -r requirements.txt` (app only) or
`pip install -r requirements-train.txt` (also needed to rerun `src/*.py`, adds Prophet).

## Deploying (Streamlit Community Cloud)

The app reads only `data/processed/` (~7.6 MB); it does not need the raw `Data_Nature/` dataset or Prophet.
`.gitignore` already excludes `.venv/`, `Data_Nature/`, logs and caches.

1. Create a GitHub repo and push this folder (`git init`, `git add .`, `git commit`, `git push`).
2. At share.streamlit.io choose **New app**, pick the repo and branch, set the main file to `app/app.py`,
   and under Advanced settings choose **Python 3.12** (what the models were trained and tested on).
3. `requirements.txt` pins `xgboost`/`scikit-learn` to the versions that produced the pickled models in
   `data/processed/models/*.joblib`. If you retrain with different versions, regenerate and re-commit those files.

The app was verified in a clean virtualenv built only from `requirements.txt` plus the committed files.

## Current results

- Model A (indoor temp): RMSE 0.31 C on held-out test period.
- Model B (energy, XGBoost): RMSE 0.88 kWh vs. Prophet baseline RMSE 3.11 kWh -- the driver-aware
  model is ~3.5x more accurate, the core evidence for the model-choice decision in the plan.
- ~2,550 room-timestamps flagged across the three anomaly rules in the 8-week window (see the
  Anomalies tab's occupancy-sensor caveat before treating these as confirmed faults).

## Known limitations / next steps

- Occupancy is inferred from motion sensors that time out after seconds-to-minutes, so they
  undercount true presence (noted in the dataset's own documentation). The "HVAC on, no occupancy"
  rule uses a 2-hour no-motion window to reduce false positives, but it's a lead, not a confirmed
  fault -- tightening this (e.g. the gap-filling heuristic the dataset authors describe) is a good
  next step.
- Scoped to Block B / 8 weeks / 8 rooms. Steps 1-4 in the plan are parameterized by block and date
  range specifically so this can be re-run for more rooms/blocks/time without rework.
- No real floor-plan geometry exists in the dataset, so the "twin" visualization is a schematic 2D
  grid, not a true spatial layout (see plan for the reasoning).
