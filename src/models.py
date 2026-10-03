"""Train Model A (indoor temp), Model B (energy), a Prophet baseline, and
compute anomaly/efficiency flags. Saves everything the Streamlit app needs
so the app doesn't retrain on every run.
"""
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from config import PROCESSED_DIR, PROCESSED_SRC_DIR, ROOMS_TABLE_PATH, WINDOW_START, WINDOW_END, BLOCK
from diagnosis import classify_room_row
from features import (
    ENERGY_FEATURE_COLS,
    ENERGY_TARGET,
    TEMP_FEATURE_COLS,
    TEMP_TARGET,
    make_energy_table,
    make_temp_table,
)

MODELS_DIR = PROCESSED_DIR / "models"
TEST_FRACTION = 0.2  # last 20% of the window, by time, held out
CO2_EXCESS_PPM = 20  # CO2 above room baseline that counts as "someone is/was here"; see compute_anomalies


def time_split(df: pd.DataFrame, date_col: str = "Date"):
    df = df.sort_values(date_col)
    cutoff = df[date_col].quantile(1 - TEST_FRACTION)
    train = df[df[date_col] < cutoff]
    test = df[df[date_col] >= cutoff]
    return train, test


def regression_metrics(actual: pd.Series, pred: np.ndarray) -> dict:
    """RMSE/MAE plus the two ASHRAE Guideline 14 measurement & verification
    metrics: CV(RMSE) (typical error as % of mean actual) and NMBE (overall
    bias as % of mean actual). Guideline 14 accepts hourly models under 30%
    CV(RMSE) and within +-10% NMBE."""
    actual = np.asarray(actual)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    mean_actual = float(actual.mean())
    cv_rmse = rmse / mean_actual * 100 if mean_actual else float("nan")
    nmbe = float((actual - pred).mean()) / mean_actual * 100 if mean_actual else float("nan")
    return {"rmse": rmse, "mae": mae, "cv_rmse": cv_rmse, "nmbe": nmbe, "n_test": int(len(actual))}


def train_temp_model(feature_cols=None):
    feature_cols = feature_cols or TEMP_FEATURE_COLS
    rooms_df = pd.read_parquet(ROOMS_TABLE_PATH)
    table = make_temp_table(rooms_df)
    train, test = time_split(table)

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=0)
    model.fit(train[feature_cols], train[TEMP_TARGET])

    pred = model.predict(test[feature_cols])
    metrics = regression_metrics(test[TEMP_TARGET], pred)
    print(f"[Model A: indoor temp] test RMSE={metrics['rmse']:.3f} C, CV(RMSE)={metrics['cv_rmse']:.2f}%, NMBE={metrics['nmbe']:.2f}%, n_test={metrics['n_test']}")

    # Residuals on the full table (train+test) -- used for anomaly detection.
    table = table.copy()
    table["V2_next_pred"] = model.predict(table[feature_cols])
    table["temp_residual"] = table[TEMP_TARGET] - table["V2_next_pred"]

    return model, table, metrics


def run_occupancy_ablation():
    """Does knowing presence/CO2 actually help the temperature model, or is
    weather+HVAC state sufficient? Mirrors the reference demo's solar-radiation
    ablation: train the same model with and without a feature group and compare
    held-out error."""
    without_cols = [c for c in TEMP_FEATURE_COLS if c not in ("presence", "co2_ppm")]
    _, _, with_metrics = train_temp_model(TEMP_FEATURE_COLS)
    _, _, without_metrics = train_temp_model(without_cols)
    print(f"[Ablation] with presence+CO2: RMSE={with_metrics['rmse']:.3f} C | without: RMSE={without_metrics['rmse']:.3f} C")
    return {"with_occupancy": with_metrics, "without_occupancy": without_metrics}


def train_energy_model():
    cons_df = pd.read_csv(
        PROCESSED_SRC_DIR / f"data-model-consumo{BLOCK}-60T.csv",
        sep=";",
        index_col=0,
        parse_dates=["Date"],
    )
    cons_df = cons_df[(cons_df["Date"] >= WINDOW_START) & (cons_df["Date"] <= WINDOW_END)]
    table = make_energy_table(cons_df)
    train, test = time_split(table)

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=0)
    model.fit(train[ENERGY_FEATURE_COLS], train[ENERGY_TARGET])

    pred = model.predict(test[ENERGY_FEATURE_COLS])
    metrics = regression_metrics(test[ENERGY_TARGET], pred)
    print(f"[Model B: energy, XGBoost] test RMSE={metrics['rmse']:.3f} kWh, CV(RMSE)={metrics['cv_rmse']:.2f}%, NMBE={metrics['nmbe']:.2f}%, n_test={metrics['n_test']}")

    table = table.copy()
    table["energy_pred"] = model.predict(table[ENERGY_FEATURE_COLS])
    table["energy_residual"] = table[ENERGY_TARGET] - table["energy_pred"]

    return model, table, metrics


def train_prophet_baseline(energy_table: pd.DataFrame):
    """Univariate baseline: trend + daily/weekly seasonality only, no
    weather/HVAC/occupancy. Used purely as a comparison point in the demo,
    not for what-if simulation or as the anomaly baseline (see plan Step 6)."""
    from prophet import Prophet

    df = energy_table[["Date", ENERGY_TARGET]].rename(columns={"Date": "ds", ENERGY_TARGET: "y"})
    df["ds"] = df["ds"].dt.tz_localize(None)
    train, test = time_split(df, date_col="ds")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(train)

    future = test[["ds"]]
    forecast = m.predict(future)
    pred = forecast["yhat"].values
    metrics = regression_metrics(test["y"], pred)
    print(f"[Model B baseline: Prophet] test RMSE={metrics['rmse']:.3f} kWh, CV(RMSE)={metrics['cv_rmse']:.2f}%, NMBE={metrics['nmbe']:.2f}%, n_test={metrics['n_test']}")

    comparison = test.copy()
    comparison["prophet_pred"] = pred
    return m, comparison, metrics


def compute_anomalies(temp_table: pd.DataFrame, energy_table: pd.DataFrame) -> pd.DataFrame:
    """Residual-based (vs. Model A/B's 'expected given conditions') plus
    simple rule-based efficiency checks, per plan Step 7."""
    t = temp_table.copy()
    t["temp_residual_z"] = (t["temp_residual"] - t["temp_residual"].mean()) / t["temp_residual"].std()
    t["anomaly_temp_residual"] = t["temp_residual_z"].abs() > 3

    # Rule: HVAC on (V4 > 0) with no motion detected in the last 2 hours (12 x 10-min
    # steps), per room. A longer window than "no motion right now" is used because the
    # dataset's own documentation notes presence sensors are motion-triggered and time
    # out after a few seconds to minutes -- they undercount true occupancy, so a short
    # window would flag most occupied-but-still rooms as empty.
    t = t.sort_values(["room", "Date"])
    t["presence_rolling"] = t.groupby("room")["presence"].transform(lambda s: s.rolling(12, min_periods=1).max())
    t["anomaly_hvac_no_motion"] = (t["V4"] > 0) & (t["presence_rolling"] == 0)

    # CO2 occupancy evidence. Sensors have room-specific offsets (median 130-430 ppm across
    # rooms), so absolute levels are meaningless; use the excess over each room's own rolling
    # 24h 10th-percentile baseline. A room counts as occupied if the excess topped
    # CO2_EXCESS_PPM at any point in the last 2 hours (CO2 lingers after people leave, which
    # also covers still-but-occupied rooms that motion sensors miss).
    co2_base = t.groupby("room")["co2_ppm"].transform(lambda s: s.rolling(144, min_periods=36).quantile(0.1))
    t["co2_excess"] = t["co2_ppm"] - co2_base
    t["co2_excess_2h"] = t.groupby("room")["co2_excess"].transform(lambda s: s.rolling(12, min_periods=1).max())
    t["co2_occupied"] = t["co2_excess_2h"] > CO2_EXCESS_PPM
    co2_known = t["co2_excess_2h"].notna()  # first ~6h of each room has no baseline yet
    # Flag only when BOTH signals say empty; where CO2 is unknown, do not flag.
    t["anomaly_hvac_no_occupancy"] = t["anomaly_hvac_no_motion"] & co2_known & ~t["co2_occupied"]

    # Rule: indoor temp far from setpoint while HVAC has been on for a while (possible fault).
    t["setpoint_gap"] = (t["V2"] - t["V12"]).abs()
    t["anomaly_setpoint_gap"] = (t["V4"] > 0) & (t["setpoint_gap"] > 3)

    t["anomaly_type"] = t.apply(classify_room_row, axis=1)
    t["any_anomaly"] = t["anomaly_type"].notna()

    e = energy_table.copy()
    e["energy_residual_z"] = (e["energy_residual"] - e["energy_residual"].mean()) / e["energy_residual"].std()
    e["anomaly_energy_spike"] = e["energy_residual_z"] > 3

    t.to_parquet(MODELS_DIR / "temp_anomalies.parquet", index=False)
    e.to_parquet(MODELS_DIR / "energy_anomalies.parquet", index=False)
    return t, e


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    temp_model, temp_table, temp_metrics = train_temp_model()
    joblib.dump(temp_model, MODELS_DIR / "temp_model.joblib")
    temp_table.to_parquet(MODELS_DIR / "temp_table.parquet", index=False)

    energy_model, energy_table, energy_metrics = train_energy_model()
    joblib.dump(energy_model, MODELS_DIR / "energy_model.joblib")
    energy_table.to_parquet(MODELS_DIR / "energy_table.parquet", index=False)

    prophet_model, prophet_comparison, prophet_metrics = train_prophet_baseline(energy_table)
    prophet_comparison.to_parquet(MODELS_DIR / "prophet_comparison.parquet", index=False)

    temp_anoms, energy_anoms = compute_anomalies(temp_table, energy_table)

    ablation = run_occupancy_ablation()

    metrics = {
        "temp_model": temp_metrics,
        "energy_model_xgboost": energy_metrics,
        "energy_model_prophet_baseline": prophet_metrics,
        "occupancy_ablation": ablation,
        "n_temp_anomalies": int(temp_anoms["any_anomaly"].sum()),
        "n_energy_anomalies": int(energy_anoms["anomaly_energy_spike"].sum()),
    }
    import json

    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
