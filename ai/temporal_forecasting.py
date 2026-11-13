"""
Temporal Crowd Forecasting Module (Stage 2)
MSc Mini Project: AI-Based Crowd Density and Occupancy Monitoring

This module implements time-series forecasting on historical crowd counts
captured at fixed intervals from a live video stream.

Methods implemented:
1. Holt's Linear Exponential Smoothing (Double Exponential Smoothing)
   - Tracks both current crowd level and growth/decay trend
2. Linear Trend Regression (OLS)
   - Calculates crowd velocity (people/minute rate of change) and R² confidence
3. Adaptive Future Crowd Density Prediction
   - Forecasts future crowd counts, future occupancy %, future density level,
     future occupancy status, multi-step horizon schedules, and headroom ETA.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def get_density_level(count: int) -> str:
    """Classifies crowd count into density level matching detector/views.py."""
    if count < 50:
        return "Low"
    elif count < 200:
        return "Medium"
    elif count < 500:
        return "High"
    else:
        return "Very High"


def get_occupancy_status(occupancy_pct: float) -> str:
    """Classifies occupancy percentage into status matching detector/views.py."""
    if occupancy_pct < 50:
        return "Low Occupancy"
    elif occupancy_pct < 75:
        return "Moderate Occupancy"
    elif occupancy_pct < 90:
        return "High Occupancy"
    elif occupancy_pct <= 100:
        return "Near Capacity"
    else:
        return "Over Capacity"


def holt_linear_forecast(
    series: List[float],
    future_steps: int = 4,
    alpha: float = 0.4,
    beta: float = 0.2
) -> List[float]:
    """
    Holt's Linear Exponential Smoothing.
    Level:  L_t = alpha * y_t + (1 - alpha) * (L_{t-1} + T_{t-1})
    Trend:  T_t = beta * (level - last_level) + (1 - beta) * last_trend
    Forecast: y_{t+h} = L_t + h * T_t
    """
    n = len(series)
    if n == 0:
        return [0.0] * future_steps
    if n == 1:
        return [float(series[0])] * future_steps

    # Initialize level and trend
    level = series[0]
    trend = series[1] - series[0]

    for t in range(1, n):
        y_val = series[t]
        last_level = level
        last_trend = trend

        level = alpha * y_val + (1.0 - alpha) * (last_level + last_trend)
        trend = beta * (level - last_level) + (1.0 - beta) * last_trend

    # Generate multi-step forecasts
    predictions = []
    for h in range(1, future_steps + 1):
        # Damping trend slightly for longer horizons
        damped_h = h * (0.95 ** (h - 1))
        pred = level + damped_h * trend
        pred = max(0.0, pred)  # Crowd count cannot be negative
        predictions.append(pred)

    return predictions


def linear_trend_analysis(
    series: List[float],
    interval_seconds: float = 3.0
) -> Dict[str, Any]:
    """
    Computes linear regression slope (rate of change) and R² fit.
    Guarded against zero counts and preliminary noisy frames.
    """
    n = len(series)
    if n < 3:
        return {
            "slope_per_sec": 0.0,
            "slope_per_min": 0.0,
            "r_squared": 1.0 if (n > 0 and series[0] == 0) else 0.5,
            "direction": "Establishing Baseline" if n > 0 else "No Data"
        }

    # If all series values are zero (empty scene):
    if all(val == 0.0 for val in series):
        return {
            "slope_per_sec": 0.0,
            "slope_per_min": 0.0,
            "r_squared": 1.0,
            "direction": "Stable (Clear Space)"
        }

    # Use sliding window of the last min(n, 12) points for current trend
    window_size = min(n, 12)
    y = np.array(series[-window_size:], dtype=np.float64)
    x = np.arange(window_size, dtype=np.float64) * interval_seconds

    # OLS fit: y = m*x + c
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx > 1e-8:
        slope_per_sec = float(ss_xy / ss_xx)
    else:
        slope_per_sec = 0.0

    if ss_yy > 1e-8 and ss_xx > 1e-8:
        r_squared = float(min(1.0, max(0.0, (ss_xy ** 2) / (ss_xx * ss_yy))))
    else:
        r_squared = 1.0 if np.all(y == y[0]) else 0.0

    slope_per_min = slope_per_sec * 60.0

    # Trend Direction classification
    if slope_per_min > 8.0:
        direction = "Rapidly Accumulating"
    elif slope_per_min > 2.0:
        direction = "Gradually Increasing"
    elif slope_per_min < -8.0:
        direction = "Rapidly Dispersing"
    elif slope_per_min < -2.0:
        direction = "Gradually Decreasing"
    else:
        direction = "Stable / Steady"

    return {
        "slope_per_sec": slope_per_sec,
        "slope_per_min": slope_per_min,
        "r_squared": round(r_squared, 3),
        "direction": direction
    }


def forecast_crowd_series(
    counts: List[int],
    max_capacity: int = 100,
    interval_seconds: float = 3.0,
    future_steps: int = 4
) -> Dict[str, Any]:
    """
    Stage 2 Master Forecasting Function:
    Inputs:
        counts: List of historical crowd counts (ordered chronologically)
        max_capacity: Maximum space capacity
        interval_seconds: Fixed capture interval in seconds
        future_steps: Number of future intervals to forecast (default 4)
    Returns:
        Structured dictionary containing future crowd predictions,
        future occupancy %, future density status, and trend analysis.
    """
    if not counts:
        counts = [0]

    current_count = int(counts[-1])
    float_counts = [float(c) for c in counts]
    n = len(float_counts)

    # 1. Holt's Double Exponential Smoothing Forecast
    raw_predictions = holt_linear_forecast(
        float_counts,
        future_steps=future_steps,
        alpha=0.4,
        beta=0.2
    )

    # If series is all 0, ensure raw_predictions are strictly 0
    if all(c == 0 for c in counts):
        raw_predictions = [0.0] * future_steps

    # 2. Trend & Velocity Analysis
    trend = linear_trend_analysis(float_counts, interval_seconds=interval_seconds)

    # Build per-step future predictions
    step_forecasts = []
    for idx, pred_val in enumerate(raw_predictions):
        step_num = idx + 1
        horizon_sec = int(round(step_num * interval_seconds))
        pred_int = int(round(pred_val))

        # Future occupancy %
        if max_capacity > 0:
            future_occ = (pred_int / max_capacity) * 100.0
        else:
            future_occ = 0.0

        d_level = get_density_level(pred_int)
        o_status = get_occupancy_status(future_occ)

        step_forecasts.append({
            "step": step_num,
            "horizon_seconds": horizon_sec,
            "time_label": f"+{horizon_sec}s",
            "predicted_count": pred_int,
            "occupancy_percentage": round(future_occ, 1),
            "density_level": d_level,
            "occupancy_status": o_status,
        })

    # Primary future prediction (next horizon step 1)
    primary_pred = step_forecasts[0]
    future_crowd_count = primary_pred["predicted_count"]
    future_occupancy_pct = primary_pred["occupancy_percentage"]
    future_density_level = primary_pred["density_level"]
    future_occupancy_status = primary_pred["occupancy_status"]

    # Extended prediction (at horizon end)
    extended_pred = step_forecasts[-1]

    # Time to Capacity calculation
    time_to_capacity_sec = None
    time_to_capacity_text = "N/A (Density Safe)"
    risk_alert = "Normal"

    # Only evaluate capacity breach warning if crowd count is actually significant
    # (e.g. current_count >= max_capacity * 0.4 or future_crowd_count >= max_capacity * 0.5)
    # AND slope is positively trending with sufficient data and reasonable fit
    if max_capacity > 0:
        if current_count >= max_capacity or future_crowd_count >= max_capacity:
            risk_alert = "Critical: Over Capacity"
            time_to_capacity_text = "Already At/Over Capacity"
        elif current_count >= max_capacity * 0.85 or future_crowd_count >= max_capacity * 0.85:
            risk_alert = "Warning: Crowd Approaching Capacity"
            if trend["slope_per_sec"] > 0.02 and n >= 3:
                diff = max_capacity - current_count
                sec_eta = diff / trend["slope_per_sec"]
                if sec_eta <= 300:
                    time_to_capacity_sec = int(round(sec_eta))
                    minutes = round(sec_eta / 60.0, 1)
                    time_to_capacity_text = f"~{time_to_capacity_sec} sec ({minutes} min)"
        elif n >= 3 and trend["r_squared"] >= 0.4 and trend["slope_per_sec"] > 0.05 and (current_count >= max_capacity * 0.4 or future_crowd_count >= max_capacity * 0.5):
            diff = max_capacity - current_count
            sec_eta = diff / trend["slope_per_sec"]
            if sec_eta <= 300:
                time_to_capacity_sec = int(round(sec_eta))
                minutes = round(sec_eta / 60.0, 1)
                time_to_capacity_text = f"~{time_to_capacity_sec} sec ({minutes} min)"
                if sec_eta <= 60:
                    risk_alert = "High Risk: Capacity Breach Imminent"
                else:
                    risk_alert = "Caution: Crowd Inflow Accelerating"
        else:
            time_to_capacity_text = "N/A (Safe Capacity Headroom)"
            risk_alert = "Normal"

    # Headroom (remaining capacity)
    capacity_headroom = max(0, max_capacity - future_crowd_count)

    # Narrative explanation of future crowd density
    if n < 2:
        density_summary = "Awaiting more captured frames to compute temporal crowd density forecast."
    elif current_count == 0 and future_crowd_count == 0:
        density_summary = f"No individuals detected across recent {n} captured frames. Projected crowd density remains Low with full safe capacity headroom ({max_capacity} people)."
    elif trend["slope_per_min"] > 3.0 and future_crowd_count > current_count:
        density_summary = f"Crowd density is actively increasing at +{trend['slope_per_min']:.1f} people/min. Next projection anticipates {future_crowd_count} people ({future_density_level} Density, {future_occupancy_pct}% occupancy)."
    elif trend["slope_per_min"] < -3.0:
        density_summary = f"Crowd is dispersing at {abs(trend['slope_per_min']):.1f} people/min. Future density is projected to ease to {future_crowd_count} people ({future_density_level} Density)."
    else:
        density_summary = f"Crowd density is holding steady across {n} captured frames. Projected future density is {future_density_level} (~{future_crowd_count} people, {future_occupancy_pct}% occupancy)."

    return {
        "model_name": "Holt's Linear Trend + OLS Velocity",
        "history_count": n,
        "current_count": current_count,
        "max_capacity": max_capacity,
        "interval_seconds": interval_seconds,
        "step_forecasts": step_forecasts,
        # Immediate next prediction:
        "future_crowd_count": future_crowd_count,
        "future_occupancy_percentage": future_occupancy_pct,
        "future_density_level": future_density_level,
        "future_occupancy_status": future_occupancy_status,
        # Extended future prediction:
        "extended_future_count": extended_pred["predicted_count"],
        "extended_horizon_seconds": extended_pred["horizon_seconds"],
        "extended_occupancy_percentage": extended_pred["occupancy_percentage"],
        "extended_occupancy_status": extended_pred["occupancy_status"],
        # Details & Analytics:
        "trend_direction": trend["direction"],
        "slope_per_min": round(trend["slope_per_min"], 2),
        "r_squared": trend["r_squared"],
        "time_to_capacity_text": time_to_capacity_text,
        "time_to_capacity_sec": time_to_capacity_sec,
        "risk_alert": risk_alert,
        "capacity_headroom": capacity_headroom,
        "density_summary": density_summary,
    }
