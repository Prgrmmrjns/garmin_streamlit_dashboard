#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Garmin Data Analysis
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta, date
import warnings
import os
from dotenv import load_dotenv
import garminconnect

warnings.filterwarnings('ignore')
import subprocess
import sys
import json
import requests
import uuid
from typing import Optional

# Load environment variables — same behaviour locally and in cloud.
# Local: .env is loaded here. Cloud: use Streamlit Secrets (injected into env below).
load_dotenv()

# DEBUG: Use session_state to persist debug log across reruns
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []

def _log(msg):
    st.session_state.debug_log.append(msg)

_log(f"[1] load_dotenv() called")
_log(f"[2] GARMIN_EMAIL from .env: {'SET' if os.getenv('GARMIN_EMAIL') else 'NOT SET'}")
_log(f"[3] GARMIN_PASSWORD from .env: {'SET' if os.getenv('GARMIN_PASSWORD') else 'NOT SET'}")

try:
    if hasattr(st, "secrets") and st.secrets:
        _log(f"[4] st.secrets exists with keys: {list(st.secrets.keys())}")
        for key, value in st.secrets.items():
            if isinstance(value, str) and value.strip():
                os.environ[key] = value.strip().strip('"')
                _log(f"[5] Copied st.secrets['{key}'] to os.environ (len={len(value)})")
    else:
        _log(f"[4] st.secrets is empty or doesn't exist")
except Exception as e:
    _log(f"[4] Error reading st.secrets: {e}")

_email = os.getenv('GARMIN_EMAIL', '')
_pass = os.getenv('GARMIN_PASSWORD', '')
_log(f"[6] Final GARMIN_EMAIL: {_email[:3]}***{_email[-8:] if len(_email) > 10 else ''} (len={len(_email)})" if _email else "[6] Final GARMIN_EMAIL: NOT SET")
_log(f"[7] Final GARMIN_PASSWORD: {'SET (len=' + str(len(_pass)) + ')' if _pass else 'NOT SET'}")

# Mistral AI Configuration
CHATBOT_NAME = os.getenv("CHATBOT_NAME", "Lixxi")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

def get_mistral_api_key():
    """Get Mistral API key from session state or environment."""
    return st.session_state.get("mistral_api_key", "") or os.getenv("MISTRAL_AI_API_KEY", "")

# Page configuration
st.set_page_config(
    page_title="Garmin Data Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GARMIN CREDENTIALS LOGIN ---

def check_credentials():
    """Returns `True` if the user has entered valid Garmin credentials."""
    
    def credentials_entered():
        """Attempts to login with entered credentials."""
        email = st.session_state.get("login_email", "").strip()
        password = st.session_state.get("login_password", "").strip()
        mistral_key = st.session_state.get("login_mistral_key", "").strip()
        
        if not email or not password:
            st.session_state["login_error"] = "Please enter both email and password."
            return
        
        # Try to authenticate with Garmin
        try:
            client = garminconnect.Garmin(email=email, password=password)
            client.login()
            # Success! Store credentials in session state
            st.session_state["garmin_email"] = email
            st.session_state["garmin_password"] = password
            st.session_state["credentials_valid"] = True
            st.session_state["login_error"] = None
            # Also set in os.environ so existing code works
            os.environ["GARMIN_EMAIL"] = email
            os.environ["GARMIN_PASSWORD"] = password
            # Store Mistral API key if provided
            if mistral_key:
                st.session_state["mistral_api_key"] = mistral_key
                os.environ["MISTRAL_AI_API_KEY"] = mistral_key
            # Clean up login fields
            if "login_email" in st.session_state:
                del st.session_state["login_email"]
            if "login_password" in st.session_state:
                del st.session_state["login_password"]
            if "login_mistral_key" in st.session_state:
                del st.session_state["login_mistral_key"]
        except Exception as e:
            st.session_state["credentials_valid"] = False
            st.session_state["login_error"] = f"Garmin login failed: {str(e)}"

    # Check if already authenticated
    if st.session_state.get("credentials_valid", False):
        return True
    
    # Show login form
    st.markdown("## 🔐 Garmin Dashboard Login")
    st.markdown("Enter your **Garmin Connect** credentials to access the dashboard.")
    
    with st.form("login_form"):
        st.text_input(
            "Garmin Email:", 
            key="login_email",
            placeholder="your.email@example.com"
        )
        st.text_input(
            "Garmin Password:", 
            type="password", 
            key="login_password"
        )
        
        st.markdown("---")
        st.markdown("**Optional:** Enter your Mistral AI API key for AI chatbot features.")
        st.text_input(
            "Mistral AI API Key (optional):", 
            type="password", 
            key="login_mistral_key",
            placeholder="sk-..."
        )
        
        submitted = st.form_submit_button("🔑 Login", use_container_width=True)
        if submitted:
            credentials_entered()
    
    # Show error if login failed
    if st.session_state.get("login_error"):
        st.error(f"❌ {st.session_state['login_error']}")
    
    st.info("💡 Your credentials are used to fetch data from Garmin Connect and are not stored permanently.")
    
    return st.session_state.get("credentials_valid", False)

# Check credentials before showing anything
if not check_credentials():
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TIME_SERIES_DIR = DATA_DIR / "time_series"
DAILY_DIR = DATA_DIR / "daily"
WORKOUTS_DIR = DATA_DIR / "workouts"
REPORTS_DIR = BASE_DIR / "reports"


def check_and_update_data():
    """
    Check if data needs updating and run collection/analysis if needed.
    Returns True if update was performed.
    """
    # Check if data exists
    daily_csv = DAILY_DIR / "daily_summary.csv"
    
    if not daily_csv.exists():
        return False
    
    # Check last data date
    try:
        df = pd.read_csv(daily_csv)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max().date()
            today = date.today()
            
            # If last date is before yesterday, we need to update
            yesterday = today - timedelta(days=1)
            
            if last_date < yesterday:
                return True
    except:
        pass
    
    return False


# --- GARMIN DATA COLLECTION (integrated) ---

def _ensure_data_dirs():
    """Create data directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TIME_SERIES_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    WORKOUTS_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_time_series_data(garmin_client, date_str):
    """Fetch all time-series data for a specific date."""
    time_series_data = {}
    
    # Heart Rate
    try:
        hr_data = garmin_client.get_heart_rates(date_str)
        if hr_data and 'heartRateValues' in hr_data:
            values = hr_data['heartRateValues']
            if values:
                df = pd.DataFrame(values, columns=['timestamp_ms', 'heart_rate'])
                df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                df['date'] = date_str
                time_series_data['heart_rate'] = df[['timestamp_ms', 'datetime', 'date', 'heart_rate']]
    except Exception:
        pass
    
    # Stress and Body Battery
    try:
        stress_data = garmin_client.get_all_day_stress(date_str)
        if stress_data:
            if 'stressValuesArray' in stress_data:
                values = stress_data['stressValuesArray']
                if values:
                    df = pd.DataFrame(values, columns=['timestamp_ms', 'stress_level'])
                    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                    df['date'] = date_str
                    time_series_data['stress'] = df[['timestamp_ms', 'datetime', 'date', 'stress_level']]
            if 'bodyBatteryValuesArray' in stress_data:
                values = stress_data['bodyBatteryValuesArray']
                if values:
                    df = pd.DataFrame(values)
                    if len(df.columns) >= 3:
                        df.columns = ['timestamp_ms', 'status', 'body_battery', 'unknown'] if len(df.columns) == 4 else ['timestamp_ms', 'status', 'body_battery']
                        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                        df['date'] = date_str
                        time_series_data['body_battery'] = df[['timestamp_ms', 'datetime', 'date', 'body_battery']]
    except Exception:
        pass
    
    # Respiration
    try:
        resp_data = garmin_client.get_respiration_data(date_str)
        if resp_data and 'respirationValuesArray' in resp_data:
            values = resp_data['respirationValuesArray']
            if values:
                df = pd.DataFrame(values, columns=['timestamp_ms', 'respiration_rate'])
                df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                df['date'] = date_str
                time_series_data['respiration'] = df[['timestamp_ms', 'datetime', 'date', 'respiration_rate']]
    except Exception:
        pass
    
    # SpO2
    try:
        spo2_data = garmin_client.get_spo2_data(date_str)
        if spo2_data:
            if 'spO2SingleValues' in spo2_data and spo2_data['spO2SingleValues']:
                values = spo2_data['spO2SingleValues']
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], list) and len(values[0]) >= 2:
                        df = pd.DataFrame(values)
                        if len(df.columns) >= 2:
                            df.columns = ['timestamp_ms', 'spo2'] + [f'col_{i}' for i in range(2, len(df.columns))]
                            df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                            df['date'] = date_str
                            time_series_data['spo2'] = df[['timestamp_ms', 'datetime', 'date', 'spo2']]
            elif 'continuousReadingDTOList' in spo2_data and spo2_data['continuousReadingDTOList']:
                values = spo2_data['continuousReadingDTOList']
                if isinstance(values, list) and len(values) > 0:
                    records = []
                    for v in values:
                        if isinstance(v, dict):
                            if 'timestamp' in v and 'value' in v:
                                records.append([v.get('timestamp'), v.get('value')])
                        elif isinstance(v, list) and len(v) >= 2:
                            records.append(v[:2])
                    if records:
                        df = pd.DataFrame(records, columns=['timestamp_ms', 'spo2'])
                        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                        df['date'] = date_str
                        time_series_data['spo2'] = df[['timestamp_ms', 'datetime', 'date', 'spo2']]
    except Exception:
        pass
    
    return time_series_data


def _fetch_daily_summary(garmin_client, date_str):
    """Fetch aggregated daily data for a specific date."""
    data = {
        "date": date_str,
        "steps": None, "distance_km": None, "floors_ascended": None, "floors_descended": None,
        "highly_active_min": None, "active_min": None, "sedentary_min": None,
        "moderate_intensity_min": None, "vigorous_intensity_min": None,
        "total_calories": None, "active_calories": None, "bmr_calories": None, "remaining_calories": None,
        "min_heart_rate": None, "max_heart_rate": None, "resting_heart_rate": None, "avg_resting_heart_rate_7d": None,
        "hrv": None,
        "sleep_hours": None, "deep_sleep_min": None, "light_sleep_min": None, "rem_sleep_min": None, "awake_sleep_min": None,
        "avg_stress": None, "max_stress": None, "stress_duration_min": None, "rest_stress_min": None,
        "activity_stress_min": None, "stress_percentage": None,
        "body_battery_charged": None, "body_battery_drained": None, "body_battery_max": None, "body_battery_most_recent": None,
        "avg_spo2": None, "min_spo2": None, "max_spo2": None,
        "avg_respiration": None, "min_respiration": None, "max_respiration": None,
        "hydration_ml": None,
    }
    
    try:
        stats = garmin_client.get_stats(date_str)
        if stats:
            data["steps"] = stats.get('totalSteps')
            distance_meters = stats.get('totalDistanceMeters')
            if distance_meters:
                data["distance_km"] = round(distance_meters / 1000, 2)
            data["floors_ascended"] = stats.get('floorsAscended')
            data["floors_descended"] = stats.get('floorsDescended')
            highly_active_sec = stats.get('highlyActiveSeconds')
            if highly_active_sec:
                data["highly_active_min"] = round(highly_active_sec / 60, 1)
            active_sec = stats.get('activeSeconds')
            if active_sec:
                data["active_min"] = round(active_sec / 60, 1)
            sedentary_sec = stats.get('sedentarySeconds')
            if sedentary_sec:
                data["sedentary_min"] = round(sedentary_sec / 60, 1)
            data["moderate_intensity_min"] = stats.get('moderateIntensityMinutes')
            data["vigorous_intensity_min"] = stats.get('vigorousIntensityMinutes')
            data["total_calories"] = stats.get('totalKilocalories')
            data["active_calories"] = stats.get('activeKilocalories')
            data["bmr_calories"] = stats.get('bmrKilocalories')
            data["remaining_calories"] = stats.get('remainingKilocalories')
            data["min_heart_rate"] = stats.get('minHeartRate')
            data["max_heart_rate"] = stats.get('maxHeartRate')
            data["resting_heart_rate"] = stats.get('restingHeartRate')
            data["avg_resting_heart_rate_7d"] = stats.get('lastSevenDaysAvgRestingHeartRate')
            data["avg_stress"] = stats.get('averageStressLevel')
            data["max_stress"] = stats.get('maxStressLevel')
            stress_duration_sec = stats.get('stressDuration')
            if stress_duration_sec:
                data["stress_duration_min"] = round(stress_duration_sec / 60, 1)
            rest_stress_sec = stats.get('restStressDuration')
            if rest_stress_sec:
                data["rest_stress_min"] = round(rest_stress_sec / 60, 1)
            activity_stress_sec = stats.get('activityStressDuration')
            if activity_stress_sec:
                data["activity_stress_min"] = round(activity_stress_sec / 60, 1)
            data["stress_percentage"] = stats.get('stressPercentage')
    except Exception:
        pass
    
    try:
        hrv_data = garmin_client.get_hrv_data(date_str)
        if hrv_data and isinstance(hrv_data, dict):
            hrv_value = (
                hrv_data.get('hrvSummary', {}).get('weeklyAvg') or
                hrv_data.get('hrvSummary', {}).get('lastNightAvg') or
                hrv_data.get('weeklyAvg') or hrv_data.get('lastNightAvg') or
                hrv_data.get('value') or None
            )
            if hrv_value:
                data["hrv"] = float(hrv_value)
    except Exception:
        pass
    
    try:
        sleep_data = garmin_client.get_sleep_data(date_str)
        if sleep_data and isinstance(sleep_data, dict):
            daily_sleep = sleep_data.get('dailySleepDTO')
            if daily_sleep:
                sleep_seconds = daily_sleep.get('sleepTimeSeconds')
                if sleep_seconds:
                    data["sleep_hours"] = round(sleep_seconds / 3600, 2)
                deep_sleep_sec = daily_sleep.get('deepSleepSeconds')
                if deep_sleep_sec:
                    data["deep_sleep_min"] = round(deep_sleep_sec / 60, 1)
                light_sleep_sec = daily_sleep.get('lightSleepSeconds')
                if light_sleep_sec:
                    data["light_sleep_min"] = round(light_sleep_sec / 60, 1)
                rem_sleep_sec = daily_sleep.get('remSleepSeconds')
                if rem_sleep_sec:
                    data["rem_sleep_min"] = round(rem_sleep_sec / 60, 1)
                awake_sleep_sec = daily_sleep.get('awakeSleepSeconds')
                if awake_sleep_sec:
                    data["awake_sleep_min"] = round(awake_sleep_sec / 60, 1)
    except Exception:
        pass
    
    try:
        body_battery = garmin_client.get_body_battery(date_str)
        if body_battery and isinstance(body_battery, dict):
            data["body_battery_charged"] = body_battery.get('charged')
            data["body_battery_drained"] = body_battery.get('drained')
            data["body_battery_max"] = body_battery.get('max')
            data["body_battery_most_recent"] = body_battery.get('mostRecent')
    except Exception:
        pass
    
    try:
        spo2_data = garmin_client.get_spo2_data(date_str)
        if spo2_data and isinstance(spo2_data, dict):
            data["avg_spo2"] = spo2_data.get('averageSpO2')
            data["min_spo2"] = spo2_data.get('lowestSpO2')
            data["max_spo2"] = spo2_data.get('averageSpO2')
    except Exception:
        pass
    
    try:
        resp_data = garmin_client.get_respiration_data(date_str)
        if resp_data and isinstance(resp_data, dict):
            data["avg_respiration"] = resp_data.get('avgWakingRespirationValue')
            data["min_respiration"] = resp_data.get('lowestRespirationValue')
            data["max_respiration"] = resp_data.get('highestRespirationValue')
    except Exception:
        pass
    
    try:
        hydration_data = garmin_client.get_hydration_data(date_str)
        if hydration_data and isinstance(hydration_data, dict):
            data["hydration_ml"] = hydration_data.get('valueInML')
    except Exception:
        pass
    
    return data


def _save_time_series_data(metric_name, df, date_str):
    """Append time-series data to CSV file."""
    csv_path = TIME_SERIES_DIR / f"{metric_name}.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        if 'date' in existing_df.columns and date_str in existing_df['date'].values:
            return False
    df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    return True


def _save_daily_summary(data):
    """Append daily summary data to CSV file."""
    csv_path = DAILY_DIR / "daily_summary.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        if 'date' in existing_df.columns and data['date'] in existing_df['date'].values:
            return False
    df = pd.DataFrame([data])
    df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    return True


def _fetch_workouts(garmin_client, start_date, end_date):
    """Fetch all workouts/activities between start_date and end_date."""
    workouts = []
    try:
        activities = garmin_client.get_activities_by_date(start_date, end_date)
        if activities:
            for activity in activities:
                workout = {
                    "activity_id": activity.get('activityId'),
                    "activity_name": activity.get('activityName'),
                    "activity_type": activity.get('activityType', {}).get('typeKey') if isinstance(activity.get('activityType'), dict) else activity.get('activityType'),
                    "start_time": activity.get('startTimeLocal'),
                    "start_time_gmt": activity.get('startTimeGMT'),
                    "duration_seconds": activity.get('duration'),
                    "duration_minutes": round(activity.get('duration', 0) / 60, 1) if activity.get('duration') else None,
                    "distance_meters": activity.get('distance'),
                    "distance_km": round(activity.get('distance', 0) / 1000, 2) if activity.get('distance') else None,
                    "calories": activity.get('calories'),
                    "avg_heart_rate": activity.get('averageHR'),
                    "max_heart_rate": activity.get('maxHR'),
                    "avg_speed": activity.get('averageSpeed'),
                    "max_speed": activity.get('maxSpeed'),
                    "avg_pace_min_per_km": None,
                    "elevation_gain": activity.get('elevationGain'),
                    "elevation_loss": activity.get('elevationLoss'),
                    "steps": activity.get('steps'),
                    "avg_cadence": activity.get('averageRunningCadenceInStepsPerMinute') or activity.get('avgStrideLength'),
                    "max_cadence": activity.get('maxRunningCadenceInStepsPerMinute'),
                    "avg_power": activity.get('avgPower'),
                    "max_power": activity.get('maxPower'),
                    "training_effect_aerobic": activity.get('aerobicTrainingEffect'),
                    "training_effect_anaerobic": activity.get('anaerobicTrainingEffect'),
                    "vo2max": activity.get('vO2MaxValue'),
                    "avg_stress": activity.get('avgStress'),
                    "device_name": activity.get('deviceId'),
                    "location_name": activity.get('locationName'),
                    "description": activity.get('description'),
                }
                if workout['distance_km'] and workout['duration_minutes'] and workout['distance_km'] > 0:
                    pace = workout['duration_minutes'] / workout['distance_km']
                    workout['avg_pace_min_per_km'] = round(pace, 2)
                workouts.append(workout)
    except Exception:
        pass
    return workouts


def _save_workouts(workouts):
    """Save workouts to CSV file, avoiding duplicates by activity_id."""
    csv_path = WORKOUTS_DIR / "workouts.csv"
    if not workouts:
        return 0
    new_workouts = workouts
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        if 'activity_id' in existing_df.columns:
            existing_ids = set(existing_df['activity_id'].values)
            new_workouts = [w for w in workouts if w['activity_id'] not in existing_ids]
    if not new_workouts:
        return 0
    df = pd.DataFrame(new_workouts)
    df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    return len(new_workouts)


def _get_collected_dates():
    """Get list of dates that already have daily summary data."""
    csv_path = DAILY_DIR / "daily_summary.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if 'date' in df.columns:
                return set(df['date'].unique())
        except Exception:
            pass
    return set()


def run_data_update(days_back: int = 14):
    """
    Collect data from Garmin Connect and save to CSV files.
    Credentials are read from os.environ (populated from .env locally or st.secrets in cloud).
    """
    _log(f"[RUN] run_data_update() called with days_back={days_back}")
    
    _ensure_data_dirs()
    _log(f"[RUN] Data dirs ensured: {DATA_DIR.exists()}, {DAILY_DIR.exists()}")
    
    # Get credentials directly from os.environ
    garmin_email = os.getenv("GARMIN_EMAIL", "").strip().strip('"')
    garmin_password = os.getenv("GARMIN_PASSWORD", "").strip().strip('"')
    
    _log(f"[RUN] Email from env: {'SET (len=' + str(len(garmin_email)) + ')' if garmin_email else 'EMPTY'}")
    _log(f"[RUN] Password from env: {'SET (len=' + str(len(garmin_password)) + ')' if garmin_password else 'EMPTY'}")
    
    if not garmin_email or not garmin_password:
        _log(f"[RUN] FAILED: Missing credentials")
        st.error("Missing Garmin credentials. Please log out and log in again.")
        return False
    
    # Connect to Garmin
    try:
        _log(f"[RUN] Attempting Garmin login...")
        garmin = garminconnect.Garmin(email=garmin_email, password=garmin_password)
        garmin.login()
        _log(f"[RUN] Garmin login SUCCESS")
    except Exception as e:
        _log(f"[RUN] Garmin login FAILED: {str(e)}")
        st.error(f"Failed to connect to Garmin: {str(e)}")
        return False
    
    # Get already collected dates
    collected_dates = _get_collected_dates()
    _log(f"[RUN] Already collected dates: {len(collected_dates)}")
    
    # Calculate date range
    today = datetime.now().date()
    dates_to_collect = []
    for i in range(days_back):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        if date_str not in collected_dates:
            dates_to_collect.append(date_str)
    
    _log(f"[RUN] Dates to collect: {len(dates_to_collect)}")
    
    # Collect data for each date
    collected_count = 0
    for date_str in dates_to_collect:
        try:
            time_series_data = _fetch_time_series_data(garmin, date_str)
            for metric_name, df in time_series_data.items():
                _save_time_series_data(metric_name, df, date_str)
            daily_data = _fetch_daily_summary(garmin, date_str)
            if _save_daily_summary(daily_data):
                collected_count += 1
        except Exception as e:
            _log(f"[RUN] Error collecting {date_str}: {str(e)}")
    
    _log(f"[RUN] Collected {collected_count} new daily summaries")
    
    # Fetch workouts
    start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    workouts = _fetch_workouts(garmin, start_date, end_date)
    saved_workouts = _save_workouts(workouts)
    _log(f"[RUN] Workouts: found {len(workouts)}, saved {saved_workouts}")
    
    # Check if daily_summary.csv exists now
    daily_csv = DAILY_DIR / "daily_summary.csv"
    _log(f"[RUN] daily_summary.csv exists: {daily_csv.exists()}")
    if daily_csv.exists():
        try:
            df = pd.read_csv(daily_csv)
            _log(f"[RUN] daily_summary.csv has {len(df)} rows")
        except Exception as e:
            _log(f"[RUN] Error reading daily_summary.csv: {e}")
    
    # Run analysis script if it exists
    env = os.environ.copy()
    analyze_script = BASE_DIR / "analyze_garmin_data.py"
    if analyze_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(analyze_script)],
                check=False, capture_output=True, env=env, cwd=str(BASE_DIR),
            )
        except Exception:
            pass
    
    _log(f"[RUN] run_data_update() completed")
    return True


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_insights():
    """Load insights from JSON file."""
    insights_file = REPORTS_DIR / "insights.json"
    
    if not insights_file.exists():
        # Try to generate insights
        analyze_script = BASE_DIR / "analyze_garmin_data.py"
        if analyze_script.exists():
            try:
                subprocess.run(["python3", str(analyze_script)], check=False, capture_output=True, timeout=60)
            except:
                pass
    
    if insights_file.exists():
        try:
            with open(insights_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return None


# --- GOALS AND CHALLENGES ---
GOALS_FILE = DATA_DIR / "goals.json"

DEFAULT_GOALS = {
    "daily_steps": {
        "name": "Daily Steps",
        "target": 10000,
        "unit": "steps",
        "enabled": True,
        "icon": "👣"
    },
    "weekly_workouts": {
        "name": "Weekly Workouts",
        "target": 3,
        "unit": "workouts",
        "enabled": True,
        "icon": "🏋️"
    },
    "daily_sleep": {
        "name": "Daily Sleep",
        "target": 7.0,
        "unit": "hours",
        "enabled": True,
        "icon": "😴"
    },
    "weekly_active_minutes": {
        "name": "Weekly Active Minutes",
        "target": 150,
        "unit": "minutes",
        "enabled": True,
        "icon": "⚡"
    }
}

DEFAULT_CHALLENGES = []


def load_goals():
    """Load goals from JSON file."""
    if GOALS_FILE.exists():
        try:
            with open(GOALS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('goals', DEFAULT_GOALS), data.get('challenges', DEFAULT_CHALLENGES)
        except:
            pass
    return DEFAULT_GOALS.copy(), DEFAULT_CHALLENGES.copy()


def save_goals(goals, challenges):
    """Save goals to JSON file."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(GOALS_FILE, 'w') as f:
        json.dump({'goals': goals, 'challenges': challenges}, f, indent=2, default=str)


def calculate_goal_progress(goal_key, goals, daily_df, workouts_df, period='week'):
    """
    Calculate progress for a specific goal.
    Returns dict with progress info.
    """
    goal = goals.get(goal_key, {})
    if not goal.get('enabled', False):
        return None
    
    target = goal.get('target', 0)
    today = date.today()
    
    # Determine period dates
    if period == 'week':
        # Current week (Monday to Sunday)
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        period_start = start_of_week
        period_end = min(end_of_week, today)
        total_days = 7
        days_elapsed = (today - start_of_week).days + 1
    else:  # month
        start_of_month = today.replace(day=1)
        if today.month == 12:
            end_of_month = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_of_month = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        period_start = start_of_month
        period_end = min(end_of_month, today)
        total_days = (end_of_month - start_of_month).days + 1
        days_elapsed = (today - start_of_month).days + 1
    
    progress = {
        'goal_key': goal_key,
        'name': goal.get('name', goal_key),
        'target': target,
        'unit': goal.get('unit', ''),
        'icon': goal.get('icon', '📊'),
        'period': period,
        'period_start': period_start,
        'period_end': period_end,
        'days_elapsed': days_elapsed,
        'total_days': total_days
    }
    
    if goal_key == 'daily_steps':
        # Count days where steps >= target
        if daily_df is not None and not daily_df.empty and 'steps' in daily_df.columns:
            period_data = daily_df[
                (daily_df['date'].dt.date >= period_start) & 
                (daily_df['date'].dt.date <= period_end)
            ]
            days_achieved = len(period_data[period_data['steps'] >= target])
            avg_steps = period_data['steps'].mean() if len(period_data) > 0 else 0
            
            progress['days_achieved'] = days_achieved
            progress['days_with_data'] = len(period_data)
            progress['current_value'] = avg_steps
            progress['status'] = f"{days_achieved}/{days_elapsed} days"
            progress['percentage'] = (days_achieved / days_elapsed * 100) if days_elapsed > 0 else 0
            progress['achieved'] = days_achieved >= days_elapsed  # All days so far achieved
        else:
            progress['days_achieved'] = 0
            progress['days_with_data'] = 0
            progress['current_value'] = 0
            progress['status'] = "No data"
            progress['percentage'] = 0
            progress['achieved'] = False
    
    elif goal_key == 'weekly_workouts':
        # Count workouts this week
        if workouts_df is not None and not workouts_df.empty:
            period_workouts = workouts_df[
                (workouts_df['date'] >= period_start) & 
                (workouts_df['date'] <= period_end)
            ]
            workout_count = len(period_workouts)
            
            progress['current_value'] = workout_count
            progress['status'] = f"{workout_count}/{target} workouts"
            progress['percentage'] = min(100, (workout_count / target * 100)) if target > 0 else 0
            progress['achieved'] = workout_count >= target
        else:
            progress['current_value'] = 0
            progress['status'] = f"0/{target} workouts"
            progress['percentage'] = 0
            progress['achieved'] = False
    
    elif goal_key == 'daily_sleep':
        # Count days where sleep >= target
        if daily_df is not None and not daily_df.empty and 'sleep_hours' in daily_df.columns:
            period_data = daily_df[
                (daily_df['date'].dt.date >= period_start) & 
                (daily_df['date'].dt.date <= period_end)
            ]
            days_achieved = len(period_data[period_data['sleep_hours'] >= target])
            avg_sleep = period_data['sleep_hours'].mean() if len(period_data) > 0 else 0
            
            progress['days_achieved'] = days_achieved
            progress['days_with_data'] = len(period_data)
            progress['current_value'] = avg_sleep
            progress['status'] = f"{days_achieved}/{days_elapsed} nights"
            progress['percentage'] = (days_achieved / days_elapsed * 100) if days_elapsed > 0 else 0
            progress['achieved'] = days_achieved >= days_elapsed
        else:
            progress['days_achieved'] = 0
            progress['days_with_data'] = 0
            progress['current_value'] = 0
            progress['status'] = "No data"
            progress['percentage'] = 0
            progress['achieved'] = False
    
    elif goal_key == 'weekly_active_minutes':
        # Sum active minutes for the week
        if daily_df is not None and not daily_df.empty:
            period_data = daily_df[
                (daily_df['date'].dt.date >= period_start) & 
                (daily_df['date'].dt.date <= period_end)
            ]
            
            # Try different column names for active minutes
            active_cols = ['moderate_intensity_min', 'vigorous_intensity_min', 'active_min']
            total_active = 0
            
            for col in active_cols:
                if col in period_data.columns:
                    total_active += period_data[col].sum()
            
            # If no specific columns, estimate from steps
            if total_active == 0 and 'steps' in period_data.columns:
                # Rough estimate: 100 steps = 1 minute of activity
                total_active = period_data['steps'].sum() / 100
            
            progress['current_value'] = total_active
            progress['status'] = f"{int(total_active)}/{target} min"
            progress['percentage'] = min(100, (total_active / target * 100)) if target > 0 else 0
            progress['achieved'] = total_active >= target
        else:
            progress['current_value'] = 0
            progress['status'] = f"0/{target} min"
            progress['percentage'] = 0
            progress['achieved'] = False
    
    return progress


def calculate_challenge_progress(challenge, daily_df, workouts_df):
    """
    Calculate progress for a challenge.
    """
    today = date.today()
    start_date = datetime.strptime(challenge['start_date'], '%Y-%m-%d').date() if isinstance(challenge['start_date'], str) else challenge['start_date']
    end_date = datetime.strptime(challenge['end_date'], '%Y-%m-%d').date() if isinstance(challenge['end_date'], str) else challenge['end_date']
    
    # Check if challenge is active
    if today < start_date:
        status = 'upcoming'
    elif today > end_date:
        status = 'completed'
    else:
        status = 'active'
    
    challenge_type = challenge.get('type', 'steps')
    target = challenge.get('target', 0)
    
    progress = {
        'name': challenge.get('name', 'Challenge'),
        'type': challenge_type,
        'target': target,
        'start_date': start_date,
        'end_date': end_date,
        'status': status,
        'icon': challenge.get('icon', '🎯')
    }
    
    # Calculate progress based on challenge type
    if daily_df is not None and not daily_df.empty:
        period_data = daily_df[
            (daily_df['date'].dt.date >= start_date) & 
            (daily_df['date'].dt.date <= min(end_date, today))
        ]
        
        if challenge_type == 'total_steps':
            total = period_data['steps'].sum() if 'steps' in period_data.columns else 0
            progress['current_value'] = total
            progress['percentage'] = min(100, (total / target * 100)) if target > 0 else 0
            progress['achieved'] = total >= target
            progress['display'] = f"{int(total):,} / {int(target):,} steps"
        
        elif challenge_type == 'avg_sleep':
            avg = period_data['sleep_hours'].mean() if 'sleep_hours' in period_data.columns else 0
            progress['current_value'] = avg
            progress['percentage'] = min(100, (avg / target * 100)) if target > 0 else 0
            progress['achieved'] = avg >= target
            progress['display'] = f"{avg:.1f} / {target} hrs avg"
        
        elif challenge_type == 'workout_count':
            if workouts_df is not None and not workouts_df.empty:
                period_workouts = workouts_df[
                    (workouts_df['date'] >= start_date) & 
                    (workouts_df['date'] <= min(end_date, today))
                ]
                count = len(period_workouts)
            else:
                count = 0
            progress['current_value'] = count
            progress['percentage'] = min(100, (count / target * 100)) if target > 0 else 0
            progress['achieved'] = count >= target
            progress['display'] = f"{count} / {target} workouts"
        
        elif challenge_type == 'step_streak':
            # Days in a row with steps >= target
            streak = 0
            current_streak = 0
            sorted_data = period_data.sort_values('date')
            for _, row in sorted_data.iterrows():
                if row.get('steps', 0) >= 10000:  # Use 10k as default for streak
                    current_streak += 1
                    streak = max(streak, current_streak)
                else:
                    current_streak = 0
            
            progress['current_value'] = streak
            progress['percentage'] = min(100, (streak / target * 100)) if target > 0 else 0
            progress['achieved'] = streak >= target
            progress['display'] = f"{streak} / {target} day streak"
    else:
        progress['current_value'] = 0
        progress['percentage'] = 0
        progress['achieved'] = False
        progress['display'] = "No data"
    
    return progress


# --- LIXXI AI CHATBOT ---
CHATS_DIR = DATA_DIR / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)


def get_garmin_context(daily_df, workouts_df, days=7):
    """
    Generate a context summary of the user's Garmin data for the AI chatbot.
    """
    context_parts = []
    today = date.today()
    
    if daily_df is not None and not daily_df.empty:
        # Recent daily summary
        recent = daily_df[daily_df['date'].dt.date >= today - timedelta(days=days)].sort_values('date', ascending=False)
        
        if not recent.empty:
            latest = recent.iloc[0]
            context_parts.append(f"=== USER'S GARMIN HEALTH DATA (Last {days} Days) ===")
            context_parts.append(f"\nLatest data ({latest['date'].strftime('%Y-%m-%d')}):")
            
            if 'steps' in latest:
                context_parts.append(f"- Steps: {int(latest['steps']) if pd.notna(latest['steps']) else 'N/A'}")
            if 'sleep_hours' in latest:
                context_parts.append(f"- Sleep: {latest['sleep_hours']:.1f} hours" if pd.notna(latest['sleep_hours']) else "- Sleep: N/A")
            if 'resting_heart_rate' in latest:
                context_parts.append(f"- Resting Heart Rate: {int(latest['resting_heart_rate'])} bpm" if pd.notna(latest['resting_heart_rate']) else "- Resting Heart Rate: N/A")
            if 'avg_stress' in latest:
                context_parts.append(f"- Average Stress: {int(latest['avg_stress'])}" if pd.notna(latest['avg_stress']) else "- Average Stress: N/A")
            if 'body_battery_max' in latest:
                context_parts.append(f"- Body Battery Max: {int(latest['body_battery_max'])}" if pd.notna(latest['body_battery_max']) else "- Body Battery Max: N/A")
            if 'hrv' in latest:
                context_parts.append(f"- HRV: {int(latest['hrv'])}" if pd.notna(latest['hrv']) else "- HRV: N/A")
            if 'total_calories' in latest:
                context_parts.append(f"- Total Calories: {int(latest['total_calories'])}" if pd.notna(latest['total_calories']) else "- Total Calories: N/A")
            
            # Weekly averages
            context_parts.append(f"\nWeekly Averages ({len(recent)} days):")
            if 'steps' in recent.columns:
                context_parts.append(f"- Avg Steps: {int(recent['steps'].mean())}")
            if 'sleep_hours' in recent.columns:
                context_parts.append(f"- Avg Sleep: {recent['sleep_hours'].mean():.1f} hours")
            if 'resting_heart_rate' in recent.columns:
                context_parts.append(f"- Avg Resting HR: {recent['resting_heart_rate'].mean():.0f} bpm")
            if 'avg_stress' in recent.columns:
                context_parts.append(f"- Avg Stress: {recent['avg_stress'].mean():.0f}")
    
    if workouts_df is not None and not workouts_df.empty:
        # Recent workouts
        recent_workouts = workouts_df[workouts_df['date'] >= today - timedelta(days=days)].sort_values('date', ascending=False)
        
        if not recent_workouts.empty:
            context_parts.append(f"\nRecent Workouts ({len(recent_workouts)} in last {days} days):")
            for _, w in recent_workouts.head(5).iterrows():
                name = w.get('activity_name', 'Workout')
                dur = w.get('duration_minutes', 0)
                cal = w.get('calories', 0)
                w_date = w.get('date', '')
                date_str = w_date.strftime('%m/%d') if hasattr(w_date, 'strftime') else str(w_date)[:10]
                context_parts.append(f"- {name} on {date_str}: {int(dur)}min, {int(cal) if pd.notna(cal) else 0} calories")
    
    if not context_parts:
        return "No Garmin data available for context."
    
    return "\n".join(context_parts)


def load_chat_list():
    """Load list of all saved chats."""
    chats = []
    if CHATS_DIR.exists():
        for chat_file in sorted(CHATS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(chat_file, 'r') as f:
                    data = json.load(f)
                    chats.append({
                        'id': chat_file.stem,
                        'title': data.get('title', 'Untitled Chat'),
                        'created': data.get('created', ''),
                        'updated': data.get('updated', ''),
                        'message_count': len(data.get('messages', []))
                    })
            except:
                pass
    return chats


def load_chat(chat_id: str) -> Optional[dict]:
    """Load a specific chat by ID."""
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        try:
            with open(chat_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_chat(chat_id: str, title: str, messages: list):
    """Save a chat to disk."""
    chat_file = CHATS_DIR / f"{chat_id}.json"
    now = datetime.now().isoformat()
    
    # Load existing or create new
    if chat_file.exists():
        with open(chat_file, 'r') as f:
            data = json.load(f)
        data['messages'] = messages
        data['updated'] = now
        data['title'] = title
    else:
        data = {
            'id': chat_id,
            'title': title,
            'created': now,
            'updated': now,
            'messages': messages
        }
    
    with open(chat_file, 'w') as f:
        json.dump(data, f, indent=2)


def delete_chat(chat_id: str):
    """Delete a chat."""
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        chat_file.unlink()


def generate_chat_title(first_message: str) -> str:
    """Generate a title from the first message."""
    # Take first 40 chars of the message
    title = first_message[:40].strip()
    if len(first_message) > 40:
        title += "..."
    return title


# --- FUNCTION CALLING TOOLS FOR WORKOUT CREATION ---

WORKOUT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_workout",
            "description": "Create a new workout to be scheduled on Garmin Connect. Use this when the user asks to create, schedule, or plan a workout. The workout will be shown to the user for confirmation before being created.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workout_name": {
                        "type": "string",
                        "description": "Name of the workout (e.g., 'Morning Run', 'HIIT Session', 'Strength Training')"
                    },
                    "sport_type": {
                        "type": "string",
                        "enum": ["running", "cycling", "swimming", "strength_training", "walking", "hiking", "yoga", "other"],
                        "description": "Type of sport/activity"
                    },
                    "structure": {
                        "type": "string",
                        "enum": ["simple", "intervals", "custom"],
                        "description": "Workout structure type: 'simple' for basic time-based, 'intervals' for interval training, 'custom' for detailed step-by-step"
                    },
                    "total_duration_minutes": {
                        "type": "integer",
                        "description": "Total duration of workout in minutes (for simple workouts)"
                    },
                    "target_hr_zone": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                        "description": "Target heart rate zone (1-5, optional)"
                    },
                    "warmup_minutes": {
                        "type": "integer",
                        "description": "Warmup duration in minutes (for interval workouts)"
                    },
                    "interval_minutes": {
                        "type": "integer",
                        "description": "Interval duration in minutes (for interval workouts)"
                    },
                    "recovery_minutes": {
                        "type": "integer",
                        "description": "Recovery duration in minutes (for interval workouts)"
                    },
                    "cooldown_minutes": {
                        "type": "integer",
                        "description": "Cooldown duration in minutes (for interval workouts)"
                    },
                    "repeats": {
                        "type": "integer",
                        "description": "Number of interval repeats (for interval workouts)"
                    },
                    "interval_target_zone": {
                        "type": "integer",
                        "enum": [3, 4, 5],
                        "description": "Target HR zone for intervals (for interval workouts)"
                    },
                    "steps": {
                        "type": "array",
                        "description": "Custom workout steps (for custom structure)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_type": {
                                    "type": "string",
                                    "enum": ["warmup", "interval", "recovery", "cooldown", "rest"],
                                    "description": "Type of step"
                                },
                                "duration_minutes": {
                                    "type": "integer",
                                    "description": "Duration of step in minutes"
                                },
                                "target_hr_zone": {
                                    "type": "integer",
                                    "enum": [1, 2, 3, 4, 5],
                                    "description": "Target HR zone for this step (optional, 0 for no target)"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Optional description of the step"
                                }
                            },
                            "required": ["step_type", "duration_minutes"]
                        }
                    },
                    "schedule_date": {
                        "type": "string",
                        "description": "Date to schedule the workout (YYYY-MM-DD format, e.g., '2026-01-31'). Use today's date if not specified."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes or instructions for the workout"
                    }
                },
                "required": ["workout_name", "sport_type", "structure"]
            }
        }
    }
]


def build_workout_json(workout_params: dict) -> tuple:
    """
    Build the Garmin workout JSON from the function call parameters.
    Returns (workout_json, total_duration_seconds, readable_summary)
    """
    sport_type_map = {
        "running": {"sportTypeId": 1, "sportTypeKey": "running"},
        "cycling": {"sportTypeId": 2, "sportTypeKey": "cycling"},
        "swimming": {"sportTypeId": 5, "sportTypeKey": "swimming"},
        "strength_training": {"sportTypeId": 4, "sportTypeKey": "strength_training"},
        "walking": {"sportTypeId": 9, "sportTypeKey": "walking"},
        "hiking": {"sportTypeId": 17, "sportTypeKey": "hiking"},
        "yoga": {"sportTypeId": 43, "sportTypeKey": "yoga"},
        "other": {"sportTypeId": 0, "sportTypeKey": "other"},
    }
    
    step_type_map = {
        "warmup": {"stepTypeId": 1, "stepTypeKey": "warmup"},
        "interval": {"stepTypeId": 3, "stepTypeKey": "interval"},
        "recovery": {"stepTypeId": 4, "stepTypeKey": "recovery"},
        "cooldown": {"stepTypeId": 2, "stepTypeKey": "cooldown"},
        "rest": {"stepTypeId": 5, "stepTypeKey": "rest"},
    }
    
    workout_name = workout_params.get("workout_name", "AI Generated Workout")
    sport_type = sport_type_map.get(workout_params.get("sport_type", "other"), sport_type_map["other"])
    structure = workout_params.get("structure", "simple")
    
    workout_steps = []
    total_duration = 0
    summary_parts = [f"**{workout_name}** ({workout_params.get('sport_type', 'other').replace('_', ' ').title()})"]
    
    def get_target_type(hr_zone):
        if hr_zone and hr_zone > 0:
            return {
                "workoutTargetTypeId": 2,
                "workoutTargetTypeKey": "heart.rate.zone",
                "targetValueOne": hr_zone,
            }
        return {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"}
    
    if structure == "simple":
        duration_min = workout_params.get("total_duration_minutes", 30)
        target_zone = workout_params.get("target_hr_zone")
        
        workout_steps.append({
            "type": "ExecutableStepDTO",
            "stepOrder": 1,
            "stepType": step_type_map["interval"],
            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
            "endConditionValue": duration_min * 60,
            "targetType": get_target_type(target_zone),
        })
        total_duration = duration_min * 60
        summary_parts.append(f"- Duration: {duration_min} minutes")
        if target_zone:
            summary_parts.append(f"- Target: HR Zone {target_zone}")
    
    elif structure == "intervals":
        warmup = workout_params.get("warmup_minutes", 5)
        interval = workout_params.get("interval_minutes", 4)
        recovery = workout_params.get("recovery_minutes", 2)
        cooldown = workout_params.get("cooldown_minutes", 5)
        repeats = workout_params.get("repeats", 4)
        interval_zone = workout_params.get("interval_target_zone", 4)
        
        step_order = 1
        
        # Warmup
        if warmup > 0:
            workout_steps.append({
                "type": "ExecutableStepDTO",
                "stepOrder": step_order,
                "stepType": step_type_map["warmup"],
                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                "endConditionValue": warmup * 60,
                "targetType": get_target_type(2),  # Zone 2 for warmup
            })
            step_order += 1
            total_duration += warmup * 60
            summary_parts.append(f"- Warmup: {warmup} min (Zone 2)")
        
        # Interval/Recovery repeat group
        repeat_steps = []
        repeat_steps.append({
            "type": "ExecutableStepDTO",
            "stepOrder": 1,
            "stepType": step_type_map["interval"],
            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
            "endConditionValue": interval * 60,
            "targetType": get_target_type(interval_zone),
        })
        
        if recovery > 0:
            repeat_steps.append({
                "type": "ExecutableStepDTO",
                "stepOrder": 2,
                "stepType": step_type_map["recovery"],
                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                "endConditionValue": recovery * 60,
                "targetType": get_target_type(None),
            })
        
        workout_steps.append({
            "type": "RepeatGroupDTO",
            "stepOrder": step_order,
            "stepType": {"stepTypeId": 6, "stepTypeKey": "repeat"},
            "numberOfIterations": repeats,
            "workoutSteps": repeat_steps,
            "endCondition": {"conditionTypeId": 7, "conditionTypeKey": "iterations"},
            "endConditionValue": float(repeats),
        })
        step_order += 1
        total_duration += (interval + recovery) * 60 * repeats
        summary_parts.append(f"- Intervals: {repeats}x ({interval} min hard @ Zone {interval_zone} + {recovery} min recovery)")
        
        # Cooldown
        if cooldown > 0:
            workout_steps.append({
                "type": "ExecutableStepDTO",
                "stepOrder": step_order,
                "stepType": step_type_map["cooldown"],
                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                "endConditionValue": cooldown * 60,
                "targetType": get_target_type(None),
            })
            total_duration += cooldown * 60
            summary_parts.append(f"- Cooldown: {cooldown} min")
    
    elif structure == "custom":
        steps = workout_params.get("steps", [])
        for i, step in enumerate(steps):
            step_type_key = step.get("step_type", "interval")
            duration = step.get("duration_minutes", 5)
            target_zone = step.get("target_hr_zone")
            
            workout_steps.append({
                "type": "ExecutableStepDTO",
                "stepOrder": i + 1,
                "stepType": step_type_map.get(step_type_key, step_type_map["interval"]),
                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                "endConditionValue": duration * 60,
                "targetType": get_target_type(target_zone),
            })
            total_duration += duration * 60
            zone_str = f" @ Zone {target_zone}" if target_zone else ""
            summary_parts.append(f"- {step_type_key.title()}: {duration} min{zone_str}")
    
    summary_parts.append(f"\n**Total Duration:** {total_duration // 60} minutes")
    
    schedule_date = workout_params.get("schedule_date", date.today().isoformat())
    summary_parts.append(f"**Scheduled for:** {schedule_date}")
    
    if workout_params.get("notes"):
        summary_parts.append(f"\n**Notes:** {workout_params['notes']}")
    
    workout_json = {
        "workoutName": workout_name,
        "sportType": sport_type,
        "workoutSegments": [{
            "segmentOrder": 1,
            "sportType": sport_type,
            "workoutSteps": workout_steps
        }],
        "estimatedDurationInSecs": int(total_duration),
    }
    
    return workout_json, total_duration, "\n".join(summary_parts)


def call_mistral_api(messages: list, system_prompt: str, garmin_context: str, enable_tools: bool = False) -> dict:
    """
    Call Mistral AI API with the conversation history.
    Returns a dict with 'content' and optionally 'tool_calls'.
    """
    mistral_key = get_mistral_api_key()
    if not mistral_key:
        return {"content": "❌ Mistral API key not configured. Please log out and log back in with your Mistral API key to use the AI assistant.", "tool_calls": None}
    
    # Build the full system prompt with Garmin context
    tool_instructions = ""
    if enable_tools:
        tool_instructions = """

IMPORTANT - WORKOUT CREATION:
When the user asks you to create, schedule, or plan a workout, use the create_workout function. 
- Always ask clarifying questions if needed (workout type, duration, intensity, date)
- For interval training, suggest appropriate warmup, intervals, recovery, and cooldown
- Default to today's date if no date is specified
- After creating a workout, explain what you've created and wait for user confirmation"""

    full_system = f"""You are {CHATBOT_NAME}, a friendly and knowledgeable AI fitness assistant. You have access to the user's Garmin fitness data and can provide personalized health and fitness advice.

Your personality:
- Friendly, encouraging, and supportive
- Knowledgeable about fitness, nutrition, sleep, and recovery
- Give specific advice based on the user's actual data
- Use emojis occasionally to be engaging
- Be concise but helpful

{garmin_context}

Based on this data, you can:
- Analyze their activity levels and suggest improvements
- Comment on their sleep patterns and recovery
- Suggest workout plans based on their current fitness level
- Explain health metrics like HRV, Body Battery, stress levels
- Motivate them to reach their fitness goals
- Answer any questions about their Garmin data
- Create and schedule workouts on their Garmin device (when function calling is enabled)

Always be encouraging and supportive. If they're doing well, celebrate their achievements!{tool_instructions}"""

    # Build messages array for API
    api_messages = [{"role": "system", "content": full_system}]
    
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    request_body = {
        "model": MISTRAL_MODEL,
        "messages": api_messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    # Add tools if enabled
    if enable_tools:
        request_body["tools"] = WORKOUT_TOOLS
        request_body["tool_choice"] = "auto"
    
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {mistral_key}",
                "Content-Type": "application/json"
            },
            json=request_body,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            
            result = {
                "content": message.get("content", ""),
                "tool_calls": None
            }
            
            # Check for tool calls
            if "tool_calls" in message and message["tool_calls"]:
                result["tool_calls"] = message["tool_calls"]
            
            return result
        else:
            return {"content": f"❌ API Error ({response.status_code}): {response.text}", "tool_calls": None}
    
    except requests.exceptions.Timeout:
        return {"content": "❌ Request timed out. Please try again.", "tool_calls": None}
    except Exception as e:
        return {"content": f"❌ Error: {str(e)}", "tool_calls": None}


@st.cache_data
def load_time_series_data(metric_name):
    """Load time-series data from CSV file."""
    csv_path = TIME_SERIES_DIR / f"{metric_name}.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error loading {metric_name}: {str(e)}")
        return None


@st.cache_data
def load_daily_summary():
    """Load daily summary data."""
    csv_path = DAILY_DIR / "daily_summary.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading daily summary: {str(e)}")
        return None


@st.cache_data
def load_workouts():
    """Load workouts data."""
    csv_path = WORKOUTS_DIR / "workouts.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['date'] = df['start_time'].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading workouts: {str(e)}")
        return None


@st.cache_data
def load_scheduled_workouts():
    """Load scheduled workouts created from this app."""
    csv_path = WORKOUTS_DIR / "scheduled_workouts.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        if "scheduled_date" in df.columns:
            df["scheduled_date"] = pd.to_datetime(df["scheduled_date"]).dt.date
        return df
    except Exception as e:
        st.error(f"Error loading scheduled workouts: {str(e)}")
        return None


def save_scheduled_workout(workout_id, workout_name, workout_type, duration_minutes, scheduled_date):
    """Save a scheduled workout to local CSV for display."""
    csv_path = WORKOUTS_DIR / "scheduled_workouts.csv"
    record = {
        "workout_id": workout_id,
        "workout_name": workout_name,
        "workout_type": workout_type,
        "duration_minutes": duration_minutes,
        "scheduled_date": scheduled_date,
        "scheduled_at": datetime.now().isoformat(),
    }

    df = pd.DataFrame([record])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    # Clear cached scheduled workouts so UI refreshes
    load_scheduled_workouts.clear()


def get_garmin_client():
    """Get authenticated Garmin client using session credentials."""
    # Try session state first, then fall back to environment
    email = st.session_state.get("garmin_email", "") or os.getenv("GARMIN_EMAIL", "").strip().strip('"')
    password = st.session_state.get("garmin_password", "") or os.getenv("GARMIN_PASSWORD", "").strip().strip('"')
    
    if not email or not password:
        return None, "Garmin credentials not found. Please log in again."
    
    try:
        client = garminconnect.Garmin(email=email, password=password)
        client.login()
        return client, None
    except Exception as e:
        return None, str(e)




def plot_daily_metric(df, metric_col, title, color='#1f77b4', kind='line'):
    """Helper to plot a daily metric."""
    if metric_col not in df.columns:
        return None
    
    if kind == 'line':
        fig = px.line(df, x='date', y=metric_col, title=title, markers=True)
        fig.update_traces(line_color=color, line_width=3, marker_size=8)
    else:
        fig = px.bar(df, x='date', y=metric_col, title=title)
        fig.update_traces(marker_color=color)
        
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified"
    )
    return fig


def compute_activity_score(daily_df, days=7):
    """Compute activity score (0-100) for last N days."""
    if daily_df is None or daily_df.empty:
        return None

    recent = daily_df.sort_values("date").tail(days).copy()
    if recent.empty:
        return None

    def clamp(val, low=0.0, high=1.0):
        return max(low, min(high, val))

    # Define scoring components (normalized 0-1)
    components = []

    if "steps" in recent.columns:
        target = 10000
        steps_score = recent["steps"].fillna(0).mean() / target
        components.append(clamp(steps_score))

    if "sleep_hours" in recent.columns:
        target = 7.5
        sleep_score = recent["sleep_hours"].fillna(0).mean() / target
        components.append(clamp(sleep_score))

    if "active_min" in recent.columns:
        target = 30
        active_score = recent["active_min"].fillna(0).mean() / target
        components.append(clamp(active_score))
    elif "moderate_intensity_min" in recent.columns or "vigorous_intensity_min" in recent.columns:
        mod = recent["moderate_intensity_min"].fillna(0).mean() if "moderate_intensity_min" in recent.columns else 0
        vig = recent["vigorous_intensity_min"].fillna(0).mean() if "vigorous_intensity_min" in recent.columns else 0
        intensity_score = (mod + vig) / 30
        components.append(clamp(intensity_score))

    if "body_battery_max" in recent.columns:
        target = 80
        bb_score = recent["body_battery_max"].fillna(0).mean() / target
        components.append(clamp(bb_score))

    if "distance_km" in recent.columns:
        target = 7
        dist_score = recent["distance_km"].fillna(0).mean() / target
        components.append(clamp(dist_score))

    if "avg_stress" in recent.columns:
        # Lower stress is better; invert against 50
        stress_val = recent["avg_stress"].fillna(50).mean()
        stress_score = 1.0 - clamp(stress_val / 50)
        components.append(clamp(stress_score))

    if "resting_heart_rate" in recent.columns:
        # Lower RHR is better; invert against 70
        rhr_val = recent["resting_heart_rate"].fillna(70).mean()
        rhr_score = 1.0 - clamp(rhr_val / 70)
        components.append(clamp(rhr_score))

    if "hrv" in recent.columns:
        target = 60
        hrv_score = recent["hrv"].fillna(0).mean() / target
        components.append(clamp(hrv_score))

    if not components:
        return None

    score = int(round(sum(components) / len(components) * 100))
    return max(0, min(100, score))


def get_display_name(garmin_client):
    """Return a friendly display name from Garmin profile data."""
    if not garmin_client:
        return None

    # Try user profile endpoint first
    try:
        profile = garmin_client.get_user_profile()
        if isinstance(profile, dict):
            user = profile.get("userData") or profile
            for key in ("fullName", "displayName", "username", "name"):
                val = user.get(key)
                if val:
                    return val
    except Exception:
        pass

    # Fallbacks from client
    if getattr(garmin_client, "full_name", None):
        return garmin_client.full_name
    if getattr(garmin_client, "display_name", None):
        return garmin_client.display_name
    return None


def plot_metric_series(df, date_col, value_col, title):
    """Plot a time series line chart for a metric."""
    if df is None or df.empty or value_col not in df.columns:
        return None
    fig = px.line(df, x=date_col, y=value_col, title=title, markers=True)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _safe_daily_fetch(garmin_client, fetch_fn, date_str):
    try:
        return fetch_fn(date_str)
    except Exception:
        return None


def fetch_vo2max_and_fitness_df(garmin_client, start_date, end_date):
    """Fetch VO2 max and fitness age using get_max_metrics()."""
    rows = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        data = _safe_daily_fetch(garmin_client, garmin_client.get_max_metrics, date_str)
        if isinstance(data, dict):
            # Extract metrics from the response
            generic = data.get("generic", {}) if isinstance(data.get("generic"), dict) else {}
            row = {
                "date": current,
                "vo2max": generic.get("vo2Max"),
                "fitness_age": generic.get("fitnessAge"),
            }
            # Only add row if we have at least one metric
            if row["vo2max"] or row["fitness_age"]:
                rows.append(row)
        current += timedelta(days=1)
    return pd.DataFrame(rows) if rows else None


def fetch_training_readiness_df(garmin_client, start_date, end_date):
    rows = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        data = _safe_daily_fetch(garmin_client, garmin_client.get_training_readiness, date_str)
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            readiness = data.get("trainingReadinessScore") or data.get("score")
            sleep_score = data.get("sleepScore")
            recovery = data.get("recoveryTime")
            if readiness or sleep_score or recovery:
                row = {
                    "date": current,
                    "training_readiness": readiness,
                    "sleep_score": sleep_score,
                    "recovery_time": recovery,
                }
                rows.append(row)
        current += timedelta(days=1)
    return pd.DataFrame(rows) if rows else None


def fetch_endurance_score_df(garmin_client, start_date, end_date):
    try:
        data = garmin_client.get_endurance_score(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if isinstance(data, dict) and "stats" in data:
            data = data["stats"]
        if isinstance(data, list):
            rows = []
            for item in data:
                rows.append({
                    "date": pd.to_datetime(item.get("calendarDate")).date() if item.get("calendarDate") else None,
                    "endurance_score": item.get("value") or item.get("enduranceScore"),
                })
            df = pd.DataFrame(rows)
            return df.dropna(subset=["date"]) if not df.empty else None
    except Exception:
        return None
    return None


def fetch_hill_score_df(garmin_client, start_date, end_date):
    try:
        data = garmin_client.get_hill_score(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if isinstance(data, dict) and "stats" in data:
            data = data["stats"]
        if isinstance(data, list):
            rows = []
            for item in data:
                rows.append({
                    "date": pd.to_datetime(item.get("calendarDate")).date() if item.get("calendarDate") else None,
                    "hill_score": item.get("value") or item.get("hillScore"),
                })
            df = pd.DataFrame(rows)
            return df.dropna(subset=["date"]) if not df.empty else None
    except Exception:
        return None
    return None


def fetch_race_predictions_df(garmin_client, start_date, end_date):
    try:
        data = garmin_client.get_race_predictions(
            startdate=start_date.strftime("%Y-%m-%d"),
            enddate=end_date.strftime("%Y-%m-%d"),
            _type="daily",
        )
        if isinstance(data, list):
            rows = []
            for item in data:
                rows.append({
                    "date": pd.to_datetime(item.get("calendarDate")).date() if item.get("calendarDate") else None,
                    "distance_m": item.get("distanceMeters") or item.get("distance"),
                    "predicted_time_sec": item.get("timeInSeconds") or item.get("prediction"),
                })
            df = pd.DataFrame(rows)
            return df.dropna(subset=["date"]) if not df.empty else None
    except Exception:
        return None
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏃 Garmin Data Dashboard</h1>', unsafe_allow_html=True)
    
    # Auto-update check (on first load)
    if 'data_checked' not in st.session_state:
        if check_and_update_data():
            with st.spinner("🔄 Updating data... This may take a minute."):
                run_data_update()
                # Clear all cached data to reload fresh data
                load_daily_summary.clear()
                load_time_series_data.clear()
                load_workouts.clear()
                load_scheduled_workouts.clear()
                load_insights.clear()
        st.session_state.data_checked = True
    
    # Load data (or collect from Garmin if none exists — e.g. cloud deployment)
    daily_df = load_daily_summary()
    _log(f"[MAIN] First load_daily_summary(): {'has data' if daily_df is not None and not daily_df.empty else 'NO DATA'}")
    _log(f"[MAIN] collection_attempted: {st.session_state.get('collection_attempted', False)}")
    
    if (daily_df is None or daily_df.empty) and not st.session_state.get("collection_attempted", False):
        st.session_state["collection_attempted"] = True
        _log(f"[MAIN] Starting data collection...")
        with st.spinner("📥 Collecting data from Garmin... This may take a minute."):
            result = run_data_update()
            _log(f"[MAIN] run_data_update() returned: {result}")
        load_daily_summary.clear()
        load_time_series_data.clear()
        load_workouts.clear()
        load_scheduled_workouts.clear()
        load_insights.clear()
        _log(f"[MAIN] Caches cleared, about to rerun...")
        st.rerun()

    daily_df = load_daily_summary()
    if daily_df is None or daily_df.empty:
        st.error(
            "❌ No data found. Click **Sync New Data** in the sidebar to fetch your Garmin data."
        )
        # Show debug log
        st.subheader("🔍 Debug Log (what happened)")
        for line in st.session_state.get("debug_log", []):
            st.text(line)
        st.stop()
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            # Clear all credential-related session state
            for key in ["credentials_valid", "garmin_email", "garmin_password", "mistral_api_key", "login_error"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear environment variables
            for key in ["GARMIN_EMAIL", "GARMIN_PASSWORD", "MISTRAL_AI_API_KEY"]:
                if key in os.environ:
                    del os.environ[key]
            st.rerun()
        
        st.markdown("---")
        
        # Data update section
        st.subheader("🔄 Data Management")
        
        # Show last data date
        if daily_df is not None and not daily_df.empty:
            last_data_date = daily_df['date'].max().date()
            days_behind = (date.today() - last_data_date).days
            
            if days_behind == 0:
                st.success(f"✅ Data up to date (today)")
            elif days_behind == 1:
                st.info(f"📅 Last data: yesterday")
            else:
                st.warning(f"⚠️ Data is {days_behind} days behind")
        
        if st.button("🔄 Sync New Data", use_container_width=True):
            with st.spinner("Syncing with Garmin..."):
                run_data_update()
                # Clear caches
                load_daily_summary.clear()
                load_time_series_data.clear()
                load_workouts.clear()
                load_scheduled_workouts.clear()
                load_insights.clear()
                st.session_state.data_checked = False
                st.success("✅ Data synced!")
                st.rerun()
        
        st.markdown("---")
        
        # Date range for Daily Overview
        st.subheader("Daily Overview Range")
        min_date = daily_df['date'].min().date()
        max_date = daily_df['date'].max().date()
        
        # Default to last 7 days
        default_start = max_date - timedelta(days=6)
        if default_start < min_date:
            default_start = min_date
            
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

    # Filter daily data
    filtered_daily = daily_df[
        (daily_df['date'].dt.date >= start_date) & 
        (daily_df['date'].dt.date <= end_date)
    ].sort_values('date')

    # Load workouts data
    workouts_df = load_workouts()
    scheduled_workouts_df = load_scheduled_workouts()

    # Tabs (Home first as default)
    tab_home, tab_daily, tab_high_res, tab_fitness, tab_workouts, tab_goals, tab_insights, tab_lixxi = st.tabs(
        ["🏠 Home", "📅 Daily Overview", "📈 High Resolution Analysis", "💪 Fitness Metrics", "🏋️ Workouts", "🎯 Goals", "🔍 Data Insights", f"🤖 {CHATBOT_NAME}"]
    )

    # --- TAB 0: HOME ---
    with tab_home:
        garmin_client, garmin_error = get_garmin_client()
        raw_name = get_display_name(garmin_client) if garmin_client and not garmin_error else None
        # Avoid showing UUID-like IDs
        username = raw_name if raw_name and len(str(raw_name)) < 32 else "there"

        st.markdown(f"## 👋 Welcome, {username}")
        st.markdown("Here's your quick health snapshot and what's coming up.")

        # Activity Score (last 7 days)
        activity_score = compute_activity_score(daily_df, days=7)
        col_score, col_hint = st.columns([1, 2])
        with col_score:
            if activity_score is not None:
                st.metric("Activity Score (7-day)", f"{activity_score}/100")
                st.progress(activity_score / 100)
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=activity_score,
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1f77b4"},
                            "steps": [
                                {"range": [0, 40], "color": "#f8d7da"},
                                {"range": [40, 70], "color": "#fff3cd"},
                                {"range": [70, 100], "color": "#d4edda"},
                            ],
                        },
                    )
                )
                gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=0))
                st.plotly_chart(gauge, use_container_width=True)
            else:
                st.metric("Activity Score (7-day)", "N/A")
        with col_hint:
            st.markdown(
                "Based on steps, sleep, activity minutes, body battery, stress, resting HR, and HRV."
            )

        st.markdown("---")

        # Latest day summary
        latest_row = daily_df.sort_values("date").iloc[-1]
        latest_date = latest_row["date"].date() if pd.notna(latest_row["date"]) else None
        prev_row = (
            daily_df.sort_values("date").iloc[-2]
            if len(daily_df) > 1
            else None
        )

        st.markdown(f"### 📌 Latest Summary ({latest_date})")
        col1, col2, col3, col4 = st.columns(4)

        def safe_metric(val):
            return "N/A" if pd.isna(val) else val

        with col1:
            steps = latest_row.get("steps")
            prev_steps = prev_row.get("steps") if prev_row is not None else None
            delta_steps = (
                f"{int(steps - prev_steps):+}"
                if pd.notna(steps) and pd.notna(prev_steps)
                else None
            )
            st.metric("Steps", f"{int(steps):,}" if pd.notna(steps) else "N/A", delta=delta_steps)

        with col2:
            sleep = latest_row.get("sleep_hours")
            prev_sleep = prev_row.get("sleep_hours") if prev_row is not None else None
            delta_sleep = (
                f"{sleep - prev_sleep:+.1f}h"
                if pd.notna(sleep) and pd.notna(prev_sleep)
                else None
            )
            st.metric("Sleep", f"{sleep:.1f}h" if pd.notna(sleep) else "N/A", delta=delta_sleep)

        with col3:
            rhr = latest_row.get("resting_heart_rate")
            st.metric("Resting HR", f"{int(rhr)} bpm" if pd.notna(rhr) else "N/A")

        with col4:
            stress = latest_row.get("avg_stress")
            st.metric("Avg Stress", f"{int(stress)}" if pd.notna(stress) else "N/A")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            bb = latest_row.get("body_battery_max")
            st.metric("Body Battery", f"{int(bb)}" if pd.notna(bb) else "N/A")
        with col6:
            active_min = latest_row.get("active_min")
            st.metric("Active Minutes", f"{int(active_min)}" if pd.notna(active_min) else "N/A")
        with col7:
            hrv = latest_row.get("hrv")
            st.metric("HRV", f"{hrv:.0f}" if pd.notna(hrv) else "N/A")
        with col8:
            calories = latest_row.get("total_calories")
            st.metric("Calories", f"{int(calories)}" if pd.notna(calories) else "N/A")

        st.markdown("---")

        # Scheduled workouts (next 7 days)
        st.markdown("### 📅 Scheduled Workouts (Next 7 Days)")
        today = date.today()
        upcoming_end = today + timedelta(days=6)

        if scheduled_workouts_df is not None and not scheduled_workouts_df.empty:
            upcoming = scheduled_workouts_df[
                (scheduled_workouts_df["scheduled_date"] >= today)
                & (scheduled_workouts_df["scheduled_date"] <= upcoming_end)
            ].sort_values("scheduled_date")

            if upcoming.empty:
                st.info("No scheduled workouts in the next 7 days.")
            else:
                for _, w in upcoming.iterrows():
                    w_date = w["scheduled_date"]
                    w_name = w.get("workout_name", "Workout")
                    w_type = w.get("workout_type", "other")
                    w_dur = w.get("duration_minutes", 0)
                    st.success(
                        f"{w_date.strftime('%a %b %d')}: {w_name} ({w_type}, {int(w_dur)} min)"
                    )
        else:
            st.info("No scheduled workouts found. Schedule one in the Workouts tab.")

    # --- TAB 1: DAILY OVERVIEW ---
    with tab_daily:
        st.markdown("### 📊 Daily Metrics Configuration")
        
        # Get all numeric columns (exclude date/timestamp columns)
        all_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter to only include metrics that have at least some non-null data
        all_metrics = [c for c in all_cols 
                      if 'date' not in c.lower() 
                      and 'timestamp' not in c.lower()
                      and daily_df[c].notna().any()]
        all_metrics.sort()

        # Define default metrics to show on first load (filter to only those that exist)
        default_metrics_candidates = ['steps', 'resting_heart_rate', 'sleep_hours', 'body_battery_max', 'avg_stress']
        default_metrics = [m for m in default_metrics_candidates if m in all_metrics]

        # Initialize session state for selected metrics if not exists
        if 'selected_daily_metrics' not in st.session_state:
            st.session_state.selected_daily_metrics = default_metrics

        # Ensure current session state only contains valid metrics
        st.session_state.selected_daily_metrics = [m for m in st.session_state.selected_daily_metrics if m in all_metrics]

        # Daily Metric Selection Form
        with st.form("daily_metric_form"):
            selected_metrics = st.multiselect(
                "Select daily metrics to display:",
                options=all_metrics,
                default=st.session_state.selected_daily_metrics,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            submit_daily = st.form_submit_button("Update Dashboard")
            if submit_daily:
                st.session_state.selected_daily_metrics = selected_metrics
                st.rerun()
        
        st.markdown(f"### Trends ({start_date} to {end_date})")
        
        # Get the current selected metrics from session state
        current_metrics = st.session_state.selected_daily_metrics
        
        if not current_metrics:
            st.info("No metrics selected. Use the dropdown above to select metrics to display.")
        else:
            # Define colors for common metrics
            metric_colors = {
                'steps': '#1f77b4',
                'resting_heart_rate': '#d62728',
                'max_heart_rate': '#ff7f0e',
                'sleep_hours': '#9467bd',
                'body_battery_max': '#2ca02c',
                'body_battery_min': '#98df8a',
                'avg_stress': '#ff7f0e',
                'max_stress': '#e377c2',
                'calories_total': '#17becf',
                'distance_km': '#bcbd22',
                'active_minutes': '#7f7f7f',
                'floors_climbed': '#8c564b',
            }
            default_color = '#7f7f7f'
            
            # Display metrics in a 2-column layout
            for i in range(0, len(current_metrics), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(current_metrics):
                        metric = current_metrics[idx]
                        if metric in filtered_daily.columns:
                            color = metric_colors.get(metric, default_color)
                            title = metric.replace('_', ' ').title()
                            fig = plot_daily_metric(filtered_daily, metric, title, color=color, kind='line')
                            if fig:
                                col.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: HIGH RESOLUTION ---
    with tab_high_res:
        st.markdown("### 📈 High Resolution Configuration")
        
        # Check which time-series data files are available
        available_hr_metrics = []
        metric_file_map = {
            'Heart Rate': 'heart_rate',
            'Stress': 'stress',
            'Body Battery': 'body_battery',
            'Respiration': 'respiration',
            'SpO2': 'spo2'
        }
        
        for metric_name, file_prefix in metric_file_map.items():
            csv_path = TIME_SERIES_DIR / f"{file_prefix}.csv"
            if csv_path.exists():
                # Check if file has data
                try:
                    df_check = pd.read_csv(csv_path, nrows=1)
                    if not df_check.empty:
                        available_hr_metrics.append(metric_name)
                except:
                    pass
        
        # Add HRV During Sleep if we can fetch it from Garmin
        garmin_client, garmin_error = get_garmin_client()
        if garmin_client and not garmin_error:
            available_hr_metrics.append('HRV During Sleep')
        
        # Initialize session state for high-res metrics if not exists
        default_hr_candidates = ['Heart Rate', 'Body Battery', 'Stress', 'HRV During Sleep']
        default_hr = [m for m in default_hr_candidates if m in available_hr_metrics]
        
        if 'selected_hr_metrics' not in st.session_state:
            st.session_state.selected_hr_metrics = default_hr
        
        # Ensure current session state only contains valid metrics
        st.session_state.selected_hr_metrics = [m for m in st.session_state.selected_hr_metrics if m in available_hr_metrics]

        # Date Selector and Metric Selector for High Res
        with st.form("high_res_form"):
            col_date, col_metrics = st.columns([1, 3])
            with col_date:
                selected_date = st.date_input(
                    "Select Date for Detail View",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col_metrics:
                hr_metrics = st.multiselect(
                    "Select Metrics to Plot:",
                    options=available_hr_metrics,
                    default=st.session_state.selected_hr_metrics
                )
            
            submit_hr = st.form_submit_button("Update High-Res View")
            if submit_hr:
                st.session_state.selected_hr_metrics = hr_metrics
                st.rerun()

        st.markdown("### Intraday Analysis")
        
        # Use the metrics from session state
        current_hr_metrics = st.session_state.selected_hr_metrics
        
        # Load and plot selected metrics for the selected date
        if current_hr_metrics:
            # Map friendly names to file prefixes and column names
            metric_map = {
                'Heart Rate': ('heart_rate', 'heart_rate', 'red'),
                'Stress': ('stress', 'stress_level', 'orange'),
                'Body Battery': ('body_battery', 'body_battery', 'green'),
                'Respiration': ('respiration', 'respiration_rate', 'blue'),
                'SpO2': ('spo2', 'spo2', 'purple')
            }
            
            for metric_name in current_hr_metrics:
                # Special handling for HRV During Sleep
                if metric_name == 'HRV During Sleep':
                    garmin_client, garmin_error = get_garmin_client()
                    if garmin_client and not garmin_error:
                        try:
                            date_str = selected_date.strftime("%Y-%m-%d")
                            hrv_data = garmin_client.get_hrv_data(date_str)
                            
                            if hrv_data and isinstance(hrv_data, dict):
                                hrv_readings = hrv_data.get("hrvReadings", [])
                                if hrv_readings:
                                    df_hrv = pd.DataFrame(hrv_readings)
                                    if 'readingTimeLocal' in df_hrv.columns and 'hrvValue' in df_hrv.columns:
                                        df_hrv['time'] = pd.to_datetime(df_hrv['readingTimeLocal'])
                                        
                                        # Filter for selected date
                                        df_hrv_day = df_hrv[df_hrv['time'].dt.date == selected_date]
                                        
                                        if not df_hrv_day.empty:
                                            fig = px.line(df_hrv_day, x='time', y='hrvValue', 
                                                         title=f'HRV During Sleep on {selected_date}',
                                                         labels={'hrvValue': 'HRV (ms)', 'time': 'Time'})
                                            fig.update_traces(line_color='#2ca02c', line_width=2)
                                            fig.update_layout(height=350, xaxis_title="Time", yaxis_title="HRV (ms)")
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Add summary stats
                                            hrv_summary_data = hrv_data.get("hrvSummary", {})
                                            if isinstance(hrv_summary_data, dict):
                                                status = hrv_summary_data.get("status", "").replace("_", " ").title()
                                                last_night = hrv_summary_data.get("lastNightAvg")
                                                weekly = hrv_summary_data.get("weeklyAvg")
                                                if last_night and weekly:
                                                    st.info(f"**HRV Status:** {status} | Last Night Avg: {last_night} ms | Weekly Avg: {weekly} ms")
                                        else:
                                            st.warning(f"No HRV data for {selected_date}")
                                    else:
                                        st.warning("HRV data format not recognized")
                                else:
                                    st.warning(f"No HRV readings for {selected_date}")
                            else:
                                st.warning(f"Could not fetch HRV data for {selected_date}")
                        except Exception as e:
                            st.warning(f"Error fetching HRV data: {str(e)}")
                    else:
                        st.warning("Garmin not connected - cannot fetch HRV data")
                    continue
                
                # Regular time-series metrics
                file_prefix, col_name, color = metric_map[metric_name]
                
                # Load data (cached)
                df_hr = load_time_series_data(file_prefix)
                
                if df_hr is not None and not df_hr.empty:
                    # Filter for selected date
                    day_data = df_hr[df_hr['datetime'].dt.date == selected_date]
                    
                    if not day_data.empty:
                        fig = px.line(day_data, x='datetime', y=col_name, title=f"{metric_name} on {selected_date}")
                        fig.update_traces(line_color=color, line_width=1.5)
                        fig.update_layout(height=350, xaxis_title="Time", yaxis_title=metric_name)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No {metric_name} data available for {selected_date}")
                else:
                    st.warning(f"Could not load {metric_name} data.")

    # --- TAB 3: FITNESS METRICS ---
    with tab_fitness:
        st.markdown("### 💪 Fitness & Performance Metrics")
        
        garmin_client, garmin_error = get_garmin_client()
        if garmin_error:
            st.warning(f"⚠️ Garmin not connected: {garmin_error}")
            st.info("Connect to Garmin to view fitness metrics")
        else:
            today_str = date.today().strftime("%Y-%m-%d")
            
            # Fetch today's data with multiple methods
            with st.spinner("Fetching fitness data..."):
                # Try multiple endpoints for comprehensive data
                max_metrics = None
                training_status = None
                training_readiness = None
                fitness_age_data = None
                hrv_data = None
                lactate_threshold = None
                
                try:
                    max_metrics = garmin_client.get_max_metrics(today_str)
                except:
                    pass
                
                try:
                    training_status = garmin_client.get_training_status(today_str)
                except:
                    pass
                
                try:
                    training_readiness = garmin_client.get_training_readiness(today_str)
                    if isinstance(training_readiness, list) and training_readiness:
                        training_readiness = training_readiness[0]
                except:
                    pass
                
                try:
                    fitness_age_data = garmin_client.get_fitnessage_data(today_str)
                except:
                    pass
                
                try:
                    hrv_data = garmin_client.get_hrv_data(today_str)
                except:
                    pass
                
                try:
                    lactate_threshold = garmin_client.get_lactate_threshold(latest=True)
                except:
                    pass
            
            # Display VO2 Max
            st.subheader("📊 VO2 Max & Fitness")
            col1, col2, col3, col4 = st.columns(4)
            
            vo2_found = False
            with col1:
                # Extract from training_status.mostRecentVO2Max.generic
                vo2max_value = None
                if training_status and isinstance(training_status, dict):
                    vo2_recent = training_status.get("mostRecentVO2Max", {})
                    if isinstance(vo2_recent, dict):
                        generic = vo2_recent.get("generic", {})
                        if isinstance(generic, dict):
                            vo2max_value = generic.get("vo2MaxValue")
                            vo2max_precise = generic.get("vo2MaxPreciseValue")
                
                if vo2max_value:
                    st.metric("VO2 Max", f"{vo2max_value:.1f}", 
                             delta=f"Precise: {vo2max_precise:.1f}" if vo2max_precise else None)
                    vo2_found = True
                else:
                    st.metric("VO2 Max", "Not available")
            
            with col2:
                # Fitness age from fitness_age_data
                fitness_age = None
                chrono_age = None
                if fitness_age_data and isinstance(fitness_age_data, dict):
                    fitness_age = fitness_age_data.get("fitnessAge")
                    chrono_age = fitness_age_data.get("chronologicalAge")
                
                if fitness_age:
                    delta_text = f"{int(chrono_age - fitness_age):+} vs actual" if chrono_age else None
                    st.metric("Fitness Age", f"{int(fitness_age)} yrs", delta=delta_text)
                else:
                    st.metric("Fitness Age", "Not available")
            
            with col3:
                # HRV from hrv_data.hrvSummary
                if hrv_data and isinstance(hrv_data, dict):
                    hrv_summary = hrv_data.get("hrvSummary", {})
                    if isinstance(hrv_summary, dict):
                        last_night = hrv_summary.get("lastNightAvg")
                        weekly = hrv_summary.get("weeklyAvg")
                        status = hrv_summary.get("status", "").replace("_", " ").title()
                        if last_night:
                            st.metric("HRV (Last Night)", f"{last_night} ms", 
                                     delta=f"Weekly: {weekly}" if weekly else None)
                        elif weekly:
                            st.metric("HRV (Weekly)", f"{weekly} ms")
                        else:
                            st.metric("HRV", "Not available")
                    else:
                        st.metric("HRV", "Not available")
                else:
                    st.metric("HRV", "Not available")
            
            with col4:
                # Actual age
                if chrono_age:
                    st.metric("Actual Age", f"{int(chrono_age)} yrs")
                else:
                    st.metric("Actual Age", "Not available")
            
            # Training Status & Load
            st.markdown("---")
            st.subheader("🏃 Training Status & Load")
            
            # Extract training status details
            if training_status and isinstance(training_status, dict):
                recent_status = training_status.get("mostRecentTrainingStatus", {})
                if isinstance(recent_status, dict):
                    latest_data = recent_status.get("latestTrainingStatusData", {})
                    if isinstance(latest_data, dict):
                        # Get first device data
                        device_data = next(iter(latest_data.values())) if latest_data else None
                        
                        if device_data:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                status_val = device_data.get("trainingStatus", 0)
                                status_text = device_data.get("trainingStatusFeedbackPhrase", "").replace("_", " ").title()
                                status_map = {0: "No Status", 1: "Detraining", 2: "Recovery", 3: "Maintaining", 4: "Productive", 5: "Peaking", 6: "Overreaching", 7: "Strained"}
                                status_display = status_map.get(status_val, status_text or "Unknown")
                                st.metric("Training Status", status_display)
                            
                            with col2:
                                acute_load = device_data.get("acuteTrainingLoadDTO", {})
                                if isinstance(acute_load, dict):
                                    acute = acute_load.get("dailyTrainingLoadAcute")
                                    chronic = acute_load.get("dailyTrainingLoadChronic")
                                    if acute:
                                        st.metric("Acute Load", f"{int(acute)}", 
                                                 delta=f"Chronic: {int(chronic)}" if chronic else None)
                                    else:
                                        st.metric("Acute Load", "N/A")
                                else:
                                    st.metric("Acute Load", "N/A")
                            
                            with col3:
                                acute_load = device_data.get("acuteTrainingLoadDTO", {})
                                if isinstance(acute_load, dict):
                                    acwr = acute_load.get("dailyAcuteChronicWorkloadRatio")
                                    acwr_status = acute_load.get("acwrStatus", "").replace("_", " ").title()
                                    if acwr:
                                        st.metric("ACWR", f"{acwr:.2f}", delta=acwr_status)
                                    else:
                                        st.metric("ACWR", "N/A")
                                else:
                                    st.metric("ACWR", "N/A")
                
                # Training Load Balance
                load_balance = training_status.get("mostRecentTrainingLoadBalance", {})
                if isinstance(load_balance, dict):
                    metrics_map = load_balance.get("metricsTrainingLoadBalanceDTOMap", {})
                    if isinstance(metrics_map, dict) and metrics_map:
                        device_load = next(iter(metrics_map.values()))
                        if isinstance(device_load, dict):
                            st.markdown("**Monthly Training Load Balance:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                low = device_load.get("monthlyLoadAerobicLow")
                                if low:
                                    st.metric("Aerobic Low", f"{int(low)}")
                            with col2:
                                high = device_load.get("monthlyLoadAerobicHigh")
                                if high:
                                    st.metric("Aerobic High", f"{int(high)}")
                            with col3:
                                anaerobic = device_load.get("monthlyLoadAnaerobic")
                                if anaerobic:
                                    st.metric("Anaerobic", f"{int(anaerobic)}")
            
            # Lactate Threshold
            st.markdown("---")
            st.subheader("🔬 Lactate Threshold & Power")
            if lactate_threshold and isinstance(lactate_threshold, dict):
                col1, col2, col3, col4 = st.columns(4)
                
                # Speed and Heart Rate data
                speed_hr = lactate_threshold.get("speed_and_heart_rate", {})
                power_data = lactate_threshold.get("power", {})
                
                with col1:
                    if isinstance(speed_hr, dict):
                        hr = speed_hr.get("heartRate") or speed_hr.get("hearRate")
                        if hr:
                            st.metric("LT Heart Rate", f"{int(hr)} bpm")
                        else:
                            st.metric("LT Heart Rate", "N/A")
                    else:
                        st.metric("LT Heart Rate", "N/A")
                
                with col2:
                    if isinstance(speed_hr, dict):
                        speed = speed_hr.get("speed")
                        if speed:
                            # Convert m/s to min/km pace
                            pace = 1000 / (speed * 60) if speed > 0 else 0
                            pace_min = int(pace)
                            pace_sec = int((pace - pace_min) * 60)
                            st.metric("LT Pace", f"{pace_min}:{pace_sec:02d} /km")
                        else:
                            st.metric("LT Pace", "N/A")
                    else:
                        st.metric("LT Pace", "N/A")
                
                with col3:
                    if isinstance(power_data, dict):
                        ftp = power_data.get("functionalThresholdPower")
                        if ftp:
                            st.metric("FTP (Running)", f"{int(ftp)} W")
                        else:
                            st.metric("FTP", "N/A")
                    else:
                        st.metric("FTP", "N/A")
                
                with col4:
                    if isinstance(power_data, dict):
                        power_to_weight = power_data.get("powerToWeight")
                        if power_to_weight:
                            st.metric("Power/Weight", f"{power_to_weight:.2f} W/kg")
                        else:
                            st.metric("Power/Weight", "N/A")
                    else:
                        st.metric("Power/Weight", "N/A")
            else:
                st.info("Lactate threshold data not available")
            
            # Weekly/Historical Trends
            st.markdown("---")
            st.subheader("📈 Weekly Trends")
            
            # Get weekly steps stats
            try:
                weekly_steps = garmin_client.get_weekly_steps(today_str, 8)  # Last 8 weeks
                
                if weekly_steps:
                    df_steps = pd.DataFrame(weekly_steps)
                    if 'averageSteps' in df_steps.columns:
                        fig = px.line(df_steps, x='calendarDate', y='averageSteps', 
                                     title='Weekly Avg Steps (8 weeks)',
                                     markers=True)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            except:
                pass
            
            # Device Info
            if training_status and isinstance(training_status, dict):
                load_balance = training_status.get("mostRecentTrainingLoadBalance", {})
                devices = load_balance.get("recordedDevices", [])
                if devices:
                    st.markdown("---")
                    st.subheader("⌚ Training Device")
                    for device in devices:
                        device_name = device.get("deviceName", "Unknown")
                        img_url = device.get("imageURL")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if img_url:
                                st.image(img_url, width=100)
                        with col2:
                            st.markdown(f"**{device_name}**")
                            st.caption("Primary training device for these metrics")

    # --- TAB 3: WORKOUTS (Unified) ---
    with tab_workouts:
        st.markdown("### 🏋️ Workouts")
        
        # Connect to Garmin for scheduled workouts
        garmin_client, garmin_error = get_garmin_client()
        
        # Week navigation
        if 'week_offset' not in st.session_state:
            st.session_state.week_offset = 0
        
        col_prev, col_title, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("◀ Previous Week"):
                st.session_state.week_offset -= 1
                st.rerun()
        with col_next:
            if st.button("Next Week ▶"):
                st.session_state.week_offset += 1
                st.rerun()
        
        # Calculate week dates
        today = date.today()
        week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=st.session_state.week_offset)
        week_end = week_start + timedelta(days=6)
        
        with col_title:
            st.markdown(f"#### 📅 Week of {week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}")
        
        # Activity type emojis
        type_emojis = {
            'running': '🏃', 'cycling': '🚴', 'swimming': '🏊', 'walking': '🚶',
            'hiking': '🥾', 'strength_training': '🏋️', 'yoga': '🧘', 'cardio': '❤️',
            'hiit': '🔥', 'other': '🏃'
        }
        
        # Create week view with 7 columns
        day_cols = st.columns(7)
        
        for i, day_col in enumerate(day_cols):
            current_day = week_start + timedelta(days=i)
            day_name = current_day.strftime('%a')
            day_num = current_day.strftime('%d')
            is_today = current_day == today
            
            with day_col:
                # Day header
                if is_today:
                    st.markdown(f"**🔵 {day_name}**")
                    st.markdown(f"**{day_num}**")
                else:
                    st.markdown(f"**{day_name}**")
                    st.markdown(f"{day_num}")
                
                # Get completed workouts for this day (from local CSV)
                if workouts_df is not None and not workouts_df.empty:
                    day_completed = workouts_df[workouts_df['date'] == current_day]
                    for _, workout in day_completed.iterrows():
                        activity_type = str(workout.get('activity_type', 'other')).lower()
                        emoji = type_emojis.get(activity_type, '🏃')
                        name = workout.get('activity_name', 'Workout')
                        duration = workout.get('duration_minutes', 0)
                        if pd.notna(duration) and duration > 0:
                            st.success(f"{emoji} {int(duration)}m")
                        else:
                            st.success(f"{emoji} Done")
                
                # Scheduled workouts from local schedule log
                if scheduled_workouts_df is not None and not scheduled_workouts_df.empty:
                    day_scheduled = scheduled_workouts_df[
                        scheduled_workouts_df["scheduled_date"] == current_day
                    ]
                    for _, sw in day_scheduled.iterrows():
                        w_type = str(sw.get("workout_type", "other")).lower()
                        emoji = type_emojis.get(w_type, "🏃")
                        dur = sw.get("duration_minutes", 0)
                        if pd.notna(dur) and dur > 0:
                            st.info(f"{emoji} {int(dur)}m")
                        else:
                            st.info(f"{emoji} Planned")
        
        st.markdown("---")
        
        # Three sections side by side
        section_col1, section_col2 = st.columns(2)
        
        # Section 1: Create New Workout
        with section_col1:
            st.subheader("➕ Create & Schedule Workout")
            
            if garmin_error:
                st.warning(f"⚠️ Garmin not connected: {garmin_error}")
            else:
                workout_types = {
                    "Running": {"sportTypeId": 1, "sportTypeKey": "running"},
                    "Cycling": {"sportTypeId": 2, "sportTypeKey": "cycling"},
                    "Swimming": {"sportTypeId": 3, "sportTypeKey": "swimming"},
                    "Strength Training": {"sportTypeId": 4, "sportTypeKey": "strength_training"},
                    "Walking": {"sportTypeId": 9, "sportTypeKey": "walking"},
                    "Hiking": {"sportTypeId": 7, "sportTypeKey": "hiking"},
                    "Yoga": {"sportTypeId": 42, "sportTypeKey": "yoga"},
                    "HIIT": {"sportTypeId": 29, "sportTypeKey": "hiit"},
                }
                
                workout_type = st.selectbox("Workout Type", options=list(workout_types.keys()), index=0)
                workout_name = st.text_input("Workout Name", value=f"My {workout_type}")
                
                # Workout structure selection
                structure = st.radio(
                    "Workout Structure",
                    ["Simple", "Intervals", "Custom"],
                    horizontal=True,
                    help="Simple: single continuous effort | Intervals: warmup + repeats + cooldown | Custom: build your own"
                )
                
                workout_steps = []
                total_duration = 0
                
                if structure == "Simple":
                    duration_minutes = st.slider("Duration (min)", 10, 180, 60, 5)
                    target = st.selectbox("Target Zone", ["No Target", "Heart Rate Zone 2", "Heart Rate Zone 3", "Heart Rate Zone 4"])
                    
                    # Create single interval step
                    target_type = {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"}
                    if target != "No Target":
                        # Simple HR zone target (zone 2, 3, or 4)
                        zone_num = int(target.split("Zone ")[1])
                        target_type = {
                            "workoutTargetTypeId": 2,
                            "workoutTargetTypeKey": "heart.rate.zone",
                            "targetValueOne": zone_num,
                        }
                    
                    workout_steps = [{
                        "type": "ExecutableStepDTO",
                        "stepOrder": 1,
                        "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                        "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                        "endConditionValue": duration_minutes * 60,
                        "targetType": target_type,
                    }]
                    total_duration = duration_minutes * 60
                
                elif structure == "Intervals":
                    col1, col2 = st.columns(2)
                    with col1:
                        warmup_min = st.number_input("Warmup (min)", 0, 30, 10, 1)
                        interval_min = st.number_input("Interval (min)", 1, 30, 5, 1)
                        cooldown_min = st.number_input("Cooldown (min)", 0, 30, 10, 1)
                    with col2:
                        recovery_min = st.number_input("Recovery (min)", 0, 10, 2, 1)
                        repeats = st.number_input("Repeats", 1, 20, 5, 1)
                        interval_target = st.selectbox("Interval Target", ["No Target", "HR Zone 4", "HR Zone 5"])
                    
                    step_order = 1
                    
                    # Warmup
                    if warmup_min > 0:
                        workout_steps.append({
                            "type": "ExecutableStepDTO",
                            "stepOrder": step_order,
                            "stepType": {"stepTypeId": 1, "stepTypeKey": "warmup"},
                            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                            "endConditionValue": warmup_min * 60,
                            "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"},
                        })
                        step_order += 1
                        total_duration += warmup_min * 60
                    
                    # Interval/Recovery Repeat Group
                    repeat_steps = []
                    
                    # Interval step
                    interval_target_type = {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"}
                    if interval_target != "No Target":
                        zone_num = int(interval_target.split("Zone ")[1])
                        interval_target_type = {
                            "workoutTargetTypeId": 2,
                            "workoutTargetTypeKey": "heart.rate.zone",
                            "targetValueOne": zone_num,
                        }
                    
                    repeat_steps.append({
                        "type": "ExecutableStepDTO",
                        "stepOrder": 1,
                        "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                        "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                        "endConditionValue": interval_min * 60,
                        "targetType": interval_target_type,
                    })
                    
                    # Recovery step
                    if recovery_min > 0:
                        repeat_steps.append({
                            "type": "ExecutableStepDTO",
                            "stepOrder": 2,
                            "stepType": {"stepTypeId": 4, "stepTypeKey": "recovery"},
                            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                            "endConditionValue": recovery_min * 60,
                            "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"},
                        })
                    
                    # Add repeat group
                    workout_steps.append({
                        "type": "RepeatGroupDTO",
                        "stepOrder": step_order,
                        "stepType": {"stepTypeId": 6, "stepTypeKey": "repeat"},
                        "numberOfIterations": repeats,
                        "workoutSteps": repeat_steps,
                        "endCondition": {"conditionTypeId": 7, "conditionTypeKey": "iterations"},
                        "endConditionValue": float(repeats),
                    })
                    step_order += 1
                    total_duration += (interval_min + recovery_min) * 60 * repeats
                    
                    # Cooldown
                    if cooldown_min > 0:
                        workout_steps.append({
                            "type": "ExecutableStepDTO",
                            "stepOrder": step_order,
                            "stepType": {"stepTypeId": 2, "stepTypeKey": "cooldown"},
                            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                            "endConditionValue": cooldown_min * 60,
                            "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"},
                        })
                        total_duration += cooldown_min * 60
                
                else:  # Custom
                    st.info("Build your custom workout:")
                    
                    if 'custom_steps' not in st.session_state:
                        st.session_state.custom_steps = []
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        step_type = st.selectbox("Step Type", ["Warmup", "Interval", "Recovery", "Cooldown", "Rest"])
                    with col2:
                        step_duration = st.number_input("Duration (min)", 1, 60, 10, 1, key="custom_dur")
                    with col3:
                        step_target = st.selectbox("Target", ["No Target", "HR Zone 2", "HR Zone 3", "HR Zone 4", "HR Zone 5"])
                    
                    if st.button("➕ Add Step"):
                        st.session_state.custom_steps.append({
                            "type": step_type,
                            "duration": step_duration,
                            "target": step_target,
                        })
                        st.rerun()
                    
                    if st.session_state.custom_steps:
                        st.markdown("**Current Steps:**")
                        for i, step in enumerate(st.session_state.custom_steps):
                            col_step, col_del = st.columns([4, 1])
                            with col_step:
                                st.text(f"{i+1}. {step['type']}: {step['duration']}min ({step['target']})")
                            with col_del:
                                if st.button("🗑️", key=f"del_{i}"):
                                    st.session_state.custom_steps.pop(i)
                                    st.rerun()
                        
                        if st.button("🔄 Clear All"):
                            st.session_state.custom_steps = []
                            st.rerun()
                        
                        # Build workout from custom steps
                        step_type_map = {
                            "Warmup": {"stepTypeId": 1, "stepTypeKey": "warmup"},
                            "Interval": {"stepTypeId": 3, "stepTypeKey": "interval"},
                            "Recovery": {"stepTypeId": 4, "stepTypeKey": "recovery"},
                            "Cooldown": {"stepTypeId": 2, "stepTypeKey": "cooldown"},
                            "Rest": {"stepTypeId": 5, "stepTypeKey": "rest"},
                        }
                        
                        for i, step_info in enumerate(st.session_state.custom_steps):
                            step_type_dict = step_type_map[step_info["type"]]
                            target_type = {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"}
                            
                            if step_info["target"] != "No Target":
                                zone_num = int(step_info["target"].split("Zone ")[1])
                                target_type = {
                                    "workoutTargetTypeId": 2,
                                    "workoutTargetTypeKey": "heart.rate.zone",
                                    "targetValueOne": zone_num,
                                }
                            
                            workout_steps.append({
                                "type": "ExecutableStepDTO",
                                "stepOrder": i + 1,
                                "stepType": step_type_dict,
                                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                                "endConditionValue": step_info["duration"] * 60,
                                "targetType": target_type,
                            })
                            total_duration += step_info["duration"] * 60
                
                # Schedule date
                schedule_date = st.date_input("Schedule for Date", value=today, min_value=today)
                
                if st.button("🚀 Create & Schedule Workout", type="primary", use_container_width=True):
                    if not workout_steps:
                        st.error("❌ Please add at least one workout step")
                    else:
                        with st.spinner("Creating workout..."):
                            try:
                                sport_type = workout_types[workout_type]
                                workout_json = {
                                    "workoutName": workout_name,
                                    "sportType": sport_type,
                                    "workoutSegments": [{
                                        "segmentOrder": 1,
                                        "sportType": sport_type,
                                        "workoutSteps": workout_steps
                                    }],
                                    "estimatedDurationInSecs": int(total_duration),
                                }
                                
                                result = garmin_client.upload_workout(workout_json)
                                workout_id = result.get('workoutId')
                                
                                if workout_id:
                                    schedule_url = f"/workout-service/schedule/{workout_id}"
                                    schedule_payload = {"date": schedule_date.strftime("%Y-%m-%d")}
                                    garmin_client.garth.post("connectapi", schedule_url, json=schedule_payload, api=True)
                                    save_scheduled_workout(
                                        workout_id=workout_id,
                                        workout_name=workout_name,
                                        workout_type=workout_type,
                                        duration_minutes=int(total_duration / 60),
                                        scheduled_date=schedule_date,
                                    )
                                    st.success(f"✅ Workout '{workout_name}' scheduled for {schedule_date}!")
                                    st.balloons()
                                    # Clear custom steps if used
                                    if structure == "Custom":
                                        st.session_state.custom_steps = []
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to create workout")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        # Section 2: Saved Workout Templates
        with section_col2:
            st.subheader("📋 Saved Workouts")
            
            if garmin_error:
                st.info("Connect to Garmin to see saved workouts")
            else:
                try:
                    existing_workouts = garmin_client.get_workouts(start=0, limit=10)
                    
                    if existing_workouts:
                        for workout in existing_workouts:
                            w_id = workout.get('workoutId')
                            w_name = workout.get('workoutName', 'Unnamed')
                            w_sport = workout.get('sportType', {}).get('sportTypeKey', 'other')
                            est_secs = workout.get('estimatedDurationInSecs') or 0
                            w_duration = int(est_secs // 60) if est_secs else 0
                            emoji = type_emojis.get(w_sport.lower(), '🏃')
                            
                            with st.expander(f"{emoji} {w_name} ({w_duration}m)"):
                                quick_date = st.date_input("Schedule for", value=today, key=f"qd_{w_id}")
                                if st.button("Schedule", key=f"sch_{w_id}", use_container_width=True):
                                    try:
                                        url = f"/workout-service/schedule/{w_id}"
                                        garmin_client.garth.post(
                                            "connectapi",
                                            url,
                                            json={"date": quick_date.strftime("%Y-%m-%d")},
                                            api=True,
                                        )
                                        save_scheduled_workout(
                                            workout_id=w_id,
                                            workout_name=w_name,
                                            workout_type=w_sport,
                                            duration_minutes=w_duration,
                                            scheduled_date=quick_date,
                                        )
                                        st.success("✅ Scheduled!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))
                    else:
                        st.info("No saved workouts")
                except Exception as e:
                    st.warning(f"Could not load: {str(e)}")
        
        # Section 3: Completed Workouts Detail List
        st.markdown("---")
        st.subheader("✅ Completed Workouts This Week")
        
        if workouts_df is None or workouts_df.empty:
            st.info("No workout data. Run `python3 collect_garmin_data.py` to fetch.")
        else:
            week_workouts = workouts_df[
                (workouts_df['date'] >= week_start) & 
                (workouts_df['date'] <= week_end)
            ].sort_values('start_time', ascending=False)
            
            if week_workouts.empty:
                st.info("No completed workouts this week.")
            else:
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Workouts", len(week_workouts))
                with col2:
                    total_dur = week_workouts['duration_minutes'].sum()
                    st.metric("Total Time", f"{int(total_dur)} min")
                with col3:
                    total_cal = week_workouts['calories'].sum()
                    st.metric("Calories", f"{int(total_cal)}" if pd.notna(total_cal) else "N/A")
                with col4:
                    total_dist = week_workouts['distance_km'].sum()
                    st.metric("Distance", f"{total_dist:.1f} km" if pd.notna(total_dist) else "N/A")
                
                # Workout cards
                for _, workout in week_workouts.iterrows():
                    activity_type = str(workout.get('activity_type', 'other')).lower()
                    activity_name = workout.get('activity_name', 'Workout')
                    start_time = workout.get('start_time')
                    emoji = type_emojis.get(activity_type, '🏃')
                    
                    time_str = pd.to_datetime(start_time).strftime('%a %b %d, %H:%M') if pd.notna(start_time) else ""
                    
                    with st.expander(f"{emoji} **{activity_name}** - {time_str}"):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            dur = workout.get('duration_minutes')
                            st.metric("Duration", f"{int(dur)} min" if pd.notna(dur) else "N/A")
                        with c2:
                            dist = workout.get('distance_km')
                            st.metric("Distance", f"{dist:.2f} km" if pd.notna(dist) and dist > 0 else "N/A")
                        with c3:
                            cal = workout.get('calories')
                            st.metric("Calories", f"{int(cal)}" if pd.notna(cal) else "N/A")
                        with c4:
                            hr = workout.get('avg_heart_rate')
                            st.metric("Avg HR", f"{int(hr)} bpm" if pd.notna(hr) else "N/A")
    
    # --- TAB 5: GOALS AND CHALLENGES ---
    with tab_goals:
        st.markdown("### 🎯 Goals & Challenges")
        
        # Load current goals and challenges
        goals, challenges = load_goals()
        
        # Initialize session state for editing
        if 'editing_goals' not in st.session_state:
            st.session_state.editing_goals = False
        if 'editing_challenges' not in st.session_state:
            st.session_state.editing_challenges = False
        
        # --- WEEKLY GOALS PROGRESS ---
        st.subheader("📊 This Week's Progress")
        
        today = date.today()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        
        st.caption(f"Week of {start_of_week.strftime('%b %d')} - {end_of_week.strftime('%b %d, %Y')}")
        
        # Calculate progress for each goal
        goal_cols = st.columns(4)
        goal_keys = ['daily_steps', 'weekly_workouts', 'daily_sleep', 'weekly_active_minutes']
        
        for i, goal_key in enumerate(goal_keys):
            if goal_key in goals and goals[goal_key].get('enabled', False):
                progress = calculate_goal_progress(goal_key, goals, daily_df, workouts_df)
                
                if progress:
                    with goal_cols[i]:
                        # Progress card
                        st.markdown(f"**{progress['icon']} {progress['name']}**")
                        
                        # Progress bar
                        pct = progress.get('percentage', 0)
                        st.progress(min(1.0, pct / 100))
                        
                        # Status text
                        if progress.get('achieved', False):
                            st.success(f"✅ {progress['status']}")
                        elif pct >= 70:
                            st.info(f"🔵 {progress['status']}")
                        elif pct >= 40:
                            st.warning(f"🟡 {progress['status']}")
                        else:
                            st.error(f"🔴 {progress['status']}")
                        
                        # Target info
                        target = progress.get('target', 0)
                        unit = progress.get('unit', '')
                        st.caption(f"Target: {target} {unit}")
        
        st.markdown("---")
        
        # --- DETAILED GOAL BREAKDOWN ---
        st.subheader("📈 Detailed Goal Tracking")
        
        detail_tabs = st.tabs(["👣 Steps", "🏋️ Workouts", "😴 Sleep", "⚡ Activity"])
        
        # Steps detail
        with detail_tabs[0]:
            steps_goal = goals.get('daily_steps', {})
            if steps_goal.get('enabled', False):
                target = steps_goal.get('target', 10000)
                
                if daily_df is not None and 'steps' in daily_df.columns:
                    week_data = daily_df[
                        (daily_df['date'].dt.date >= start_of_week) & 
                        (daily_df['date'].dt.date <= today)
                    ].sort_values('date')
                    
                    if not week_data.empty:
                        # Daily steps bar chart
                        fig = go.Figure()
                        
                        colors = ['#2ca02c' if s >= target else '#ff7f0e' if s >= target * 0.7 else '#d62728' 
                                  for s in week_data['steps']]
                        
                        fig.add_trace(go.Bar(
                            x=week_data['date'].dt.strftime('%a %m/%d'),
                            y=week_data['steps'],
                            marker_color=colors,
                            text=week_data['steps'].astype(int),
                            textposition='outside'
                        ))
                        
                        # Add target line
                        fig.add_hline(y=target, line_dash="dash", line_color="green", 
                                      annotation_text=f"Target: {target:,}")
                        
                        fig.update_layout(
                            title="Daily Steps This Week",
                            yaxis_title="Steps",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            days_hit = len(week_data[week_data['steps'] >= target])
                            st.metric("Days at Goal", f"{days_hit}/{len(week_data)}")
                        with col2:
                            st.metric("Week Average", f"{int(week_data['steps'].mean()):,}")
                        with col3:
                            st.metric("Week Total", f"{int(week_data['steps'].sum()):,}")
                    else:
                        st.info("No step data for this week yet.")
            else:
                st.info("Steps goal is disabled. Enable it in settings below.")
        
        # Workouts detail
        with detail_tabs[1]:
            workout_goal = goals.get('weekly_workouts', {})
            if workout_goal.get('enabled', False):
                target = workout_goal.get('target', 3)
                
                if workouts_df is not None and not workouts_df.empty:
                    week_workouts = workouts_df[
                        (workouts_df['date'] >= start_of_week) & 
                        (workouts_df['date'] <= today)
                    ].sort_values('date' if 'date' in workouts_df.columns else 'start_time')
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        completed = len(week_workouts)
                        remaining = max(0, target - completed)
                        
                        st.metric("Completed", f"{completed}/{target}")
                        
                        if completed >= target:
                            st.success("🎉 Goal achieved!")
                        else:
                            days_left = (end_of_week - today).days
                            st.info(f"{remaining} more workout(s) needed ({days_left} days left)")
                    
                    with col2:
                        if not week_workouts.empty:
                            st.markdown("**Workouts this week:**")
                            for _, w in week_workouts.iterrows():
                                name = w.get('activity_name', 'Workout')
                                dur = w.get('duration_minutes', 0)
                                cal = w.get('calories', 0)
                                w_date = w.get('date', w.get('start_time', ''))
                                if isinstance(w_date, str):
                                    w_date = pd.to_datetime(w_date)
                                date_str = w_date.strftime('%a') if hasattr(w_date, 'strftime') else str(w_date)[:10]
                                st.write(f"• **{name}** ({date_str}) - {int(dur)}min, {int(cal) if pd.notna(cal) else 0} cal")
                        else:
                            st.info("No workouts logged this week.")
                else:
                    st.info("No workout data available.")
            else:
                st.info("Workouts goal is disabled. Enable it in settings below.")
        
        # Sleep detail
        with detail_tabs[2]:
            sleep_goal = goals.get('daily_sleep', {})
            if sleep_goal.get('enabled', False):
                target = sleep_goal.get('target', 7.0)
                
                if daily_df is not None and 'sleep_hours' in daily_df.columns:
                    week_data = daily_df[
                        (daily_df['date'].dt.date >= start_of_week) & 
                        (daily_df['date'].dt.date <= today)
                    ].sort_values('date')
                    
                    if not week_data.empty:
                        fig = go.Figure()
                        
                        colors = ['#2ca02c' if s >= target else '#ff7f0e' if s >= target * 0.85 else '#d62728' 
                                  for s in week_data['sleep_hours']]
                        
                        fig.add_trace(go.Bar(
                            x=week_data['date'].dt.strftime('%a %m/%d'),
                            y=week_data['sleep_hours'],
                            marker_color=colors,
                            text=[f"{s:.1f}h" for s in week_data['sleep_hours']],
                            textposition='outside'
                        ))
                        
                        fig.add_hline(y=target, line_dash="dash", line_color="green",
                                      annotation_text=f"Target: {target}h")
                        
                        fig.update_layout(
                            title="Sleep Duration This Week",
                            yaxis_title="Hours",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            nights_hit = len(week_data[week_data['sleep_hours'] >= target])
                            st.metric("Nights at Goal", f"{nights_hit}/{len(week_data)}")
                        with col2:
                            st.metric("Week Average", f"{week_data['sleep_hours'].mean():.1f}h")
                        with col3:
                            debt = max(0, target * len(week_data) - week_data['sleep_hours'].sum())
                            st.metric("Sleep Debt", f"{debt:.1f}h")
                    else:
                        st.info("No sleep data for this week yet.")
            else:
                st.info("Sleep goal is disabled. Enable it in settings below.")
        
        # Activity detail
        with detail_tabs[3]:
            activity_goal = goals.get('weekly_active_minutes', {})
            if activity_goal.get('enabled', False):
                target = activity_goal.get('target', 150)
                
                if daily_df is not None:
                    week_data = daily_df[
                        (daily_df['date'].dt.date >= start_of_week) & 
                        (daily_df['date'].dt.date <= today)
                    ].sort_values('date')
                    
                    # Calculate active minutes
                    active_cols = ['moderate_intensity_min', 'vigorous_intensity_min']
                    available_cols = [c for c in active_cols if c in week_data.columns]
                    
                    if available_cols:
                        week_data['total_active'] = week_data[available_cols].sum(axis=1)
                        total_active = week_data['total_active'].sum()
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=week_data['date'].dt.strftime('%a %m/%d'),
                            y=week_data['total_active'],
                            marker_color='#1f77b4',
                            text=week_data['total_active'].astype(int),
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title="Active Minutes This Week",
                            yaxis_title="Minutes",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Week Total", f"{int(total_active)} min")
                        with col2:
                            remaining = max(0, target - total_active)
                            st.metric("Remaining", f"{int(remaining)} min")
                        with col3:
                            pct = min(100, total_active / target * 100)
                            st.metric("Progress", f"{pct:.0f}%")
                        
                        if total_active >= target:
                            st.success("🎉 Weekly activity goal achieved!")
                    else:
                        st.info("No activity data available.")
            else:
                st.info("Activity goal is disabled. Enable it in settings below.")
        
        st.markdown("---")
        
        # --- CHALLENGES SECTION ---
        st.subheader("🏆 Challenges")
        
        challenge_col1, challenge_col2 = st.columns([2, 1])
        
        with challenge_col1:
            if challenges:
                for idx, challenge in enumerate(challenges):
                    progress = calculate_challenge_progress(challenge, daily_df, workouts_df)
                    
                    status = progress.get('status', 'active')
                    pct = progress.get('percentage', 0)
                    achieved = progress.get('achieved', False)
                    
                    # Challenge card
                    with st.expander(f"{progress['icon']} **{progress['name']}** - {progress['display']}", expanded=(status == 'active')):
                        st.progress(min(1.0, pct / 100))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            start = progress['start_date']
                            end = progress['end_date']
                            st.write(f"📅 {start} - {end}")
                        with col2:
                            if achieved:
                                st.success("✅ Achieved!")
                            elif status == 'completed':
                                st.error("❌ Not achieved")
                            elif status == 'upcoming':
                                st.info("⏳ Upcoming")
                            else:
                                st.info(f"🔵 {pct:.0f}% complete")
                        with col3:
                            if st.button("🗑️ Remove", key=f"rm_chal_{idx}"):
                                challenges.pop(idx)
                                save_goals(goals, challenges)
                                st.rerun()
            else:
                st.info("No active challenges. Create one below!")
        
        with challenge_col2:
            st.markdown("**➕ New Challenge**")
            
            with st.form("new_challenge_form"):
                chal_name = st.text_input("Challenge Name", placeholder="e.g., 100K Steps Week")
                
                chal_type = st.selectbox("Type", [
                    ("total_steps", "Total Steps"),
                    ("workout_count", "Workout Count"),
                    ("avg_sleep", "Average Sleep"),
                    ("step_streak", "Step Streak (10K+)")
                ], format_func=lambda x: x[1])
                
                chal_target = st.number_input("Target", min_value=1, value=70000 if chal_type[0] == 'total_steps' else 5)
                
                chal_duration = st.selectbox("Duration", ["This Week", "Next Week", "This Month"])
                
                if st.form_submit_button("Create Challenge", type="primary"):
                    # Calculate dates based on duration
                    if chal_duration == "This Week":
                        c_start = start_of_week
                        c_end = end_of_week
                    elif chal_duration == "Next Week":
                        c_start = start_of_week + timedelta(days=7)
                        c_end = end_of_week + timedelta(days=7)
                    else:  # This Month
                        c_start = today.replace(day=1)
                        if today.month == 12:
                            c_end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
                        else:
                            c_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
                    
                    icons = {
                        'total_steps': '👣',
                        'workout_count': '🏋️',
                        'avg_sleep': '😴',
                        'step_streak': '🔥'
                    }
                    
                    new_challenge = {
                        'name': chal_name or f"{chal_type[1]} Challenge",
                        'type': chal_type[0],
                        'target': chal_target,
                        'start_date': c_start.isoformat(),
                        'end_date': c_end.isoformat(),
                        'icon': icons.get(chal_type[0], '🎯')
                    }
                    
                    challenges.append(new_challenge)
                    save_goals(goals, challenges)
                    st.success(f"✅ Challenge '{new_challenge['name']}' created!")
                    st.rerun()
        
        st.markdown("---")
        
        # --- GOAL SETTINGS ---
        st.subheader("⚙️ Goal Settings")
        
        with st.expander("Edit Goal Targets", expanded=False):
            settings_cols = st.columns(2)
            
            with settings_cols[0]:
                st.markdown("**👣 Daily Steps Goal**")
                steps_enabled = st.checkbox("Enable", value=goals.get('daily_steps', {}).get('enabled', True), key="steps_en")
                steps_target = st.number_input("Target Steps", min_value=1000, max_value=50000, 
                                               value=goals.get('daily_steps', {}).get('target', 10000), step=500, key="steps_tgt")
                
                st.markdown("**😴 Daily Sleep Goal**")
                sleep_enabled = st.checkbox("Enable", value=goals.get('daily_sleep', {}).get('enabled', True), key="sleep_en")
                sleep_target = st.number_input("Target Hours", min_value=4.0, max_value=12.0,
                                               value=float(goals.get('daily_sleep', {}).get('target', 7.0)), step=0.5, key="sleep_tgt")
            
            with settings_cols[1]:
                st.markdown("**🏋️ Weekly Workouts Goal**")
                workout_enabled = st.checkbox("Enable", value=goals.get('weekly_workouts', {}).get('enabled', True), key="workout_en")
                workout_target = st.number_input("Target Workouts", min_value=1, max_value=14,
                                                 value=goals.get('weekly_workouts', {}).get('target', 3), key="workout_tgt")
                
                st.markdown("**⚡ Weekly Active Minutes Goal**")
                active_enabled = st.checkbox("Enable", value=goals.get('weekly_active_minutes', {}).get('enabled', True), key="active_en")
                active_target = st.number_input("Target Minutes", min_value=30, max_value=500,
                                                value=goals.get('weekly_active_minutes', {}).get('target', 150), step=10, key="active_tgt")
            
            if st.button("💾 Save Goal Settings", type="primary"):
                goals['daily_steps'] = {
                    'name': 'Daily Steps',
                    'target': steps_target,
                    'unit': 'steps',
                    'enabled': steps_enabled,
                    'icon': '👣'
                }
                goals['weekly_workouts'] = {
                    'name': 'Weekly Workouts',
                    'target': workout_target,
                    'unit': 'workouts',
                    'enabled': workout_enabled,
                    'icon': '🏋️'
                }
                goals['daily_sleep'] = {
                    'name': 'Daily Sleep',
                    'target': sleep_target,
                    'unit': 'hours',
                    'enabled': sleep_enabled,
                    'icon': '😴'
                }
                goals['weekly_active_minutes'] = {
                    'name': 'Weekly Active Minutes',
                    'target': active_target,
                    'unit': 'minutes',
                    'enabled': active_enabled,
                    'icon': '⚡'
                }
                
                save_goals(goals, challenges)
                st.success("✅ Goals saved!")
                st.rerun()
    
    # --- TAB 6: DATA INSIGHTS ---
    with tab_insights:
        st.markdown("### 🔍 Data Insights & Correlations")
        
        # Load insights
        insights = load_insights()
        
        if insights is None:
            st.info("⏳ Generating insights... Please refresh in a moment.")
            # Try to generate now
            if st.button("🔄 Generate Insights Now"):
                with st.spinner("Analyzing your data..."):
                    run_data_update()
                    load_insights.clear()
                    st.rerun()
            st.stop()
        
        # Data overview
        data_range = insights.get('data_range', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Days", data_range.get('days', 'N/A'))
        with col2:
            start = data_range.get('start', 'N/A')
            if start != 'N/A':
                start = pd.to_datetime(start).strftime('%Y-%m-%d')
            st.metric("From", start)
        with col3:
            end = data_range.get('end', 'N/A')
            if end != 'N/A':
                end = pd.to_datetime(end).strftime('%Y-%m-%d')
            st.metric("To", end)
        
        st.markdown("---")
        
        # Recovery Score
        recovery = insights.get('recovery', {})
        if recovery:
            st.subheader("💪 Recovery Status")
            
            recovery_score = recovery.get('recovery_score', 0)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=recovery_score,
                    title={'text': "Recovery Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ca02c"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ffcccc"},
                            {'range': [40, 70], 'color': "#ffffcc"},
                            {'range': [70, 100], 'color': "#ccffcc"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.markdown("**Factors:**")
                for factor in recovery.get('factors', []):
                    st.write(factor)
                
                recommendation = recovery.get('recommendation', '')
                if recommendation:
                    st.info(f"💡 **Recommendation:** {recommendation}")
        
        st.markdown("---")
        
        # Key Trends
        st.subheader("📈 Key Trends (Last 30 vs Previous 30 Days)")
        
        key_insights = insights.get('key_insights', {})
        trends = key_insights.get('trends', []) if isinstance(key_insights, dict) else []
        achievements = key_insights.get('achievements', []) if isinstance(key_insights, dict) else []
        concerns = key_insights.get('concerns', []) if isinstance(key_insights, dict) else []
        recommendations = key_insights.get('recommendations', []) if isinstance(key_insights, dict) else []
        
        if trends:
            for trend in trends:
                metric = trend.get('metric', 'Metric')
                text = trend.get('text', '')
                change_pct = trend.get('change_pct', 0)
                current_avg = trend.get('current_avg', 0)
                previous_avg = trend.get('previous_avg', 0)
                
                if change_pct > 0:
                    st.success(f"📈 **{metric}:** {text}")
                    st.caption(f"Current: {current_avg:.1f} | Previous: {previous_avg:.1f}")
                else:
                    st.warning(f"📉 **{metric}:** {text}")
                    st.caption(f"Current: {current_avg:.1f} | Previous: {previous_avg:.1f}")
        
        if achievements:
            st.markdown("**🏆 Achievements:**")
            for achievement in achievements:
                st.success(achievement)
        
        if concerns:
            st.markdown("**⚠️ Areas for Improvement:**")
            for concern in concerns:
                st.warning(concern)
        
        if recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in recommendations:
                st.info(rec)
        
        if not trends and not achievements and not concerns:
            st.info("No significant trends detected yet. More data needed.")
        
        st.markdown("---")
        
        # Correlation Analysis
        st.subheader("🔗 Key Correlations")
        
        correlations = insights.get('correlations', {})
        strong_corr = correlations.get('strong_correlations', [])
        
        if strong_corr:
            st.markdown("Discover relationships between your health metrics:")
            
            for i, corr in enumerate(strong_corr[:8]):  # Show top 8
                metric1 = corr.get('metric1', '').replace('_', ' ').title()
                metric2 = corr.get('metric2', '').replace('_', ' ').title()
                corr_val = corr.get('correlation', 0)
                strength = corr.get('strength', 'Moderate')
                interpretation = corr.get('interpretation', '')
                
                # Color based on correlation value
                if corr_val > 0.7:
                    emoji = "🔴"
                    color = "red"
                elif corr_val > 0.5:
                    emoji = "🟠"
                    color = "orange"
                elif corr_val < -0.7:
                    emoji = "🔵"
                    color = "blue"
                else:
                    emoji = "🟡"
                    color = "yellow"
                
                with st.expander(f"{emoji} **{metric1}** ↔ **{metric2}** ({strength}: {corr_val:.2f})"):
                    st.write(f"**Correlation:** {corr_val:.3f}")
                    st.write(f"**Interpretation:** {interpretation}")
                    
                    # Show scatter plot
                    if metric1.lower().replace(' ', '_') in filtered_daily.columns and metric2.lower().replace(' ', '_') in filtered_daily.columns:
                        col1_name = corr.get('metric1')
                        col2_name = corr.get('metric2')
                        
                        fig_scatter = px.scatter(
                            filtered_daily,
                            x=col1_name,
                            y=col2_name,
                            title=f"{metric1} vs {metric2}",
                            trendline="ols"
                        )
                        fig_scatter.update_layout(height=300)
                        st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No strong correlations found yet. More data needed for correlation analysis.")
        
        # Correlation Heatmap
        if correlations.get('matrix'):
            st.markdown("---")
            st.subheader("🌡️ Correlation Heatmap")
            
            corr_matrix_dict = correlations['matrix']
            corr_df = pd.DataFrame(corr_matrix_dict)
            
            # Remove columns/rows where all values are 0 or NaN
            cols_to_keep = []
            for col in corr_df.columns:
                col_values = corr_df[col].drop(col)  # Exclude diagonal (self-correlation)
                # Keep column if it has at least one non-zero, non-NaN value
                if not col_values.isna().all() and not (col_values == 0).all():
                    cols_to_keep.append(col)
            
            if len(cols_to_keep) >= 2:
                corr_df = corr_df.loc[cols_to_keep, cols_to_keep]
                
                # Create heatmap with plotly
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=[col.replace('_', ' ').title() for col in corr_df.columns],
                    y=[col.replace('_', ' ').title() for col in corr_df.columns],
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_df.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 8},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_heatmap.update_layout(
                    title="Correlation Matrix - All Metrics",
                    height=600,
                    xaxis={'side': 'bottom'},
                    yaxis={'autorange': 'reversed'}
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough valid metrics for correlation heatmap (all correlations are zero).")
        
        st.markdown("---")
        
        # Weekly Patterns
        st.subheader("📅 Weekly Patterns")
        
        weekly_patterns = insights.get('weekly_patterns', {})
        
        if weekly_patterns:
            pattern_metric = st.selectbox(
                "Select metric to analyze weekly pattern:",
                options=[k.replace('_', ' ').title() for k in weekly_patterns.keys()],
                key="weekly_pattern_select"
            )
            
            pattern_key = pattern_metric.lower().replace(' ', '_')
            
            if pattern_key in weekly_patterns:
                pattern_data = weekly_patterns[pattern_key]
                daily_avgs = pattern_data.get('daily_averages', [])
                
                if daily_avgs:
                    df_pattern = pd.DataFrame(daily_avgs)
                    
                    # Bar chart
                    fig_weekly = px.bar(
                        df_pattern,
                        x='day_of_week',
                        y='mean',
                        error_y='std',
                        title=f"Average {pattern_metric} by Day of Week",
                        labels={'mean': pattern_metric, 'day_of_week': 'Day'}
                    )
                    fig_weekly.update_layout(height=350)
                    st.plotly_chart(fig_weekly, use_container_width=True)
                    
                    best_day = pattern_data.get('best_day')
                    worst_day = pattern_data.get('worst_day')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if best_day:
                            st.success(f"🏆 Best day: **{best_day}**")
                    with col2:
                        if worst_day:
                            st.info(f"📊 Lowest day: **{worst_day}**")
        else:
            st.info("Weekly patterns analysis not available. More data needed.")
        
        st.markdown("---")
        
        # Outliers
        st.subheader("📌 Notable Days")
        
        outliers = insights.get('outliers', [])
        
        if outliers:
            st.markdown("Days where your metrics were significantly different from your average:")
            
            # Group by date and show most significant outlier per day
            outlier_df = pd.DataFrame(outliers)
            outlier_df['date'] = pd.to_datetime(outlier_df['date'])
            outlier_df = outlier_df.sort_values('z_score', key=abs, ascending=False)
            
            # Show top 10 outliers
            for _, outlier in outlier_df.head(10).iterrows():
                date_str = outlier['date'].strftime('%Y-%m-%d')
                metric = outlier['metric'].replace('_', ' ').title()
                value = outlier['value']
                mean = outlier['mean']
                direction = outlier['direction']
                z_score = outlier['z_score']
                
                if direction == 'high':
                    st.success(f"🔺 **{date_str}:** {metric} was unusually high ({value:.1f} vs avg {mean:.1f})")
                else:
                    st.warning(f"🔻 **{date_str}:** {metric} was unusually low ({value:.1f} vs avg {mean:.1f})")
        else:
            st.info("No significant outliers detected.")
        
        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh Insights", type="primary"):
            with st.spinner("Regenerating insights..."):
                run_data_update()
                load_insights.clear()
                st.rerun()
    
    # --- TAB 7: LIXXI AI CHATBOT ---
    with tab_lixxi:
        st.markdown(f"### 🤖 {CHATBOT_NAME} - Your AI Fitness Assistant")
        
        # Initialize session state for chat
        if 'current_chat_id' not in st.session_state:
            st.session_state.current_chat_id = None
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'chat_title' not in st.session_state:
            st.session_state.chat_title = "New Chat"
        if 'enable_function_calling' not in st.session_state:
            st.session_state.enable_function_calling = False
        if 'pending_workout' not in st.session_state:
            st.session_state.pending_workout = None
        if 'pending_workout_params' not in st.session_state:
            st.session_state.pending_workout_params = None
        
        # Layout: sidebar for chat history, main for chat
        chat_sidebar, chat_main = st.columns([1, 3])
        
        with chat_sidebar:
            st.markdown("#### 💬 Conversations")
            
            # New Chat button
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                st.session_state.current_chat_id = str(uuid.uuid4())
                st.session_state.chat_messages = []
                st.session_state.chat_title = "New Chat"
                st.session_state.pending_workout = None
                st.session_state.pending_workout_params = None
                st.rerun()
            
            st.markdown("---")
            
            # Function Calling Toggle
            st.markdown("#### ⚙️ Settings")
            enable_fc = st.toggle(
                "🔧 Allow Function Calling",
                value=st.session_state.enable_function_calling,
                help="Enable Lixxi to create and schedule workouts on your Garmin"
            )
            if enable_fc != st.session_state.enable_function_calling:
                st.session_state.enable_function_calling = enable_fc
                st.rerun()
            
            if st.session_state.enable_function_calling:
                st.success("✅ Workout creation enabled")
                st.caption("Lixxi can now create workouts for you!")
            else:
                st.info("💡 Enable to let Lixxi create workouts")
            
            st.markdown("---")
            
            # Load and display chat list
            st.markdown("#### 📚 History")
            chat_list = load_chat_list()
            
            if chat_list:
                for chat_info in chat_list[:12]:  # Show last 12 chats
                    chat_id = chat_info['id']
                    title = chat_info['title'][:22] + "..." if len(chat_info['title']) > 22 else chat_info['title']
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        is_current = st.session_state.current_chat_id == chat_id
                        btn_type = "primary" if is_current else "secondary"
                        
                        if st.button(f"💬 {title}", key=f"load_{chat_id}", use_container_width=True, type=btn_type):
                            chat_data = load_chat(chat_id)
                            if chat_data:
                                st.session_state.current_chat_id = chat_id
                                st.session_state.chat_messages = chat_data.get('messages', [])
                                st.session_state.chat_title = chat_data.get('title', 'Chat')
                                st.session_state.pending_workout = None
                                st.session_state.pending_workout_params = None
                                st.rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{chat_id}", help="Delete chat"):
                            delete_chat(chat_id)
                            if st.session_state.current_chat_id == chat_id:
                                st.session_state.current_chat_id = None
                                st.session_state.chat_messages = []
                                st.session_state.chat_title = "New Chat"
                            st.rerun()
            else:
                st.info("No saved chats yet.")
        
        with chat_main:
            # Function calling status banner
            if st.session_state.enable_function_calling:
                st.info("🔧 **Function Calling Enabled** - Ask me to create a workout and I'll help you build it!")
            
            # Display chat messages
            chat_container = st.container()
            
            with chat_container:
                if not st.session_state.chat_messages:
                    # Welcome message
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; color: white; margin-bottom: 1rem;">
                        <h2>👋 Hi! I'm {CHATBOT_NAME}</h2>
                        <p>I'm your personal AI fitness assistant. I have access to your Garmin data and can help you with:</p>
                        <ul style="text-align: left; display: inline-block;">
                            <li>📊 Analyzing your activity and fitness trends</li>
                            <li>😴 Understanding your sleep patterns</li>
                            <li>💪 Suggesting workout improvements</li>
                            <li>❤️ Explaining health metrics (HRV, Body Battery, etc.)</li>
                            <li>🏋️ <strong>Creating & scheduling workouts</strong> (enable Function Calling)</li>
                        </ul>
                        <p><strong>Try asking me something!</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Suggested prompts
                    st.markdown("**💡 Suggested questions:**")
                    suggested_cols = st.columns(2)
                    
                    suggestions = [
                        "How was my activity this week?",
                        "Am I getting enough sleep?",
                        "What's my stress level like?",
                        "Create a 30-minute interval run for me"
                    ]
                    
                    for i, suggestion in enumerate(suggestions):
                        with suggested_cols[i % 2]:
                            if st.button(f"💬 {suggestion}", key=f"suggest_{i}", use_container_width=True):
                                if not st.session_state.current_chat_id:
                                    st.session_state.current_chat_id = str(uuid.uuid4())
                                
                                st.session_state.chat_messages.append({
                                    "role": "user",
                                    "content": suggestion
                                })
                                
                                garmin_context = get_garmin_context(daily_df, workouts_df, days=14)
                                with st.spinner(f"{CHATBOT_NAME} is thinking..."):
                                    response = call_mistral_api(
                                        st.session_state.chat_messages,
                                        "",
                                        garmin_context,
                                        enable_tools=st.session_state.enable_function_calling
                                    )
                                
                                # Handle response (may include tool calls)
                                if response.get("tool_calls") and st.session_state.enable_function_calling:
                                    # Process tool call
                                    for tool_call in response["tool_calls"]:
                                        if tool_call["function"]["name"] == "create_workout":
                                            args = json.loads(tool_call["function"]["arguments"])
                                            workout_json, duration, summary = build_workout_json(args)
                                            st.session_state.pending_workout = workout_json
                                            st.session_state.pending_workout_params = args
                                            
                                            # Add AI message about the workout
                                            st.session_state.chat_messages.append({
                                                "role": "assistant",
                                                "content": f"I've created a workout for you! Here's what I'm proposing:\n\n{summary}\n\n**Please review and confirm below to schedule this workout on your Garmin.**"
                                            })
                                else:
                                    st.session_state.chat_messages.append({
                                        "role": "assistant",
                                        "content": response.get("content", "I couldn't process that request.")
                                    })
                                
                                st.session_state.chat_title = generate_chat_title(suggestion)
                                save_chat(
                                    st.session_state.current_chat_id,
                                    st.session_state.chat_title,
                                    st.session_state.chat_messages
                                )
                                st.rerun()
                
                else:
                    # Display existing messages
                    for msg in st.session_state.chat_messages:
                        role = msg["role"]
                        content = msg["content"]
                        
                        if role == "user":
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(content)
                        elif role == "assistant":
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(content)
            
            # Pending Workout Confirmation
            if st.session_state.pending_workout:
                st.markdown("---")
                st.markdown("### 🏋️ Workout Ready for Confirmation")
                
                workout_params = st.session_state.pending_workout_params or {}
                workout_json = st.session_state.pending_workout
                
                # Editable workout details
                with st.expander("✏️ Edit Workout Details", expanded=True):
                    edit_col1, edit_col2 = st.columns(2)
                    
                    with edit_col1:
                        edited_name = st.text_input(
                            "Workout Name",
                            value=workout_params.get("workout_name", "AI Workout"),
                            key="edit_workout_name"
                        )
                        
                        sport_options = ["running", "cycling", "swimming", "strength_training", "walking", "hiking", "yoga", "other"]
                        current_sport = workout_params.get("sport_type", "running")
                        sport_idx = sport_options.index(current_sport) if current_sport in sport_options else 0
                        edited_sport = st.selectbox(
                            "Sport Type",
                            options=sport_options,
                            index=sport_idx,
                            format_func=lambda x: x.replace("_", " ").title(),
                            key="edit_sport_type"
                        )
                    
                    with edit_col2:
                        # Schedule date
                        schedule_str = workout_params.get("schedule_date", date.today().isoformat())
                        try:
                            default_date = datetime.strptime(schedule_str, "%Y-%m-%d").date()
                        except:
                            default_date = date.today()
                        
                        edited_date = st.date_input(
                            "Schedule Date",
                            value=default_date,
                            min_value=date.today(),
                            key="edit_schedule_date"
                        )
                        
                        # Duration display
                        duration_mins = workout_json.get("estimatedDurationInSecs", 0) // 60
                        st.metric("Total Duration", f"{duration_mins} minutes")
                    
                    # Show workout steps summary
                    st.markdown("**Workout Steps:**")
                    steps = workout_json.get("workoutSegments", [{}])[0].get("workoutSteps", [])
                    for i, step in enumerate(steps):
                        step_type = step.get("stepType", {}).get("stepTypeKey", "unknown")
                        if step.get("type") == "RepeatGroupDTO":
                            repeats = step.get("numberOfIterations", 1)
                            st.write(f"  {i+1}. 🔄 Repeat {repeats}x")
                            for j, sub_step in enumerate(step.get("workoutSteps", [])):
                                sub_type = sub_step.get("stepType", {}).get("stepTypeKey", "unknown")
                                sub_dur = sub_step.get("endConditionValue", 0) // 60
                                target = sub_step.get("targetType", {})
                                zone = target.get("targetValueOne", "")
                                zone_str = f" @ Zone {zone}" if zone else ""
                                st.write(f"      - {sub_type.title()}: {sub_dur} min{zone_str}")
                        else:
                            step_dur = step.get("endConditionValue", 0) // 60
                            target = step.get("targetType", {})
                            zone = target.get("targetValueOne", "")
                            zone_str = f" @ Zone {zone}" if zone else ""
                            st.write(f"  {i+1}. {step_type.title()}: {step_dur} min{zone_str}")
                
                # Confirmation buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if st.button("✅ Confirm & Schedule", type="primary", use_container_width=True):
                        # Update workout with edited values
                        workout_json["workoutName"] = edited_name
                        
                        sport_type_map = {
                            "running": {"sportTypeId": 1, "sportTypeKey": "running"},
                            "cycling": {"sportTypeId": 2, "sportTypeKey": "cycling"},
                            "swimming": {"sportTypeId": 5, "sportTypeKey": "swimming"},
                            "strength_training": {"sportTypeId": 4, "sportTypeKey": "strength_training"},
                            "walking": {"sportTypeId": 9, "sportTypeKey": "walking"},
                            "hiking": {"sportTypeId": 17, "sportTypeKey": "hiking"},
                            "yoga": {"sportTypeId": 43, "sportTypeKey": "yoga"},
                            "other": {"sportTypeId": 0, "sportTypeKey": "other"},
                        }
                        workout_json["sportType"] = sport_type_map.get(edited_sport, sport_type_map["other"])
                        workout_json["workoutSegments"][0]["sportType"] = workout_json["sportType"]
                        
                        # Upload to Garmin
                        garmin_client, garmin_error = get_garmin_client()
                        
                        if garmin_error:
                            st.error(f"❌ Cannot connect to Garmin: {garmin_error}")
                        else:
                            with st.spinner("Creating workout on Garmin..."):
                                try:
                                    result = garmin_client.upload_workout(workout_json)
                                    workout_id = result.get('workoutId')
                                    
                                    if workout_id:
                                        # Schedule the workout
                                        schedule_url = f"/workout-service/schedule/{workout_id}"
                                        schedule_payload = {"date": edited_date.strftime("%Y-%m-%d")}
                                        garmin_client.garth.post("connectapi", schedule_url, json=schedule_payload, api=True)
                                        
                                        # Save locally
                                        save_scheduled_workout(
                                            workout_id=workout_id,
                                            workout_name=edited_name,
                                            workout_type=edited_sport,
                                            duration_minutes=duration_mins,
                                            scheduled_date=edited_date,
                                        )
                                        
                                        # Add success message to chat
                                        st.session_state.chat_messages.append({
                                            "role": "assistant",
                                            "content": f"🎉 **Workout Created Successfully!**\n\n'{edited_name}' has been scheduled for **{edited_date}** and will sync to your Garmin watch!"
                                        })
                                        
                                        st.session_state.pending_workout = None
                                        st.session_state.pending_workout_params = None
                                        
                                        save_chat(
                                            st.session_state.current_chat_id,
                                            st.session_state.chat_title,
                                            st.session_state.chat_messages
                                        )
                                        
                                        st.success(f"✅ Workout '{edited_name}' scheduled for {edited_date}!")
                                        st.balloons()
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to create workout on Garmin")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                
                with btn_col2:
                    if st.button("🔄 Modify Request", use_container_width=True):
                        st.session_state.pending_workout = None
                        st.session_state.pending_workout_params = None
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": "No problem! What changes would you like to make to the workout? You can tell me to:\n- Change the duration\n- Add more intervals\n- Adjust the intensity/HR zones\n- Change the sport type\n- Or describe a completely different workout"
                        })
                        save_chat(
                            st.session_state.current_chat_id,
                            st.session_state.chat_title,
                            st.session_state.chat_messages
                        )
                        st.rerun()
                
                with btn_col3:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.pending_workout = None
                        st.session_state.pending_workout_params = None
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": "No worries, I've cancelled the workout. Let me know if you'd like to try something different!"
                        })
                        save_chat(
                            st.session_state.current_chat_id,
                            st.session_state.chat_title,
                            st.session_state.chat_messages
                        )
                        st.rerun()
            
            # Chat input
            st.markdown("---")
            
            user_input = st.chat_input(f"Ask {CHATBOT_NAME} anything about your fitness data...")
            
            if user_input:
                # Create new chat ID if needed
                if not st.session_state.current_chat_id:
                    st.session_state.current_chat_id = str(uuid.uuid4())
                
                # Add user message
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Generate title from first message if new chat
                if len(st.session_state.chat_messages) == 1:
                    st.session_state.chat_title = generate_chat_title(user_input)
                
                # Get Garmin context
                garmin_context = get_garmin_context(daily_df, workouts_df, days=14)
                
                # Call Mistral API with function calling if enabled
                with st.spinner(f"{CHATBOT_NAME} is thinking..."):
                    response = call_mistral_api(
                        st.session_state.chat_messages,
                        "",
                        garmin_context,
                        enable_tools=st.session_state.enable_function_calling
                    )
                
                # Handle response
                if response.get("tool_calls") and st.session_state.enable_function_calling:
                    # Process tool call for workout creation
                    for tool_call in response["tool_calls"]:
                        if tool_call["function"]["name"] == "create_workout":
                            args = json.loads(tool_call["function"]["arguments"])
                            
                            # Set default date if not provided
                            if "schedule_date" not in args:
                                args["schedule_date"] = date.today().isoformat()
                            
                            workout_json, duration, summary = build_workout_json(args)
                            st.session_state.pending_workout = workout_json
                            st.session_state.pending_workout_params = args
                            
                            # Add AI message about the workout
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": f"I've designed a workout for you! 🏋️\n\n{summary}\n\n**Please review the details above and click 'Confirm & Schedule' to add it to your Garmin, or 'Modify Request' to make changes.**"
                            })
                else:
                    # Regular text response
                    content = response.get("content", "I couldn't process that request.")
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": content
                    })
                
                # Save chat
                save_chat(
                    st.session_state.current_chat_id,
                    st.session_state.chat_title,
                    st.session_state.chat_messages
                )
                
                st.rerun()
            
            # Show current chat info
            if st.session_state.current_chat_id and st.session_state.chat_messages:
                st.caption(f"💬 {st.session_state.chat_title} • {len(st.session_state.chat_messages)} messages")

if __name__ == "__main__":
    main()
