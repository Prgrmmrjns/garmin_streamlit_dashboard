#!/usr/bin/env python3
"""
Analyze and visualize collected Garmin data.
Generates comprehensive visualizations, correlations, and statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TIME_SERIES_DIR = DATA_DIR / "time_series"
DAILY_DIR = DATA_DIR / "daily"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"

# Create directories
REPORTS_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


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
        print(f"⚠️  Error loading {metric_name}: {str(e)}")
        return None


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
        print(f"⚠️  Error loading daily summary: {str(e)}")
        return None


def plot_time_series(metric_name, days=None, save=True):
    """Create time-series plots for a specific metric."""
    df = load_time_series_data(metric_name)
    if df is None or df.empty:
        print(f"⚠️  No data available for {metric_name}")
        return None
    
    # Filter by date range if specified
    if days:
        cutoff_date = df['datetime'].max() - timedelta(days=days)
        df = df[df['datetime'] >= cutoff_date]
    
    # Get the value column name
    value_col = None
    for col in ['heart_rate', 'stress_level', 'body_battery', 'respiration_rate', 'spo2']:
        if col in df.columns:
            value_col = col
            break
    
    if not value_col:
        print(f"⚠️  Could not find value column for {metric_name}")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full time series
    axes[0].plot(df['datetime'], df[value_col], linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'{metric_name.replace("_", " ").title()} Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(value_col.replace("_", " ").title())
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Daily averages
    df['date'] = df['datetime'].dt.date
    daily_avg = df.groupby('date')[value_col].mean().reset_index()
    daily_avg['date'] = pd.to_datetime(daily_avg['date'])
    
    axes[1].plot(daily_avg['date'], daily_avg[value_col], marker='o', linewidth=2, markersize=4)
    axes[1].set_title(f'{metric_name.replace("_", " ").title()} - Daily Averages', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(f'Average {value_col.replace("_", " ").title()}')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save:
        filename = VISUALIZATIONS_DIR / f"{metric_name}_time_series.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
    
    plt.close()
    return fig


def plot_daily_metrics(save=True):
    """Create visualizations for daily summary metrics."""
    df = load_daily_summary()
    if df is None or df.empty:
        print("⚠️  No daily summary data available")
        return None
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Steps over time
    ax1 = fig.add_subplot(gs[0, 0])
    if 'steps' in df.columns:
        ax1.plot(df['date'], df['steps'], marker='o', linewidth=2, markersize=4)
        ax1.set_title('Daily Steps', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Steps')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Sleep duration
    ax2 = fig.add_subplot(gs[0, 1])
    if 'sleep_hours' in df.columns:
        ax2.plot(df['date'], df['sleep_hours'], marker='o', linewidth=2, markersize=4, color='purple')
        ax2.set_title('Sleep Duration', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Hours')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Calories
    ax3 = fig.add_subplot(gs[1, 0])
    if 'total_calories' in df.columns:
        ax3.plot(df['date'], df['total_calories'], marker='o', linewidth=2, markersize=4, color='orange', label='Total')
        if 'active_calories' in df.columns:
            ax3.plot(df['date'], df['active_calories'], marker='s', linewidth=2, markersize=3, color='red', label='Active')
        ax3.set_title('Calories', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Calories')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Stress levels
    ax4 = fig.add_subplot(gs[1, 1])
    if 'avg_stress' in df.columns:
        ax4.plot(df['date'], df['avg_stress'], marker='o', linewidth=2, markersize=4, color='red')
        ax4.set_title('Average Stress Level', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Stress Level')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Activity minutes
    ax5 = fig.add_subplot(gs[2, 0])
    activity_cols = ['moderate_intensity_min', 'vigorous_intensity_min', 'active_min']
    available_cols = [col for col in activity_cols if col in df.columns]
    if available_cols:
        for col in available_cols:
            ax5.plot(df['date'], df[col], marker='o', linewidth=2, markersize=3, label=col.replace('_', ' ').title())
        ax5.set_title('Activity Minutes', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Minutes')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Heart Rate stats
    ax6 = fig.add_subplot(gs[2, 1])
    hr_cols = ['resting_heart_rate', 'max_heart_rate']
    available_hr_cols = [col for col in hr_cols if col in df.columns]
    if available_hr_cols:
        for col in available_hr_cols:
            ax6.plot(df['date'], df[col], marker='o', linewidth=2, markersize=3, label=col.replace('_', ' ').title())
        ax6.set_title('Heart Rate Stats', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('BPM')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
    
    if save:
        filename = VISUALIZATIONS_DIR / "daily_metrics.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
    
    plt.close()
    return fig


def correlation_analysis(save=True):
    """Analyze correlations between different metrics."""
    df = load_daily_summary()
    if df is None or df.empty:
        print("⚠️  No daily summary data available for correlation analysis")
        return None
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove date-related columns
    numeric_cols = [col for col in numeric_cols if 'date' not in col.lower() and 'timestamp' not in col.lower()]
    
    if len(numeric_cols) < 2:
        print("⚠️  Not enough numeric columns for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Remove columns/rows where all values are 0 or NaN
    # Check each column: if all values (excluding diagonal) are 0 or NaN, remove it
    cols_to_keep = []
    for col in corr_matrix.columns:
        col_values = corr_matrix[col].drop(col)  # Exclude diagonal (self-correlation)
        # Keep column if it has at least one non-zero, non-NaN value
        if not col_values.isna().all() and not (col_values == 0).all():
            cols_to_keep.append(col)
    
    if len(cols_to_keep) < 2:
        print("⚠️  Not enough valid columns for correlation analysis after filtering")
        return None
    
    # Filter correlation matrix
    corr_matrix = corr_matrix.loc[cols_to_keep, cols_to_keep]
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Daily Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save:
        filename = VISUALIZATIONS_DIR / "correlation_matrix.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
    
    plt.close()
    
    # Print top correlations
    print("\n📊 Top Correlations:")
    print("-" * 50)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if not np.isnan(corr_matrix.iloc[i, j]):
                corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for col1, col2, corr_val in corr_pairs[:10]:
        print(f"  {col1.replace('_', ' ').title()} ↔ {col2.replace('_', ' ').title()}: {corr_val:.3f}")
    
    return corr_matrix


def generate_statistics(save=True):
    """Generate statistical summaries."""
    df = load_daily_summary()
    if df is None or df.empty:
        print("⚠️  No daily summary data available")
        return None
    
    stats_text = []
    stats_text.append("="*70)
    stats_text.append("GARMIN DATA STATISTICAL SUMMARY")
    stats_text.append("="*70)
    stats_text.append(f"\nDate Range: {df['date'].min().date()} to {df['date'].max().date()}")
    stats_text.append(f"Total Days: {len(df)}")
    stats_text.append("\n" + "-"*70)
    
    # Key metrics statistics
    key_metrics = {
        'Steps': 'steps',
        'Sleep Hours': 'sleep_hours',
        'Total Calories': 'total_calories',
        'Active Calories': 'active_calories',
        'Resting Heart Rate': 'resting_heart_rate',
        'Average Stress': 'avg_stress',
        'Max Stress': 'max_stress',
        'HRV': 'hrv',
        'Body Battery Max': 'body_battery_max',
    }
    
    stats_text.append("\nKEY METRICS SUMMARY:")
    stats_text.append("-"*70)
    
    for metric_name, col_name in key_metrics.items():
        if col_name in df.columns:
            values = df[col_name].dropna()
            if len(values) > 0:
                stats_text.append(f"\n{metric_name}:")
                stats_text.append(f"  Mean:   {values.mean():.2f}")
                stats_text.append(f"  Median: {values.median():.2f}")
                stats_text.append(f"  Min:    {values.min():.2f}")
                stats_text.append(f"  Max:    {values.max():.2f}")
                stats_text.append(f"  Std:    {values.std():.2f}")
    
    # Weekly patterns
    stats_text.append("\n" + "-"*70)
    stats_text.append("\nWEEKLY PATTERNS:")
    stats_text.append("-"*70)
    
    df['day_of_week'] = df['date'].dt.day_name()
    
    if 'steps' in df.columns:
        weekly_steps = df.groupby('day_of_week')['steps'].mean().sort_values(ascending=False)
        stats_text.append("\nAverage Steps by Day of Week:")
        for day, steps in weekly_steps.items():
            stats_text.append(f"  {day}: {steps:.0f} steps")
    
    if 'sleep_hours' in df.columns:
        weekly_sleep = df.groupby('day_of_week')['sleep_hours'].mean().sort_values(ascending=False)
        stats_text.append("\nAverage Sleep by Day of Week:")
        for day, sleep in weekly_sleep.items():
            stats_text.append(f"  {day}: {sleep:.2f} hours")
    
    stats_text.append("\n" + "="*70)
    
    stats_output = "\n".join(stats_text)
    print(stats_output)
    
    if save:
        filename = REPORTS_DIR / "summary_stats.txt"
        with open(filename, 'w') as f:
            f.write(stats_output)
        print(f"\n✅ Saved: {filename}")
    
    return stats_output


def create_dashboard(save=True):
    """Create interactive dashboard using Plotly."""
    df = load_daily_summary()
    if df is None or df.empty:
        print("⚠️  No daily summary data available")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Steps Over Time', 'Sleep Duration', 'Calories', 'Stress Levels', 'Heart Rate', 'Activity Minutes'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Steps
    if 'steps' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['steps'], mode='lines+markers', name='Steps',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
    
    # Sleep
    if 'sleep_hours' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sleep_hours'], mode='lines+markers', name='Sleep',
                      line=dict(color='purple', width=2)),
            row=1, col=2
        )
    
    # Calories
    if 'total_calories' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['total_calories'], mode='lines+markers', name='Total Calories',
                      line=dict(color='orange', width=2)),
            row=2, col=1
        )
        if 'active_calories' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['active_calories'], mode='lines+markers', name='Active Calories',
                          line=dict(color='red', width=2)),
                row=2, col=1
            )
    
    # Stress
    if 'avg_stress' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['avg_stress'], mode='lines+markers', name='Avg Stress',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
    
    # Heart Rate
    if 'resting_heart_rate' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['resting_heart_rate'], mode='lines+markers', name='Resting HR',
                      line=dict(color='green', width=2)),
            row=3, col=1
        )
    
    # Activity
    if 'moderate_intensity_min' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['moderate_intensity_min'], mode='lines+markers', name='Moderate',
                      line=dict(color='blue', width=2)),
            row=3, col=2
        )
        if 'vigorous_intensity_min' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['vigorous_intensity_min'], mode='lines+markers', name='Vigorous',
                          line=dict(color='red', width=2)),
                row=3, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Garmin Data Dashboard",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_yaxes(title_text="Steps", row=1, col=1)
    fig.update_yaxes(title_text="Hours", row=1, col=2)
    fig.update_yaxes(title_text="Calories", row=2, col=1)
    fig.update_yaxes(title_text="Stress Level", row=2, col=2)
    fig.update_yaxes(title_text="BPM", row=3, col=1)
    fig.update_yaxes(title_text="Minutes", row=3, col=2)
    
    if save:
        filename = REPORTS_DIR / "dashboard.html"
        fig.write_html(str(filename))
        print(f"✅ Saved: {filename}")
    
    return fig


def find_weekly_patterns(df=None):
    """
    Analyze day-of-week patterns in metrics.
    """
    if df is None:
        df = load_daily_summary()
    
    if df is None or df.empty:
        return None
    
    df = df.copy()
    df['day_of_week'] = df['date'].dt.day_name()
    df['day_num'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    
    patterns = {}
    metrics = ['steps', 'sleep_hours', 'avg_stress', 'total_calories', 'resting_heart_rate']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        weekly_avg = df.groupby('day_of_week')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Sort by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg['day_num'] = weekly_avg['day_of_week'].apply(lambda x: day_order.index(x) if x in day_order else 7)
        weekly_avg = weekly_avg.sort_values('day_num')
        
        patterns[metric] = {
            'daily_averages': weekly_avg.to_dict('records'),
            'best_day': weekly_avg.loc[weekly_avg['mean'].idxmax(), 'day_of_week'] if len(weekly_avg) > 0 else None,
            'worst_day': weekly_avg.loc[weekly_avg['mean'].idxmin(), 'day_of_week'] if len(weekly_avg) > 0 else None,
        }
    
    return patterns


def identify_outliers(df=None, std_threshold=2):
    """
    Identify outlier days where metrics are significantly different from normal.
    """
    if df is None:
        df = load_daily_summary()
    
    if df is None or df.empty:
        return None
    
    df = df.copy()
    outliers = []
    
    metrics = ['steps', 'sleep_hours', 'avg_stress', 'resting_heart_rate', 'body_battery_max']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].dropna()
        if len(values) < 10:
            continue
        
        mean = values.mean()
        std = values.std()
        
        # Find outliers (beyond std_threshold standard deviations)
        df[f'{metric}_zscore'] = (df[metric] - mean) / std if std > 0 else 0
        
        outlier_days = df[abs(df[f'{metric}_zscore']) > std_threshold].copy()
        
        for _, row in outlier_days.iterrows():
            outliers.append({
                'date': row['date'],
                'metric': metric,
                'value': row[metric],
                'mean': mean,
                'std': std,
                'z_score': row[f'{metric}_zscore'],
                'direction': 'high' if row[f'{metric}_zscore'] > 0 else 'low'
            })
    
    return sorted(outliers, key=lambda x: abs(x['z_score']), reverse=True)


def get_recovery_insights(df=None):
    """
    Analyze recovery patterns (sleep, HRV, body battery, RHR).
    """
    if df is None:
        df = load_daily_summary()
    
    if df is None or df.empty:
        return None
    
    df = df.sort_values('date')
    recent = df.tail(30)  # Last 30 days
    
    recovery_score = 0
    recovery_factors = []
    
    # Sleep quality
    if 'sleep_hours' in recent.columns:
        avg_sleep = recent['sleep_hours'].mean()
        if avg_sleep >= 7.5:
            recovery_score += 25
            recovery_factors.append(f"✅ Sleep: {avg_sleep:.1f}h (Good)")
        elif avg_sleep >= 6.5:
            recovery_score += 15
            recovery_factors.append(f"⚠️ Sleep: {avg_sleep:.1f}h (Fair)")
        else:
            recovery_score += 5
            recovery_factors.append(f"❌ Sleep: {avg_sleep:.1f}h (Poor)")
    
    # HRV stability
    if 'hrv' in recent.columns:
        hrv_values = recent['hrv'].dropna()
        if len(hrv_values) > 5:
            hrv_cv = (hrv_values.std() / hrv_values.mean() * 100) if hrv_values.mean() > 0 else 100
            if hrv_cv < 15:
                recovery_score += 25
                recovery_factors.append(f"✅ HRV: Stable (CV: {hrv_cv:.1f}%)")
            elif hrv_cv < 25:
                recovery_score += 15
                recovery_factors.append(f"⚠️ HRV: Moderate (CV: {hrv_cv:.1f}%)")
            else:
                recovery_score += 5
                recovery_factors.append(f"❌ HRV: Variable (CV: {hrv_cv:.1f}%)")
    
    # Body Battery recovery
    if 'body_battery_max' in recent.columns:
        avg_bb = recent['body_battery_max'].mean()
        if avg_bb >= 80:
            recovery_score += 25
            recovery_factors.append(f"✅ Body Battery: {avg_bb:.0f} (Excellent)")
        elif avg_bb >= 60:
            recovery_score += 15
            recovery_factors.append(f"⚠️ Body Battery: {avg_bb:.0f} (Good)")
        else:
            recovery_score += 5
            recovery_factors.append(f"❌ Body Battery: {avg_bb:.0f} (Poor)")
    
    # Stress management
    if 'avg_stress' in recent.columns:
        avg_stress = recent['avg_stress'].mean()
        if avg_stress < 25:
            recovery_score += 25
            recovery_factors.append(f"✅ Stress: {avg_stress:.0f} (Low)")
        elif avg_stress < 40:
            recovery_score += 15
            recovery_factors.append(f"⚠️ Stress: {avg_stress:.0f} (Moderate)")
        else:
            recovery_score += 5
            recovery_factors.append(f"❌ Stress: {avg_stress:.0f} (High)")
    
    return {
        'recovery_score': recovery_score,
        'factors': recovery_factors,
        'recommendation': get_recovery_recommendation(recovery_score)
    }


def get_recovery_recommendation(score):
    """Get recovery recommendation based on score."""
    if score >= 80:
        return "Your recovery is excellent! You're ready for high-intensity training."
    elif score >= 60:
        return "Good recovery. You can handle moderate to high intensity workouts."
    elif score >= 40:
        return "Fair recovery. Consider moderate intensity or active recovery."
    else:
        return "Poor recovery indicators. Prioritize rest and recovery activities."


def get_key_insights(df=None, days=30):
    """
    Generate key insights from the data.
    Returns a dictionary with insights categories.
    """
    if df is None:
        df = load_daily_summary()
    
    if df is None or df.empty:
        return None
    
    df = df.sort_values('date')
    recent = df.tail(days)
    
    insights = {
        'trends': [],
        'achievements': [],
        'concerns': [],
        'recommendations': []
    }
    
    # Trend detection: compare recent period to previous period
    if len(df) > days * 2:
        previous = df.iloc[-days*2:-days]
        
        # Steps trend
        if 'steps' in df.columns:
            recent_avg = recent['steps'].mean()
            previous_avg = previous['steps'].mean()
            change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            
            if abs(change) > 10:
                direction = "increased" if change > 0 else "decreased"
                insights['trends'].append({
                    'metric': 'Steps',
                    'text': f"Steps have {direction} by {abs(change):.1f}% compared to previous {days} days",
                    'change_pct': change,
                    'current_avg': recent_avg,
                    'previous_avg': previous_avg
                })
        
        # Sleep trend
        if 'sleep_hours' in df.columns:
            recent_avg = recent['sleep_hours'].mean()
            previous_avg = previous['sleep_hours'].mean()
            change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            
            if abs(change) > 5:
                direction = "improved" if change > 0 else "declined"
                insights['trends'].append({
                    'metric': 'Sleep',
                    'text': f"Sleep duration has {direction} by {abs(change):.1f}%",
                    'change_pct': change,
                    'current_avg': recent_avg,
                    'previous_avg': previous_avg
                })
        
        # Stress trend
        if 'avg_stress' in df.columns:
            recent_avg = recent['avg_stress'].mean()
            previous_avg = previous['avg_stress'].mean()
            change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            
            if abs(change) > 10:
                direction = "increased" if change > 0 else "decreased"
                insights['trends'].append({
                    'metric': 'Stress',
                    'text': f"Stress levels have {direction} by {abs(change):.1f}%",
                    'change_pct': change,
                    'current_avg': recent_avg,
                    'previous_avg': previous_avg
                })
        
        # Resting heart rate trend
        if 'resting_heart_rate' in df.columns:
            recent_avg = recent['resting_heart_rate'].mean()
            previous_avg = previous['resting_heart_rate'].mean()
            change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            
            if abs(change) > 3:
                direction = "increased" if change > 0 else "decreased"
                insights['trends'].append({
                    'metric': 'Resting HR',
                    'text': f"Resting heart rate has {direction} by {abs(change):.1f}%",
                    'change_pct': change,
                    'current_avg': recent_avg,
                    'previous_avg': previous_avg
                })
    
    # Achievements
    if 'steps' in recent.columns:
        max_steps = recent['steps'].max()
        days_over_10k = len(recent[recent['steps'] >= 10000])
        if days_over_10k >= days * 0.7:
            insights['achievements'].append(f"Hit 10k steps on {days_over_10k}/{days} days!")
        if max_steps >= 15000:
            insights['achievements'].append(f"Peak steps: {int(max_steps):,} steps")
    
    if 'sleep_hours' in recent.columns:
        avg_sleep = recent['sleep_hours'].mean()
        if avg_sleep >= 7.5:
            insights['achievements'].append(f"Averaging {avg_sleep:.1f}h sleep/night")
    
    # Concerns
    if 'sleep_hours' in recent.columns:
        avg_sleep = recent['sleep_hours'].mean()
        if avg_sleep < 6.5:
            insights['concerns'].append(f"Sleep below recommended (avg: {avg_sleep:.1f}h)")
            insights['recommendations'].append("Try to get 7-9 hours of sleep per night")
    
    if 'avg_stress' in recent.columns:
        avg_stress = recent['avg_stress'].mean()
        if avg_stress > 40:
            insights['concerns'].append(f"Elevated stress levels (avg: {avg_stress:.0f})")
            insights['recommendations'].append("Consider stress-reduction activities like meditation or yoga")
    
    if 'steps' in recent.columns:
        avg_steps = recent['steps'].mean()
        if avg_steps < 7500:
            insights['concerns'].append(f"Below activity target (avg: {int(avg_steps):,} steps/day)")
            insights['recommendations'].append("Aim for at least 7,500-10,000 steps per day")
    
    if 'resting_heart_rate' in recent.columns:
        # Check for elevated RHR trend
        rhr_values = recent['resting_heart_rate'].dropna()
        if len(rhr_values) >= 7:
            recent_7d = rhr_values.tail(7).mean()
            previous_7d = rhr_values.iloc[-14:-7].mean() if len(rhr_values) >= 14 else recent_7d
            if recent_7d > previous_7d + 3:
                insights['concerns'].append(f"Elevated resting heart rate (recent: {recent_7d:.0f} bpm)")
                insights['recommendations'].append("Elevated RHR may indicate fatigue or illness. Consider more rest.")
    
    return insights


def should_exclude_correlation(col1, col2):
    """
    Check if a correlation pair should be excluded because they're directly derived/calculated from each other.
    Returns True if the correlation should be excluded.
    """
    m1 = col1.lower()
    m2 = col2.lower()
    
    # List of obvious correlation pairs to exclude
    exclude_pairs = [
        # Stress-related (duration vs percentage)
        ('stress_duration', 'stress_percentage'),
        ('stress_duration_min', 'stress_percentage'),
        
        # Steps and distance (distance calculated from steps)
        ('steps', 'distance_km'),
        ('steps', 'distance'),
        
        # Calories (active is part of total, remaining is calculated from total/active)
        ('total_calories', 'active_calories'),
        ('active_calories', 'remaining_calories'),
        ('total_calories', 'remaining_calories'),
        
        # Floors (ascended vs descended are related metrics)
        ('floors_ascended', 'floors_descended'),
        
        # Sleep components (light sleep is part of total sleep)
        ('sleep_hours', 'light_sleep_min'),
        ('sleep_hours', 'deep_sleep_min'),
        ('sleep_hours', 'rem_sleep_min'),
        ('total_sleep_min', 'light_sleep_min'),
        ('total_sleep_min', 'deep_sleep_min'),
        ('total_sleep_min', 'rem_sleep_min'),
        
        # Activity minutes (moderate + vigorous = active)
        ('moderate_intensity_min', 'vigorous_intensity_min'),
        ('moderate_intensity_min', 'active_min'),
        ('vigorous_intensity_min', 'active_min'),
        
        # Heart rate zones (they sum up)
        ('hr_zones_1', 'hr_zones_2'),
        ('hr_zones_1', 'hr_zones_3'),
        ('hr_zones_2', 'hr_zones_3'),
        
        # Stress levels (min/max/avg are related)
        ('min_stress', 'max_stress'),
        ('avg_stress', 'max_stress'),
        ('min_stress', 'avg_stress'),
    ]
    
    # Check if this pair matches any exclusion pattern
    for pair in exclude_pairs:
        if (pair[0] in m1 and pair[1] in m2) or (pair[0] in m2 and pair[1] in m1):
            return True
    
    # Also exclude if one metric name is contained in the other (likely derived)
    if m1 in m2 or m2 in m1:
        # But allow some legitimate cases
        if not any(allowed in m1 or allowed in m2 for allowed in ['resting', 'max', 'avg', 'min', 'body_battery', 'hrv']):
            return True
    
    return False


def get_correlation_insights(df=None):
    """
    Get key correlation insights with interpretations.
    Excludes obvious correlations where metrics are directly derived from each other.
    """
    if df is None:
        df = load_daily_summary()
    
    if df is None or df.empty:
        return None
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if 'date' not in col.lower() and 'timestamp' not in col.lower()]
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    # Remove columns/rows where all values are 0 or NaN
    cols_to_keep = []
    for col in corr_matrix.columns:
        col_values = corr_matrix[col].drop(col)  # Exclude diagonal (self-correlation)
        # Keep column if it has at least one non-zero, non-NaN value
        if not col_values.isna().all() and not (col_values == 0).all():
            cols_to_keep.append(col)
    
    if len(cols_to_keep) < 2:
        return None
    
    # Filter correlation matrix
    corr_matrix = corr_matrix.loc[cols_to_keep, cols_to_keep]
    
    # Find strong correlations (abs > 0.5)
    strong_correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val) and abs(corr_val) > 0.5:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                
                # Skip obvious correlations
                if should_exclude_correlation(col1, col2):
                    continue
                
                # Interpret the correlation
                interpretation = interpret_correlation(col1, col2, corr_val)
                
                strong_correlations.append({
                    'metric1': col1,
                    'metric2': col2,
                    'correlation': corr_val,
                    'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate',
                    'direction': 'Positive' if corr_val > 0 else 'Negative',
                    'interpretation': interpretation
                })
    
    # Sort by correlation strength
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'correlation_matrix': corr_matrix,
        'strong_correlations': strong_correlations
    }


def interpret_correlation(metric1, metric2, corr_val):
    """
    Provide human-readable interpretation of correlation between two metrics.
    """
    m1 = metric1.lower()
    m2 = metric2.lower()
    
    # Known relationships
    interpretations = {
        ('sleep', 'body_battery'): "Better sleep leads to higher body battery recovery",
        ('sleep', 'resting_heart_rate'): "More sleep is associated with lower resting heart rate",
        ('sleep', 'hrv'): "Better sleep improves heart rate variability",
        ('steps', 'calories'): "More steps naturally burn more calories",
        ('steps', 'active'): "Higher step count means more active time",
        ('stress', 'sleep'): "Higher stress tends to reduce sleep quality" if corr_val < 0 else "Stress patterns relate to sleep",
        ('stress', 'body_battery'): "High stress drains body battery faster",
        ('stress', 'hrv'): "High stress reduces heart rate variability",
        ('resting_heart_rate', 'body_battery'): "Lower RHR correlates with better recovery",
        ('hrv', 'body_battery'): "Higher HRV indicates better recovery",
    }
    
    # Check for matching patterns
    for (key1, key2), text in interpretations.items():
        if (key1 in m1 and key2 in m2) or (key1 in m2 and key2 in m1):
            return text
    
    # Generic interpretation
    if corr_val > 0:
        return f"When {metric1.replace('_', ' ')} increases, {metric2.replace('_', ' ')} tends to increase"
    else:
        return f"When {metric1.replace('_', ' ')} increases, {metric2.replace('_', ' ')} tends to decrease"


def save_insights_json():
    """
    Generate and save insights to JSON for dashboard consumption.
    """
    import json
    
    df = load_daily_summary()
    if df is None or df.empty:
        return None
    
    insights_data = {
        'generated_at': datetime.now().isoformat(),
        'data_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat(),
            'days': len(df)
        },
        'key_insights': get_key_insights(df, days=30),
        'weekly_patterns': find_weekly_patterns(df),
        'outliers': identify_outliers(df),
        'recovery': get_recovery_insights(df),
        'correlations': None  # Will be added below
    }
    
    # Get correlations (convert to JSON-serializable format)
    corr_insights = get_correlation_insights(df)
    if corr_insights:
        insights_data['correlations'] = {
            'strong_correlations': corr_insights['strong_correlations'],
            'matrix': corr_insights['correlation_matrix'].to_dict()
        }
    
    # Save to JSON
    insights_file = REPORTS_DIR / "insights.json"
    with open(insights_file, 'w') as f:
        json.dump(insights_data, f, indent=2, default=str)
    
    print(f"✅ Insights saved to: {insights_file}")
    return insights_data


def main():
    """Main function to run all analyses."""
    print("="*70)
    print("📊 Garmin Data Analysis and Visualization")
    print("="*70)
    
    # Check if data exists
    daily_df = load_daily_summary()
    if daily_df is None or daily_df.empty:
        print("\n❌ No data found! Please run collect_garmin_data.py first.")
        return
    
    print(f"\n📋 Found data for {len(daily_df)} days")
    print(f"   Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
    
    # Generate statistics
    print("\n" + "="*70)
    print("1. Generating Statistics...")
    print("="*70)
    generate_statistics()
    
    # Create time-series plots
    print("\n" + "="*70)
    print("2. Creating Time-Series Visualizations...")
    print("="*70)
    time_series_metrics = ['heart_rate', 'stress', 'body_battery', 'respiration', 'spo2']
    for metric in time_series_metrics:
        plot_time_series(metric, days=30)
    
    # Create daily metrics plots
    print("\n" + "="*70)
    print("3. Creating Daily Metrics Visualizations...")
    print("="*70)
    plot_daily_metrics()
    
    # Correlation analysis
    print("\n" + "="*70)
    print("4. Correlation Analysis...")
    print("="*70)
    correlation_analysis()
    
    # Create interactive dashboard
    print("\n" + "="*70)
    print("5. Creating Interactive Dashboard...")
    print("="*70)
    create_dashboard()
    
    # Generate insights
    print("\n" + "="*70)
    print("6. Generating Advanced Insights...")
    print("="*70)
    save_insights_json()
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print(f"\n📁 Reports saved to: {REPORTS_DIR}")
    print(f"📊 Visualizations saved to: {VISUALIZATIONS_DIR}")
    print(f"\n💡 Open dashboard.html in your browser to view interactive charts!")


if __name__ == "__main__":
    main()
