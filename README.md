# Garmin Data Dashboard

A Streamlit dashboard for visualizing your Garmin fitness data with AI-powered workout creation and goal tracking.

## Screenshots

| Home Dashboard | Goals & Challenges |
|:-:|:-:|
| ![Home](example_images/home.png) | ![Goals](example_images/goals.png) |

| Data Insights | Workouts |
|:-:|:-:|
| ![Insights](example_images/insights.png) | ![Workouts](example_images/workout.png) |

## Deploy Your Own Dashboard

### Option 1: Streamlit Community Cloud (Recommended)

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub

3. **Create a new app**:
   - Select your forked repository
   - Branch: `main`
   - Main file: `streamlit_dashboard.py`

4. **Add your secrets** in the Streamlit Cloud dashboard:
   - Go to App Settings → Secrets
   - Add the following:
   ```toml
   GARMIN_EMAIL = "your_email@example.com"
   GARMIN_PASSWORD = "your_garmin_password"
   DASHBOARD_PASSWORD = "your_dashboard_password"
   
   # Optional - for AI chatbot
   MISTRAL_AI_API_KEY = "your_mistral_api_key"
   ```

5. **Deploy** - Your dashboard will be live at `https://your-app-name.streamlit.app`

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/garmin_streamlit_dashboard.git
cd garmin_streamlit_dashboard

# Install dependencies
pip install -r requirements.txt

# Create .env file with your credentials
cat > .env << EOF
GARMIN_EMAIL=your_email@example.com
GARMIN_PASSWORD=your_garmin_password
DASHBOARD_PASSWORD=your_dashboard_password
EOF

# Launch the dashboard
streamlit run streamlit_dashboard.py
```

The dashboard opens at `http://localhost:8501`. Enter your `DASHBOARD_PASSWORD` to access.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GARMIN_EMAIL` | Yes | Your Garmin Connect email |
| `GARMIN_PASSWORD` | Yes | Your Garmin Connect password |
| `DASHBOARD_PASSWORD` | Yes | Password to access the dashboard |
| `MISTRAL_AI_API_KEY` | No | For AI chatbot features ([get key](https://console.mistral.ai/)) |
| `CHATBOT_NAME` | No | Name for AI assistant (default: "Lixxi") |

## Features

- **Daily Metrics**: Steps, sleep, heart rate, stress, body battery, HRV
- **High-Resolution Data**: Time-series analysis at 2-3 minute intervals
- **Workout Management**: Create workouts and sync to your Garmin watch
- **Goals & Challenges**: Set and track fitness goals with progress visualization
- **Data Insights**: Correlation analysis, recovery scoring, trend detection
- **AI Assistant**: Mistral-powered chatbot with workout creation via natural language

## Troubleshooting

**"Could not connect to Garmin"**
- Verify your Garmin credentials are correct
- Garmin may require re-authentication (tokens expire after ~1 year)

**"No data found"**
- The dashboard fetches data from Garmin Connect on startup
- Ensure your Garmin device has synced with Garmin Connect

**"Incorrect password"**
- Check your `DASHBOARD_PASSWORD` in secrets/`.env` (case-sensitive)

## License

For personal use. Garmin Connect API usage subject to Garmin's Terms of Service.
