import streamlit as st
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import numpy as np

# Page configuration - using fish emoji to match Victory Farms aquatic brand
st.set_page_config(
    page_title="Victory Farms Weather Forecast",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Victory Farms brand colors (green and blue)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5f2a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #00a8e8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .forecast-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-top: 4px solid #1a5f2a;
    }
    .metric-box {
        background: linear-gradient(135deg, #f0f9f0 0%, #e6f7ff 100%);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #1a5f2a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .location-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1a5f2a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #00a8e8;
    }
    .vf-brand-text {
        color: #1a5f2a;
        font-weight: bold;
    }
    .vf-blue {
        color: #00a8e8;
        font-weight: bold;
    }
    .sidebar-brand {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1a5f2a !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        background-color: #147a20 !important;
    }
    div[data-testid="stSidebarUserContent"] {
        background-color: #f8fdf8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f9f0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a5f2a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Location configuration - Victory Farms brand colors
LOCATIONS = {
    "Roo Farm": {"lat": 0.5603, "lon": 34.0623, "color": "#1a5f2a"},
    "KC-Kagano": {"lat": 2.3328, "lon": 29.0934, "color": "#00a8e8"},
    "KC-Kigembe": {"lat": 2.7334, "lon": 23.11111, "color": "#2e8b57"}
}

# Clean column name mapping
COLUMN_NAMES = {
    "wind_speed_180m": "Wind Speed",
    "wind_direction_180m": "Wind Direction",
    "temperature_180m": "Temperature",
    "rain": "Rain",
    "relative_humidity_2m": "Relative Humidity",
    "wind_gusts_10m": "Wind Gusts"
}

UNITS = {
    "Wind Speed": "m/s",
    "Wind Direction": "°",
    "Temperature": "°C",
    "Rain": "mm",
    "Relative Humidity": "%",
    "Wind Gusts": "m/s"
}

def hex_to_rgba(hex_color, alpha=0.15):
    """Convert hex color to rgba string"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

@st.cache_data(ttl=3600)
def fetch_forecast_data(location_name, forecast_days=7):
    """Fetch forecast data for a single location"""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    info = LOCATIONS[location_name]

    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "hourly": [
            "wind_speed_180m", "wind_direction_180m", "temperature_180m", 
            "rain", "relative_humidity_2m", "wind_gusts_10m"
        ],
        "timezone": "auto",
        "forecast_days": forecast_days,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "Date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert(response.Timezone().decode())
    }

    hourly_data["Location"] = location_name
    hourly_data["Latitude"] = response.Latitude()
    hourly_data["Longitude"] = response.Longitude()
    hourly_data["Wind Speed"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["Wind Direction"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["Temperature"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["Rain"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["Relative Humidity"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["Wind Gusts"] = hourly.Variables(5).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    df["Date"] = df["Date"].dt.tz_localize(None)

    return df

def create_wind_rose(df, location):
    """Create wind rose chart"""
    loc_df = df.copy()

    bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
    loc_df['Direction'] = pd.cut(loc_df['Wind Direction'], bins=bins, labels=labels, ordered=False)

    speed_bins = [0, 5, 10, 15, 20, 50]
    speed_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
    loc_df['Speed Range'] = pd.cut(loc_df['Wind Speed'], bins=speed_bins, labels=speed_labels)

    wind_data = loc_df.groupby(['Direction', 'Speed Range'], observed=True).size().reset_index(name='Count')

    fig = px.bar_polar(
        wind_data, 
        r="Count", 
        theta="Direction",
        color="Speed Range",
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Greens,
        title=f"Wind Rose - {location}"
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, wind_data['Count'].max() * 1.1])),
        showlegend=True,
        legend_title="Wind Speed (m/s)",
        height=400
    )
    return fig

def to_excel_single(df, location):
    """Convert single location dataframe to Excel bytes"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=f'{location} Forecast', index=False)
    return output.getvalue()

def to_excel_combined(dfs_dict):
    """Convert all locations to single Excel with separate sheets"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for location, df in dfs_dict.items():
            df.to_excel(writer, sheet_name=f'{location}', index=False)
    return output.getvalue()

# Sidebar with Victory Farms branding
with st.sidebar:
    st.markdown('<div class="sidebar-brand">', unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1a5f2a; margin-bottom: 0;'>🐟 <span class='vf-brand-text'>VICTORY</span><span class='vf-blue'>FARMS</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 0.8rem; margin-top: 0;'>Aquaculture Weather Intelligence</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Forecast Settings")

    st.markdown("### 📍 Farm Locations")
    selected_locations = []
    for loc in LOCATIONS.keys():
        if st.checkbox(loc, value=True, key=f"chk_{loc}"):
            selected_locations.append(loc)

    st.markdown("### 📅 Forecast Range")
    forecast_days = st.slider("Forecast Days", min_value=1, max_value=14, value=7, 
                             help="Number of days to forecast ahead")

    st.markdown("---")
    fetch_button = st.button("🚀 Get Forecast", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #1a5f2a; font-weight: 600;'>Victory Farms Ltd</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00a8e8; font-size: 0.8rem;'>Technology & Innovation</p>", unsafe_allow_html=True)

# Main content with Victory Farms branding
st.markdown('<div style="text-align: center; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
st.markdown("<h1 style='color: #1a5f2a; font-size: 2.5rem; margin-bottom: 0;'>🐟 <span class='vf-brand-text'>VICTORY</span><span class='vf-blue'>FARMS</span></h1>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header" style="margin-top: 0; font-size: 1.8rem;">Weather Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">7-Day hourly forecast for aquaculture operations</div>', unsafe_allow_html=True)

# Fetch and display data
if fetch_button or 'forecast_data' in st.session_state:
    if fetch_button:
        if not selected_locations:
            st.error("Please select at least one location!")
            st.stop()

        forecast_data = {}
        progress_bar = st.progress(0)

        for i, location in enumerate(selected_locations):
            with st.spinner(f"🌤️ Fetching forecast for {location}..."):
                try:
                    df = fetch_forecast_data(location, forecast_days)
                    forecast_data[location] = df
                    progress_bar.progress((i + 1) / len(selected_locations))
                except Exception as e:
                    st.error(f"❌ Error fetching {location}: {str(e)}")
                    st.stop()

        st.session_state['forecast_data'] = forecast_data
        st.success(f"✅ Forecast loaded for {len(forecast_data)} locations!")
    else:
        forecast_data = st.session_state['forecast_data']

    # Summary metrics across all locations
    st.markdown("### 📊 Forecast Summary")

    summary_cols = st.columns(len(forecast_data))
    for i, (location, df) in enumerate(forecast_data.items()):
        with summary_cols[i]:
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #1a5f2a; margin-top: 0;'>📍 {location}</h4>")

            avg_temp = df['Temperature'].mean()
            max_wind = df['Wind Speed'].max()
            total_rain = df['Rain'].sum()

            st.markdown(f"🌡️ Avg Temp: <b>{avg_temp:.1f}°C</b>", unsafe_allow_html=True)
            st.markdown(f"💨 Max Wind: <b>{max_wind:.1f} m/s</b>", unsafe_allow_html=True)
            st.markdown(f"🌧️ Total Rain: <b>{total_rain:.1f} mm</b>", unsafe_allow_html=True)
            st.markdown(f"📅 <small>{df['Date'].min().strftime('%b %d')} - {df['Date'].max().strftime('%b %d')}</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Individual location forecasts
    st.markdown("### 📍 Location Forecasts")

    for location, df in forecast_data.items():
        with st.container():
            st.markdown(f'<div class="forecast-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="location-header">🐟 {location}</div>', unsafe_allow_html=True)

            info = LOCATIONS[location]
            st.markdown(f"<p style='color: #666;'><b>Coordinates:</b> {info['lat']}°N, {info['lon']}°E | <b>Forecast Period:</b> {forecast_days} days</p>", unsafe_allow_html=True)

            loc_tab1, loc_tab2, loc_tab3, loc_tab4 = st.tabs(["📈 Charts", "🧭 Wind Analysis", "📋 Data Table", "📥 Export"])

            with loc_tab1:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("🌡️ Temperature (°C)", "💨 Wind Speed & Gusts (m/s)", "🌧️ Rain & Humidity"),
                    row_heights=[0.35, 0.35, 0.3]
                )

                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Temperature'], name="Temperature",
                              line=dict(color="#1a5f2a", width=2), mode='lines', fill='tozeroy',
                              fillcolor="rgba(26, 95, 42, 0.15)"),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Wind Speed'], name="Wind Speed",
                              line=dict(color="#00a8e8", width=2), mode='lines'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Wind Gusts'], name="Wind Gusts",
                              line=dict(color="#ff6b6b", width=1, dash='dash'), mode='lines'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Bar(x=df['Date'], y=df['Rain'], name="Rain",
                          marker_color="#00a8e8", opacity=0.6),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Relative Humidity'], name="Humidity",
                              line=dict(color="#1a5f2a", width=2), mode='lines', yaxis="y4"),
                    row=3, col=1
                )

                fig.update_layout(
                    height=700,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                    template="plotly_white",
                    title_text=f"{location} - Complete Forecast Overview",
                    title_font_color="#1a5f2a"
                )

                fig.update_yaxes(title_text="°C", row=1, col=1)
                fig.update_yaxes(title_text="m/s", row=2, col=1)
                fig.update_yaxes(title_text="mm", row=3, col=1)
                fig.update_yaxes(title_text="%", row=3, col=1, overlaying="y3", side="right", showgrid=False)

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Detailed Variable Views:**")
                var_cols = st.columns(3)

                variables = [
                    ("Temperature", "°C", "🌡️", "#1a5f2a"),
                    ("Wind Speed", "m/s", "💨", "#00a8e8"),
                    ("Rain", "mm", "🌧️", "#2e8b57")
                ]

                for col, (var, unit, emoji, color) in zip(var_cols, variables):
                    with col:
                        fig_var = px.area(
                            df, x="Date", y=var,
                            title=f"{emoji} {var}",
                            labels={var: f"{var} ({unit})"},
                            color_discrete_sequence=[color],
                            template="plotly_white"
                        )
                        fig_var.update_layout(height=250, showlegend=False)
                        rgba_fill = hex_to_rgba(color, 0.15)
                        fig_var.update_traces(fill='tozeroy', fillcolor=rgba_fill)
                        st.plotly_chart(fig_var, use_container_width=True)

            with loc_tab2:
                wind_cols = st.columns([2, 1])

                with wind_cols[0]:
                    fig_rose = create_wind_rose(df, location)
                    st.plotly_chart(fig_rose, use_container_width=True)

                with wind_cols[1]:
                    st.markdown("<h4 style='color: #1a5f2a;'>🧭 Wind Statistics</h4>", unsafe_allow_html=True)
                    st.markdown(f"- **Avg Speed:** {df['Wind Speed'].mean():.1f} m/s")
                    st.markdown(f"- **Max Speed:** {df['Wind Speed'].max():.1f} m/s")
                    st.markdown(f"- **Max Gusts:** {df['Wind Gusts'].max():.1f} m/s")
                    st.markdown(f"- **Prevailing Direction:** {df['Wind Direction'].mode().iloc[0]:.0f}°")

                    st.markdown("<h4 style='color: #1a5f2a; margin-top: 1.5rem;'>⚠️ Alerts</h4>", unsafe_allow_html=True)
                    high_wind = df[df['Wind Speed'] > 15]
                    if len(high_wind) > 0:
                        st.warning(f"⚠️ {len(high_wind)} hours with wind >15 m/s")
                    else:
                        st.success("✅ No high wind alerts")

                    high_gust = df[df['Wind Gusts'] > 20]
                    if len(high_gust) > 0:
                        st.error(f"🚨 {len(high_gust)} hours with gusts >20 m/s")
                    else:
                        st.success("✅ No gust alerts")

                    fig_dir = px.scatter(
                        df, x="Date", y="Wind Direction",
                        color="Wind Speed",
                        color_continuous_scale="Greens",
                        title="Wind Direction Over Time",
                        labels={"Wind Direction": "Direction (°)"},
                        template="plotly_white"
                    )
                    fig_dir.update_layout(height=280)
                    st.plotly_chart(fig_dir, use_container_width=True)

            with loc_tab3:
                display_df = df.copy()
                st.dataframe(
                    display_df.sort_values("Date"),
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                        "Wind Speed": st.column_config.NumberColumn("Wind Speed", format="%.1f m/s"),
                        "Wind Direction": st.column_config.NumberColumn("Wind Direction", format="%.0f°"),
                        "Temperature": st.column_config.NumberColumn("Temperature", format="%.1f°C"),
                        "Rain": st.column_config.NumberColumn("Rain", format="%.1f mm"),
                        "Relative Humidity": st.column_config.NumberColumn("Humidity", format="%.0f%%"),
                        "Wind Gusts": st.column_config.NumberColumn("Wind Gusts", format="%.1f m/s")
                    }
                )

            with loc_tab4:
                st.markdown("<h4 style='color: #1a5f2a;'>📥 Download Forecast Data</h4>", unsafe_allow_html=True)

                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📄 CSV - {location}",
                        data=csv,
                        file_name=f"{location.replace(' ', '_')}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col_dl2:
                    excel_data = to_excel_single(df, location)
                    st.download_button(
                        label=f"📊 Excel - {location}",
                        data=excel_data,
                        file_name=f"{location.replace(' ', '_')}_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

    # Combined export section
    st.markdown("### 📥 Combined Export")
    st.markdown("<p style='color: #666;'>Download all locations in a single file:</p>", unsafe_allow_html=True)

    comb_col1, comb_col2 = st.columns(2)

    with comb_col1:
        all_dfs = []
        for loc, df in forecast_data.items():
            all_dfs.append(df)
        combined_csv = pd.concat(all_dfs, ignore_index=True).to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📄 Combined CSV (All Locations)",
            data=combined_csv,
            file_name=f"VF_All_Locations_Forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with comb_col2:
        excel_combined = to_excel_combined(forecast_data)
        st.download_button(
            label="📊 Combined Excel (Separate Sheets)",
            data=excel_combined,
            file_name=f"VF_All_Locations_Forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

else:
    st.info("👈 Select your farm locations in the sidebar and click **Get Forecast** to view the weather forecast!")

    st.markdown("### 🎯 Dashboard Features")
    feat_cols = st.columns(3)
    features = [
        ("📈 Individual Forecasts", "Separate detailed 7-day forecasts for each farm location with interactive charts"),
        ("🧭 Wind Analysis", "Wind rose diagrams, directional patterns, and automated alerts for high winds/gusts"),
        ("📥 Flexible Export", "Download each location separately or combined in one Excel file with clean column names")
    ]
    for col, (title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"<h4 style='color: #1a5f2a;'>{title}</h4>", unsafe_allow_html=True)
            st.markdown(f"<small style='color: #666;'>{desc}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #999; margin-top: 2rem;'><small>🐟 Victory Farms Ltd - Technology & Innovation - Power BI Weather Intelligence</small></p>", unsafe_allow_html=True)