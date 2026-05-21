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

st.set_page_config(
    page_title="Victory Farms Weather Forecast",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .vf-brand-text { color: #1a5f2a; font-weight: bold; }
    .vf-blue { color: #00a8e8; font-weight: bold; }

    section[data-testid="stSidebar"] { background-color: #262730 !important; }
    section[data-testid="stSidebar"] > div { background-color: #262730 !important; }
    div[data-testid="stSidebarUserContent"] { background-color: #262730 !important; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 { color: #ffffff !important; font-weight: 600 !important; }
    section[data-testid="stSidebar"] p { color: #ffffff !important; }
    section[data-testid="stSidebar"] label { color: #ffffff !important; font-weight: 500 !important; }
    section[data-testid="stSidebar"] .stMarkdown { color: #ffffff !important; }
    .sidebar-brand { text-align: center; padding: 1rem 0; border-bottom: 1px solid #404040; margin-bottom: 1rem; }
    .sidebar-brand h2 { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #ff4b4b !important; color: #ffffff !important;
        border-radius: 8px !important; font-weight: 600 !important; border: none !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover { background-color: #ff6b6b !important; }
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] input:checked + div {
        background-color: #ff4b4b !important; border-color: #ff4b4b !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] label span { color: #ffffff !important; font-weight: 500 !important; }
    section[data-testid="stSidebar"] hr { border-color: #404040 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f9f0; border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #1a5f2a !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

LOCATIONS = {
    "Roo Farm":    {"lat": 0.5603,  "lon": 34.0623,   "color": "#1a5f2a"},
    "KC-Kagano":   {"lat": 2.3328,  "lon": 29.0934,   "color": "#00a8e8"},
    "KC-Kigembe":  {"lat": 2.7334,  "lon": 23.11111,  "color": "#2e8b57"}
}

UNITS = {
    "Wind Speed": "m/s", "Wind Direction": "°", "Temperature": "°C",
    "Rain": "mm", "Relative Humidity": "%", "Wind Gusts": "m/s"
}

def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def sparse_labels(series, interval, fmt="{:.1f}"):
    """
    ALL data points are plotted on the line/markers.
    Text labels only appear every `interval` points so they don't overlap.
    Empty string = no label drawn, but the data point still exists on the chart.
    """
    return [fmt.format(v) if i % interval == 0 else "" for i, v in enumerate(series)]

def label_interval(df):
    """
    Pick a label print interval based on dataset length so labels never crowd.
    1 day  → every 3 hrs  | 2-3 days → every 6 hrs
    4-7 days → every 12 hrs | 8-14 days → every 24 hrs
    """
    n_hours = len(df)
    if n_hours <= 24:
        return 3
    elif n_hours <= 72:
        return 6
    elif n_hours <= 168:
        return 12
    else:
        return 24

@st.cache_data(ttl=3600)
def fetch_forecast_data(location_name, forecast_days=7):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url  = "https://api.open-meteo.com/v1/forecast"
    info = LOCATIONS[location_name]
    params = {
        "latitude": info["lat"], "longitude": info["lon"],
        "hourly": ["wind_speed_180m","wind_direction_180m","temperature_180m",
                   "rain","relative_humidity_2m","wind_gusts_10m"],
        "timezone": "auto",
        "forecast_days": forecast_days,
    }
    responses = openmeteo.weather_api(url, params=params)
    response  = responses[0]
    hourly    = response.Hourly()

    hourly_data = {
        "Date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert(response.Timezone().decode())
    }
    hourly_data["Location"]         = location_name
    hourly_data["Latitude"]         = response.Latitude()
    hourly_data["Longitude"]        = response.Longitude()
    hourly_data["Wind Speed"]       = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["Wind Direction"]   = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["Temperature"]      = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["Rain"]             = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["Relative Humidity"]= hourly.Variables(4).ValuesAsNumpy()
    hourly_data["Wind Gusts"]       = hourly.Variables(5).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    df["Date"] = df["Date"].dt.tz_localize(None)
    return df

def create_wind_rose(df, location):
    loc_df = df.copy()
    bins   = [0,22.5,67.5,112.5,157.5,202.5,247.5,292.5,337.5,360]
    labels = ['N','NE','E','SE','S','SW','W','NW','N']
    loc_df['Direction']  = pd.cut(loc_df['Wind Direction'], bins=bins, labels=labels, ordered=False)
    speed_bins   = [0,5,10,15,20,50]
    speed_labels = ['0-5','5-10','10-15','15-20','20+']
    loc_df['Speed Range'] = pd.cut(loc_df['Wind Speed'], bins=speed_bins, labels=speed_labels)
    wind_data = loc_df.groupby(['Direction','Speed Range'], observed=True).size().reset_index(name='Count')
    fig = px.bar_polar(wind_data, r="Count", theta="Direction", color="Speed Range",
                       template="plotly_white", color_discrete_sequence=px.colors.sequential.Greens,
                       title=f"Wind Rose - {location}")
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, wind_data['Count'].max()*1.1])),
                      showlegend=True, legend_title="Wind Speed (m/s)", height=400)
    return fig

def to_excel_single(df, location):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=f'{location} Forecast', index=False)
    return output.getvalue()

def to_excel_combined(dfs_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for location, df in dfs_dict.items():
            df.to_excel(writer, sheet_name=f'{location}', index=False)
    return output.getvalue()

# ── Sidebar ──────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#ffffff;margin-bottom:0;font-size:1.3rem;'>🐟 VICTORY<span style='color:#00a8e8;'>FARMS</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaaaaa;font-size:0.75rem;margin-top:0.2rem;'>Aquaculture Weather Intelligence</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Controls")
    st.markdown("### 📍 Select Locations")
    selected_locations = []
    for loc in LOCATIONS.keys():
        if st.checkbox(loc, value=True, key=f"chk_{loc}"):
            selected_locations.append(loc)

    st.markdown("### 📅 Time Range")
    forecast_days = st.slider("Forecast Days", min_value=1, max_value=14, value=7,
                              help="Number of days to forecast ahead")

    st.markdown("---")
    st.markdown("### 🔄 Actions")
    fetch_button = st.button("🚀 Fetch Weather Data", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#ffffff;font-weight:600;font-size:1rem;'>Victory Farms Ltd</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#aaaaaa;font-size:0.85rem;font-style:italic;'>Technology & Innovation</p>", unsafe_allow_html=True)

# ── Header ────────────────
st.markdown('<div style="text-align:center;margin-bottom:0.5rem;">', unsafe_allow_html=True)
st.markdown("<h1 style='color:#1a5f2a;font-size:2.5rem;margin-bottom:0;'>🐟 <span class='vf-brand-text'>VICTORY</span><span class='vf-blue'>FARMS</span></h1>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header" style="margin-top:0;font-size:1.8rem;">Weather Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hourly forecast for aquaculture operations</div>', unsafe_allow_html=True)

# ── Main 

if fetch_button or 'forecast_data' in st.session_state:
    if fetch_button:
        if not selected_locations:
            st.error("Please select at least one location!")
            st.stop()

        forecast_data = {}
        progress_bar  = st.progress(0)
        for i, location in enumerate(selected_locations):
            with st.spinner(f"🌤️ Fetching forecast for {location}..."):
                try:
                    df = fetch_forecast_data(location, forecast_days)
                    forecast_data[location] = df
                    progress_bar.progress((i+1)/len(selected_locations))
                except Exception as e:
                    st.error(f"❌ Error fetching {location}: {str(e)}")
                    st.stop()
        st.session_state['forecast_data']     = forecast_data
        st.session_state['fetched_days']      = forecast_days
        st.session_state['fetched_locations'] = selected_locations
        st.success(f"Forecast loaded for {len(forecast_data)} location(s) — {forecast_days} day(s)!")
    else:
        forecast_data = st.session_state['forecast_data']

    # Warn if slider / locations changed since last fetch
    fetched_days      = st.session_state.get('fetched_days', forecast_days)
    fetched_locations = st.session_state.get('fetched_locations', selected_locations)
    if forecast_days != fetched_days or selected_locations != fetched_locations:
        st.warning(
            f"Settings changed (currently showing **{fetched_days}-day** data for "
            f"**{', '.join(fetched_locations)}**). "
            "Click **Fetch Weather Data** in the sidebar to reload with the new settings."
        )

    # ── Summary metrics 
    st.markdown("### 📊 Forecast Summary")
    summary_cols = st.columns(len(forecast_data))
    for i, (location, df) in enumerate(forecast_data.items()):
        with summary_cols[i]:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#1a5f2a;margin-top:0;'>📍 {location}</h4>")
            st.markdown(f"🌡️ Avg Temp: <b>{df['Temperature'].mean():.1f}°C</b>", unsafe_allow_html=True)
            st.markdown(f"💨 Max Wind: <b>{df['Wind Speed'].max():.1f} m/s</b>", unsafe_allow_html=True)
            st.markdown(f"🌧️ Total Rain: <b>{df['Rain'].sum():.1f} mm</b>", unsafe_allow_html=True)
            st.markdown(f"📅 <small>{df['Date'].min().strftime('%b %d')} – {df['Date'].max().strftime('%b %d')}</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📍 Location Forecasts")

    for location, df in forecast_data.items():
        with st.container():
            st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="location-header">🐟 {location}</div>', unsafe_allow_html=True)

            info = LOCATIONS[location]
            st.markdown(f"<p style='color:#666;'><b>Coordinates:</b> {info['lat']}°N, {info['lon']}°E | "
                        f"<b>Forecast Period:</b> {forecast_days} days ({len(df)} hourly points)</p>",
                        unsafe_allow_html=True)

            loc_tab1, loc_tab2, loc_tab3, loc_tab4 = st.tabs(["📈 Charts","🧭 Wind Analysis","📋 Data Table","📥 Export"])

            with loc_tab1:
                # ── Adaptive label interval ────────────────
                lbl_every = label_interval(df)   # e.g. 12 for a 7-day dataset
                # x-axis ticks: show every 6 hrs (denser than labels, lighter than every hour)
                tick_step  = max(6, lbl_every // 2)
                tick_vals  = df['Date'].iloc[::tick_step].tolist()
                num_days   = (df['Date'].max() - df['Date'].min()).days + 1
                if num_days <= 2:
                    tick_text = [d.strftime("%H:%M") for d in tick_vals]
                    x_title   = "Hour"
                else:
                    tick_text = [d.strftime("%b %d\n%H:%M") for d in tick_vals]
                    x_title   = "Date & Hour"

                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("🌡️ Temperature (°C)","💨 Wind Speed & Gusts (m/s)","🌧️ Rain & Humidity"),
                    row_heights=[0.35, 0.35, 0.3]
                )

                # ── Temperature ─────────
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['Temperature'], name="Temperature",
                    line=dict(color="#1a5f2a", width=1.5),
                    mode='lines+markers+text',
                    # ALL hourly points plotted; labels only every lbl_every hours
                    text=sparse_labels(df['Temperature'], lbl_every, "{:.1f}"),
                    textposition="top center",
                    textfont=dict(size=8, color="#1a5f2a"),
                    marker=dict(size=3, color="#1a5f2a"),
                    fill='tozeroy', fillcolor="rgba(26,95,42,0.15)"
                ), row=1, col=1)

                # ── Wind Speed ───────────
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['Wind Speed'], name="Wind Speed",
                    line=dict(color="#00a8e8", width=1.5),
                    mode='lines+markers+text',
                    text=sparse_labels(df['Wind Speed'], lbl_every, "{:.1f}"),
                    textposition="top center",
                    textfont=dict(size=8, color="#00a8e8"),
                    marker=dict(size=3, color="#00a8e8")
                ), row=2, col=1)

                # ── Wind Gusts (labels offset by half-interval so they don't clash with Wind Speed labels) ──
                gust_offset = lbl_every // 2
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['Wind Gusts'], name="Wind Gusts",
                    line=dict(color="#ff6b6b", width=1, dash='dash'),
                    mode='lines+markers+text',
                    text=[f"{v:.1f}" if i % lbl_every == gust_offset else ""
                          for i, v in enumerate(df['Wind Gusts'])],
                    textposition="bottom center",
                    textfont=dict(size=7, color="#ff6b6b"),
                    marker=dict(size=2, color="#ff6b6b")
                ), row=2, col=1)

                # Rain bars
                fig.add_trace(go.Bar(
                    x=df['Date'], y=df['Rain'], name="Rain",
                    marker_color="#00a8e8", opacity=0.6,
                    # Label only significant rain at label intervals
                    text=[f"{v:.1f}" if (v > 0.1 and i % lbl_every == 0) else ""
                          for i, v in enumerate(df['Rain'])],
                    textposition="outside",
                    textfont=dict(size=8, color="#005f8a")
                ), row=3, col=1)

                # ── Humidity line on secondary y-axis (right side) 
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['Relative Humidity'], name="Humidity",
                    line=dict(color="#2e8b57", width=1.5),
                    mode='lines+markers+text',
                    text=sparse_labels(df['Relative Humidity'], lbl_every, "{:.0f}"),
                    textposition="top center",
                    textfont=dict(size=8, color="#2e8b57"),
                    marker=dict(size=3, color="#2e8b57"),
                    yaxis="y4"
                ), row=3, col=1)

                fig.update_layout(
                    height=750,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
                    template="plotly_white",
                    title_text=f"{location} — Hourly Forecast ({len(df)} data points)",
                    title_font_color="#1a5f2a",
                    margin=dict(t=80, b=60)
                )
                for row in [1, 2, 3]:
                    fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                                     tickangle=-45, tickfont=dict(size=9), row=row, col=1)
                fig.update_xaxes(title_text=x_title, row=3, col=1)
                fig.update_yaxes(title_text="°C",  row=1, col=1)
                fig.update_yaxes(title_text="m/s", row=2, col=1)
                fig.update_yaxes(title_text="mm",  row=3, col=1)
                fig.update_yaxes(title_text="%",   row=3, col=1, overlaying="y3", side="right", showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

                # ── Detailed mini-charts (no text labels — hover for values) ──
                st.markdown("**Detailed Variable Views** *(hover for exact hourly values)*")
                var_cols  = st.columns(3)
                variables = [
                    ("Temperature","°C","🌡️","#1a5f2a"),
                    ("Wind Speed","m/s","💨","#00a8e8"),
                    ("Rain","mm","🌧️","#2e8b57")
                ]
                for col, (var, unit, emoji, color) in zip(var_cols, variables):
                    with col:
                        fig_var = px.area(df, x="Date", y=var,
                                          title=f"{emoji} {var} ({unit})",
                                          labels={var: f"{var} ({unit})"},
                                          color_discrete_sequence=[color],
                                          template="plotly_white")
                        fig_var.update_layout(height=220, showlegend=False, margin=dict(t=40,b=30,l=40,r=10))
                        fig_var.update_traces(
                            fill='tozeroy', fillcolor=hex_to_rgba(color, 0.2),
                            mode='lines+markers',       # no text — hover shows exact value
                            marker=dict(size=3, color=color),
                            line=dict(width=1.2),
                            hovertemplate=f"<b>%{{x|%b %d %H:%M}}</b><br>{var}: %{{y:.2f}} {unit}<extra></extra>"
                        )
                        fig_var.update_xaxes(tickmode="array", tickvals=tick_vals,
                                             ticktext=tick_text, tickangle=-45, tickfont=dict(size=8))
                        st.plotly_chart(fig_var, use_container_width=True)

            with loc_tab2:
                wind_cols = st.columns([2,1])
                with wind_cols[0]:
                    st.plotly_chart(create_wind_rose(df, location), use_container_width=True)
                with wind_cols[1]:
                    st.markdown("<h4 style='color:#1a5f2a;'>🧭 Wind Statistics</h4>", unsafe_allow_html=True)
                    st.markdown(f"- **Avg Speed:** {df['Wind Speed'].mean():.1f} m/s")
                    st.markdown(f"- **Max Speed:** {df['Wind Speed'].max():.1f} m/s")
                    st.markdown(f"- **Max Gusts:** {df['Wind Gusts'].max():.1f} m/s")
                    st.markdown(f"- **Prevailing Direction:** {df['Wind Direction'].mode().iloc[0]:.0f}°")
                    st.markdown("<h4 style='color:#1a5f2a;margin-top:1.5rem;'>⚠️ Alerts</h4>", unsafe_allow_html=True)
                    high_wind = df[df['Wind Speed'] > 15]
                    st.warning(f"⚠️ {len(high_wind)} hours with wind >15 m/s") if len(high_wind) else st.success("✅ No high wind alerts")
                    high_gust = df[df['Wind Gusts'] > 20]
                    st.error(f"🚨 {len(high_gust)} hours with gusts >20 m/s") if len(high_gust) else st.success("✅ No gust alerts")

                    fig_dir = px.scatter(df, x="Date", y="Wind Direction", color="Wind Speed",
                                        color_continuous_scale="Greens", title="Wind Direction Over Time",
                                        labels={"Wind Direction":"Direction (°)"}, template="plotly_white")
                    fig_dir.update_layout(height=280)
                    fig_dir.update_traces(
                        mode='markers', marker=dict(size=4),
                        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Direction: %{y:.0f}°<extra></extra>"
                    )
                    fig_dir.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=-45, tickfont=dict(size=8))
                    st.plotly_chart(fig_dir, use_container_width=True)

            with loc_tab3:
                st.dataframe(
                    df.sort_values("Date"),
                    use_container_width=True, height=400,
                    column_config={
                        "Date":             st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                        "Wind Speed":       st.column_config.NumberColumn("Wind Speed",   format="%.1f m/s"),
                        "Wind Direction":   st.column_config.NumberColumn("Wind Dir",     format="%.0f°"),
                        "Temperature":      st.column_config.NumberColumn("Temperature",  format="%.1f°C"),
                        "Rain":             st.column_config.NumberColumn("Rain",         format="%.1f mm"),
                        "Relative Humidity":st.column_config.NumberColumn("Humidity",     format="%.0f%%"),
                        "Wind Gusts":       st.column_config.NumberColumn("Wind Gusts",   format="%.1f m/s")
                    }
                )

            with loc_tab4:
                st.markdown("<h4 style='color:#1a5f2a;'>📥 Download Forecast Data</h4>", unsafe_allow_html=True)
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label=f"📄 CSV — {location}",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"{location.replace(' ','_')}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv", use_container_width=True
                    )
                with col_dl2:
                    st.download_button(
                        label=f"📊 Excel — {location}",
                        data=to_excel_single(df, location),
                        file_name=f"{location.replace(' ','_')}_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

    # ── Combined export 
    st.markdown("### 📥 Combined Export")
    comb_col1, comb_col2 = st.columns(2)
    with comb_col1:
        combined_csv = pd.concat(list(forecast_data.values()), ignore_index=True).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Combined CSV (All Locations)", data=combined_csv,
            file_name=f"VF_All_Locations_Forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True
        )
    with comb_col2:
        st.download_button(
            label="📊 Combined Excel (Separate Sheets)", data=to_excel_combined(forecast_data),
            file_name=f"VF_All_Locations_Forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

else:
    st.info("👈 Select your farm locations in the sidebar and click **Fetch Weather Data** to view the hourly forecast!")
    st.markdown("### 🎯 Dashboard Features")
    feat_cols = st.columns(3)
    features  = [
        ("📈 Hourly Forecasts",   "All hourly data points plotted; labels shown every few hours to stay readable"),
        ("🧭 Wind Analysis",      "Wind rose diagrams, directional patterns, and automated alerts for high winds/gusts"),
        ("📥 Flexible Export",    "Download each location as CSV/Excel or combined in one multi-sheet Excel file")
    ]
    for col, (title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"<h4 style='color:#1a5f2a;'>{title}</h4>", unsafe_allow_html=True)
            st.markdown(f"<small style='color:#666;'>{desc}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#999;margin-top:2rem;'><small>🐟 Victory Farms Ltd — Technology & Innovation</small></p>", unsafe_allow_html=True)