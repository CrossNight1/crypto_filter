"""
Market Radar - Visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import METRIC_LABELS, BENCHMARK_SYMBOL, BINANCE_URL, ALL_METRICS
import streamlit.components.v1 as components

# Helper for Auto-Opening URLs
def open_url_js(url):
    js = f"""
    <script>
        window.open("{url}", "_blank").focus();
    </script>
    """
    components.html(js, height=0)

def open_url(symbol):
    url = f"{BINANCE_URL}{symbol}_perpetual"
    link_text = f"Trade {symbol} on Binance Futures"
    
    open_url_js(url)
    return url, link_text

def format_metric(key):
    return METRIC_LABELS.get(key, key)

def get_metric_key(m):
    return 'return' if m == 'metric_return' else m

def clear_global_selection():
    if 'global_selection' in st.session_state:
        del st.session_state['global_selection']
    if 'last_opened_url' in st.session_state:
        st.session_state['last_opened_url'] = None

# Cached Data Helpers
@st.cache_data(show_spinner=False)
def get_cached_data(symbol, interval):
    return DataManager().load_data(symbol, interval)

@st.cache_data(show_spinner=False)
def get_cached_trajectories(compare_symbols, interval, x_metric, y_metric, z_metric, step_size, max_points, z_log=False, benchmark_symbol=BENCHMARK_SYMBOL):
    manager = DataManager()
    engine = MetricsEngine()
    window_size = step_size * max_points + 1
    combined_df = pd.DataFrame()
    
    # Pre-calculate benchmark returns if needed
    benchmark_returns = None
    if any('rel_strength_z' in m for m in [x_metric, y_metric, z_metric]):
        b_df = manager.load_data(benchmark_symbol, interval)
        if b_df is not None and not b_df.empty:
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()

    for sym in compare_symbols:
        df = manager.load_data(sym, interval)
        if df is not None and not df.empty:
            dates = df['open_time']
            
            key_x = get_metric_key(x_metric)
            key_y = get_metric_key(y_metric)
            key_z = get_metric_key(z_metric)
            
            sx = engine.calculate_rolling_metric(df, key_x, window=window_size, step=step_size, benchmark_returns=benchmark_returns, interval=interval)
            sy = engine.calculate_rolling_metric(df, key_y, window=window_size, step=step_size, benchmark_returns=benchmark_returns, interval=interval)
            sz = None
            if key_z != 'None':
                sz = engine.calculate_rolling_metric(df, key_z, window=window_size, step=step_size, benchmark_returns=benchmark_returns, interval=interval)
            
            if not sx.dropna().empty and not sy.dropna().empty:
                common_idx = sx.index.intersection(sy.index).intersection(dates.index)
                sx = sx.loc[common_idx]
                sy = sy.loc[common_idx]
                
                if len(sx) > max_points:
                    sx = sx.iloc[-max_points:]
                    sy = sy.iloc[-max_points:]
                    if sz is not None: sz = sz.iloc[-max_points:]
                
                aligned_dates = dates.iloc[sx.index]
                order = np.linspace(0.2, 1.0, len(sx))
                
                row_data = {
                    'Date': aligned_dates,
                    'X_Value': sx.values,
                    'Y_Value': sy.values,
                    'Symbol': sym,
                    'Order': order
                }
                if sz is not None:
                    row_data['Z_Value'] = sz.values
                    z_vals = sz.fillna(0).abs().values
                    if z_log:
                        z_vals = np.log1p(z_vals)
                    
                    z_min, z_max = z_vals.min(), z_vals.max()
                    z_range = z_max - z_min
                    z_norm = (z_vals - z_min) / z_range if z_range > 1e-9 else np.zeros_like(z_vals)
                    row_data['Z_Size'] = z_norm * 15 + 5
                else:
                    row_data['Z_Value'] = 0
                    row_data['Z_Size'] = order * 15 + 5
                    
                temp_df = pd.DataFrame(row_data)
                combined_df = pd.concat([combined_df, temp_df])
    return combined_df

st.set_page_config(page_title="Market Radar", layout="wide")

st.markdown("## Market Radar")

# Instantiate manager (stateless, so no need for session_state)
manager = DataManager()
metrics_engine = MetricsEngine()

# 1. Load Data
inventory = manager.get_inventory()
if not inventory:
    st.warning("No data found. Please go to 'Data Loader' to fetch data.")
    st.stop()

# Filter by timeframes available
available_intervals = set()
symbol_map = {} # interval -> list of symbols

for symbol, intervals in inventory.items():
    for interval in intervals:
        available_intervals.add(interval)
        if interval not in symbol_map:
            symbol_map[interval] = []
        symbol_map[interval].append(symbol)

# Common Settings
st.sidebar.header("Global Settings")

selected_interval = st.sidebar.selectbox("Select Timeframe", sorted(list(available_intervals)), on_change=clear_global_selection)
all_available_symbols = sorted(symbol_map.get(selected_interval, []))

# Hint for missing timeframes
if len(available_intervals) < 14:
    st.sidebar.info("Some timeframes are missing? Fetch more data in the Data Loader page.")
st.sidebar.subheader("Ticker Filters")
exclude_symbols = st.sidebar.multiselect("Exclude Specific Tickers", all_available_symbols, on_change=clear_global_selection)
drop_zeros = st.sidebar.toggle("Auto-Exclude Zero Metrics", value=True, help="Hide symbols with 0 volatility or invalid ADF stats (flat stickers)", on_change=clear_global_selection)

# Base filtered list (manual exclude)
selected_symbols = [s for s in all_available_symbols if s not in exclude_symbols]

if not selected_symbols:
    st.error(f"No symbols left after exclusion filters for {selected_interval}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["Snapshot Radar", "Rolling Analysis"])

# --- TAB 1: SNAPSHOT ---
with tab1:
    st.markdown("### Market Snapshot")
    
    if st.button("Load & Calculate Snapshot"):
        clear_global_selection()
        with st.spinner("Loading data and calculating metrics..."):
            # Load all data into memory
            prices_data = {}
            for sym in selected_symbols:
                df = get_cached_data(sym, selected_interval)
                if df is not None and not df.empty:
                    prices_data[sym] = df
            
            # Calculate
            metrics_df = metrics_engine.compute_all_metrics(prices_data, interval=selected_interval)
            st.session_state.metrics_result = metrics_df

    # Display Visualization if results exist
    if 'metrics_result' in st.session_state:
        df = st.session_state.metrics_result
        
        # Validation columns
        numeric_cols = ALL_METRICS
        missing_cols = [c for c in numeric_cols if c not in df.columns]
        
        if missing_cols:
            st.warning(f"Metrics data is outdated (missing {missing_cols}). Please click 'Load & Calculate' to update.")
        else:
            # APPLY FILTERS
            # 1. Zero Metrics Filter
            if drop_zeros:
                df = df[df['volatility'] > 1e-9].copy()
            
            # 2. Axis & Filter Selection
            filtered_df = df.copy()
            
            def get_range_params(col_data, col_name):
                # Replace inf and -inf with NaN, then drop NaNs for min/max calculation
                clean_series = col_data[col_name].replace([np.inf, -np.inf], np.nan).dropna()
                if clean_series.empty:
                    return 0.0, 1.0, (0.0, 1.0)
                
                vmin, vmax = float(clean_series.min()), float(clean_series.max())
                
                # Final safeguard for vmin/vmax themselves being inf (should be caught by clean_series, but just in case)
                if not np.isfinite(vmin): vmin = 0.0
                if not np.isfinite(vmax): vmax = 1.0
                
                if abs(vmin - vmax) < 1e-9:
                    vmin -= 0.1
                    vmax += 0.1
                return vmin, vmax, (vmin, vmax)

            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("X Axis", numeric_cols, index=0, format_func=format_metric)
                x_log = st.checkbox("Log X", key="x_log_check")
                x_min, x_max, x_default = get_range_params(df, x_axis)
                x_range = st.slider(f"Range: {format_metric(x_axis)}", x_min, x_max, x_default, key=f"f_x_{x_axis}")
                # Safe comparison: automatically handles NaNs/Infs by excluding them
                filtered_df = filtered_df[
                    (filtered_df[x_axis] >= x_range[0]) & 
                    (filtered_df[x_axis] <= x_range[1]) &
                    (np.isfinite(filtered_df[x_axis]))
                ]

            with col2:
                y_axis = st.selectbox("Y Axis", numeric_cols, index=2, format_func=format_metric)
                y_log = st.checkbox("Log Y", key="y_log_check")
                y_min, y_max, y_default = get_range_params(df, y_axis)
                y_range = st.slider(f"Range: {format_metric(y_axis)}", y_min, y_max, y_default, key=f"f_y_{y_axis}")
                filtered_df = filtered_df[
                    (filtered_df[y_axis] >= y_range[0]) & 
                    (filtered_df[y_axis] <= y_range[1]) &
                    (np.isfinite(filtered_df[y_axis]))
                ]

            with col3:
                z_axis = st.selectbox("Size/Color (Z)", ['None'] + numeric_cols, index=1, format_func=format_metric)
                z_log = st.checkbox("Log Z", key="z_log_check")
                if z_axis != 'None':
                    z_min, z_max, z_default = get_range_params(df, z_axis)
                    z_range = st.slider(f"Range: {format_metric(z_axis)}", z_min, z_max, z_default, key=f"f_z_{z_axis}")
                    filtered_df = filtered_df[
                        (filtered_df[z_axis] >= z_range[0]) & 
                        (filtered_df[z_axis] <= z_range[1]) &
                        (np.isfinite(filtered_df[z_axis]))
                    ]

            df = filtered_df
            st.success(f"Displaying {len(df)} symbols")

            # Reset selection if axes change
            if 'prev_snapshot_params' not in st.session_state:
                st.session_state.prev_snapshot_params = (x_axis, y_axis, z_axis, x_log, y_log, z_log)
            if st.session_state.prev_snapshot_params != (x_axis, y_axis, z_axis, x_log, y_log, z_log):
                clear_global_selection()
                st.session_state.prev_snapshot_params = (x_axis, y_axis, z_axis, x_log, y_log, z_log)
                
            # Chart
            if z_axis != 'None':
                z_data = df[z_axis].abs().fillna(0)
                if z_log:
                    # Use log1p to handle zeros gracefully and keep small values
                    z_data = np.log1p(z_data)
                
                df['size_col'] = z_data
                df['size_col'] = df['size_col'].replace(0, df['size_col'].mean() * 0.1 if df['size_col'].mean() != 0 else 0.1)
                
                # If z_log is on, the color mapping also uses log-transformed data
                color_col = 'color_log' if z_log else z_axis
                if z_log:
                    # Create a temporary column for color scale to avoid modifying the display value in hover
                    df['color_log'] = np.log10(df[z_axis].clip(lower=1e-9))
                
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, size='size_col', color=color_col,
                    hover_name='symbol', log_x=x_log, log_y=y_log,
                    color_continuous_scale='Viridis',
                    title=f"{format_metric(x_axis)} vs {format_metric(y_axis)} (colored/sized by {format_metric(z_axis)}{' [LOG]' if z_log else ''})",
                    hover_data={z_axis: True, 'size_col': False, color_col: False} if z_log else {z_axis: True, 'size_col': False},
                    labels={x_axis: format_metric(x_axis), y_axis: format_metric(y_axis), z_axis: format_metric(z_axis), color_col: f"{format_metric(z_axis)}"}
                )
            else:
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, hover_name='symbol',
                    log_x=x_log, log_y=y_log, title=f"{format_metric(x_axis)} vs {format_metric(y_axis)}",
                    labels={x_axis: format_metric(x_axis), y_axis: format_metric(y_axis)}
                )
                
            # Render Chart with Selection Event
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="radar_snapshot")
            
            # Handle Selection
            if event and len(event["selection"]["points"]) > 0:
                try:
                    point = event["selection"]["points"][0]
                    point_index = point["point_index"]
                    selected_sym = df.iloc[point_index]['symbol']
                    
                    url, link_text = open_url(selected_sym)
                         
                    st.session_state['global_selection'] = (selected_sym, url, link_text)
                    
                    if st.session_state.get('last_opened_url') != url:
                        # open_url_js(url) # This is now handled by open_url(selected_sym)
                        st.session_state['last_opened_url'] = url
                        
                except Exception as e:
                    st.error(f"Could not resolve symbol index: {e}")
            else:
                clear_global_selection()

            # FEEDBACK BELOW CHART
            if 'global_selection' in st.session_state:
                s_sym, s_url, s_link = st.session_state['global_selection']
                
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.success(f"Selected: **{s_sym}**")
                    st.markdown(f"### [ {s_link} ]({s_url})")
                with c2:
                    if st.button("Clear Selection", key="clear_snapshot"):
                        clear_global_selection()
                        st.rerun()

            st.dataframe(df)

# --- TAB 2: ROLLING TRAJECTORY ---
with tab2:
    st.markdown("### RPG Path")
    
    # Inputs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        x_metric = st.selectbox("X Axis Metric", ALL_METRICS, index=2, format_func=format_metric, key="rpg_x")
        x_log = st.checkbox("Log X", value=False, key="rpg_x_log")
    with c2:
        y_metric = st.selectbox("Y Axis Metric", ALL_METRICS, index=3, format_func=format_metric, key="rpg_y")
        y_log = st.checkbox("Log Y", value=False, key="rpg_y_log")
    with c3:
        z_metric = st.selectbox("Z Axis (Size/Color)", ['None'] + ALL_METRICS, index=4, format_func=format_metric, key="rpg_z")
        z_log = st.checkbox("Log Z", value=False, key="rpg_z_log")
    with c4:
        step_size = st.number_input("Step (Stride)", min_value=1, max_value=50, value=20) 
    with c5:
        max_points = st.number_input("Path Length (Points)", min_value=1, max_value=200, value=3) 
        
    # Auto-calculate Window Size based on user formula
    # "window auto define at step * path_letgh + 1"
    window_size = step_size * max_points + 1
        
    # Symbol Selection Logic
    if 'metrics_result' in st.session_state:
        metrics_df = st.session_state.metrics_result
        if drop_zeros:
            metrics_df = metrics_df[metrics_df['volatility'] > 1e-9]
        
        numeric_y = get_metric_key(y_metric)
        if numeric_y in metrics_df.columns:
            top_5_df = metrics_df.sort_values(numeric_y, ascending=False).head(5)
            default_symbols = top_5_df['symbol'].tolist()
        else:
            default_symbols = selected_symbols[:min(5, len(selected_symbols))]

        # Get current selection from session state or default
        if "rpg_symbol_selection" in st.session_state and st.session_state["rpg_symbol_selection"]["selection"]["rows"]:
            sel_rows = st.session_state["rpg_symbol_selection"]["selection"]["rows"]
            compare_symbols = metrics_df.iloc[sel_rows]['symbol'].tolist()
        else:
            compare_symbols = default_symbols
            
        display_cols = ['symbol', 'volatility', 'sharpe', 'adf_stat', 'fip']
        available_display_cols = [c for c in display_cols if c in metrics_df.columns]
    else:
        st.warning("Please calculate metrics in the 'Snapshot Radar' tab first to enable optimized selection.")
        compare_symbols = st.multiselect("Select Symbols to Compare", selected_symbols, default=selected_symbols[:3], key="rpg_multiselect")

    # Reset selection if inputs change
    if 'prev_rpg_params' not in st.session_state:
        st.session_state.prev_rpg_params = (x_metric, y_metric, z_metric, step_size, max_points, compare_symbols)
    
    current_params = (x_metric, y_metric, z_metric, step_size, max_points, compare_symbols, x_log, y_log, z_log)
    if st.session_state.prev_rpg_params != current_params:
        clear_global_selection()
        st.session_state.prev_rpg_params = current_params

    # Help for layout: We'll render the table later
    
    gen_btn = st.button("Generate Trajectory")
    if gen_btn:
        clear_global_selection()

    if gen_btn or (compare_symbols != []):
        if not compare_symbols:
            st.error("Select at least one symbol.")
        if compare_symbols:
            with st.spinner("Calculating trajectories..."):
                combined_df = get_cached_trajectories(
                    tuple(compare_symbols), selected_interval, 
                    x_metric, y_metric, z_metric, 
                    step_size, max_points, z_log=z_log
                )
                
                if combined_df.empty:
                    st.error("No data generated.")
                else:
                    # Enhanced Plot
                    fig_traj = px.line(
                        combined_df, 
                        x='X_Value', 
                        y='Y_Value', 
                        color='Symbol',
                        title=f"path: {format_metric(x_metric)} vs {format_metric(y_metric)}" + (f" (Z: {format_metric(z_metric)})" if z_metric != 'None' else ""),
                        hover_data=['Date', 'Z_Value'] if z_metric != 'None' else ['Date'],
                        markers=False, 
                        log_x=x_log,
                        log_y=y_log,
                        labels={'X_Value': format_metric(x_metric), 'Y_Value': format_metric(y_metric), 'Z_Value': format_metric(z_metric)},
                        line_shape='spline'
                    )
                    
                    fig_traj.update_traces(line=dict(width=2, dash='dot'), opacity=0.3)
                    
                    # Let's iterate symbols to add gradient markers
                    colors = px.colors.qualitative.Plotly
                    
                    for i, sym in enumerate(compare_symbols):
                        sym_data = combined_df[combined_df['Symbol'] == sym]
                        if sym_data.empty: continue
                        
                        color = colors[i % len(colors)]
                        
                        # Add markers with gradient size/opacity
                        # Ensure no NaNs in Plotly sizes
                        sym_data = sym_data.dropna(subset=['X_Value', 'Y_Value', 'Z_Size'])
                        if sym_data.empty: continue

                        marker_dict = dict(
                            size=sym_data['Z_Size'].tolist(),
                            color=color,
                            opacity=sym_data['Order'].tolist(), 
                            line=dict(width=1, color='white')
                        )
                        
                        fig_traj.add_scatter(
                            x=sym_data['X_Value'], 
                            y=sym_data['Y_Value'],
                            mode='markers',
                            marker=marker_dict,
                            name=sym,
                            showlegend=False,
                            hoverinfo='skip'
                        )
                        
                        # Start Point (X)
                        fig_traj.add_scatter(
                            x=[sym_data.iloc[0]['X_Value']],
                            y=[sym_data.iloc[0]['Y_Value']],
                            mode='markers',
                            marker=dict(symbol='x', size=12, color=color),
                            showlegend=False,
                            name=f"{sym} Start",
                            hoverinfo='name'
                        )
                        
                        # End Point (Star/Diamond) with Label
                        last_pt = sym_data.iloc[-1]
                        fig_traj.add_scatter(
                            x=[last_pt['X_Value']],
                            y=[last_pt['Y_Value']],
                            mode='markers+text',
                            marker=dict(symbol='diamond', size=18, color=color, line=dict(width=2, color='white')),
                            text=[sym],
                            textposition="top center",
                            textfont=dict(size=14, color=color, family="Arial Black"),
                            showlegend=False,
                            name=f"{sym} End"
                        )
                    
                    # Styling
                    fig_traj.update_layout(
                        xaxis_title=format_metric(x_metric),
                        yaxis_title=format_metric(y_metric),
                        hovermode="closest",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)', 
                        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    # Render Chart with Selection
                    event_traj = st.plotly_chart(fig_traj, use_container_width=True, on_select="rerun", selection_mode="points", key="radar_traj")
                    
                    st.caption(f"Showing last {max_points} points. Opacity indicates recency (Brighter = Newer). " + (f"Size indicates {format_metric(z_metric)}." if z_metric != 'None' else "Size indicates recency."))

                    # Handle Selection
                    if event_traj and len(event_traj["selection"]["points"]) > 0:
                        try:
                            point = event_traj["selection"]["points"][0]
                            idx = point["point_index"]
                            selected_sym = point["text"]
                            
                            # if point.get("curve_number", 0) == 0:
                            #     selected_sym = combined_df.iloc[idx]['Symbol']
                            # else:
                            #     selected_sym = combined_df.iloc[idx]['Symbol']

                            if selected_sym:
                                if selected_sym in ['SPY', 'GOLD', 'GC=F']:
                                    url = f"https://finance.yahoo.com/quote/{selected_sym}"
                                    link_text = f"Show {selected_sym} on Yahoo Finance"
                                else:
                                    url = f"https://www.binance.com/en/futures/{selected_sym}"
                                    link_text = f"Trade {selected_sym} on Binance Futures"
                                
                                st.session_state['global_selection'] = (selected_sym, url, link_text)
                                
                                if st.session_state.get('last_opened_url') != url:
                                    open_url_js(url)
                                    st.session_state['last_opened_url'] = url
                                
                        except Exception as e:
                            pass
                    else:
                        clear_global_selection()

                    # FEEDBACK BELOW CHART (Rendered after handling event)
                    if 'global_selection' in st.session_state:
                        r_sym, r_url, r_link = st.session_state['global_selection']
                        
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.success(f"Selected: **{r_sym}**")
                            st.markdown(f"### [ {r_link} ]({r_url})")
                        with c2:
                            if st.button("Clear Selection", key="clear_rpg"):
                                clear_global_selection()
                                st.rerun()

                    # TABLE BELOW CHART (Restored)
                    if 'metrics_result' in st.session_state:
                         st.write("---")
                         st.subheader("Ticker Selection")
                         # Redraw selection dataframe here
                         st.dataframe(
                            metrics_df[available_display_cols],
                            on_select="rerun",
                            selection_mode="multi-row",
                            hide_index=True,
                            use_container_width=True,
                            key="rpg_symbol_selection"
                        )

