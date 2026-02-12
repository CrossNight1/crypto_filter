from shiny import ui, render, reactive
import faicons as fa
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import METRIC_LABELS, BENCHMARK_SYMBOL, BINANCE_URL, ALL_METRICS, AVAILABLE_INTERVALS
from src.logger import logger

def market_radar_ui():
    return ui.navset_card_underline(
        ui.nav_panel("Market Radar",
            ui.layout_sidebar(
                ui.sidebar(

                    ui.h5("Market Filters"),

                    ui.input_action_button(
                        "btn_calc_snapshot",
                        "Load Data",
                        class_="btn-primary w-100 mb-2"
                    ),

                    ui.input_select(
                        "radar_interval",
                        "Interval",
                        choices=AVAILABLE_INTERVALS,
                        selected="1d"
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- AXES (each own row) -----
                    ui.input_select(
                        "x_axis",
                        "X Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected=ALL_METRICS[0]
                    ),

                    ui.input_select(
                        "y_axis",
                        "Y Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected=ALL_METRICS[2]
                    ),

                    ui.input_select(
                        "z_axis",
                        "Z Axis",
                        choices={"None": "None", **{m: METRIC_LABELS.get(m, m) for m in ALL_METRICS}},
                        selected=ALL_METRICS[1]
                    ),

                    # ----- LOG SCALE (single row) -----
                    ui.layout_columns(
                        ui.input_checkbox("x_log", "Log X"),
                        ui.input_checkbox("y_log", "Log Y"),
                        # ui.input_checkbox("z_log", "Log Z"),
                        col_widths=[5, 5]
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    ui.input_switch(
                        "drop_zeros",
                        "Exclude Zero Metrics",
                        value=True
                    )
                ),
                ui.div(
                    # Hidden markers to ensure reactive outputs are transmitted
                    ui.div(ui.output_text("snapshot_ready"), class_="d-none"),
                    
                    ui.panel_conditional(
                        "input.btn_calc_snapshot == 0",
                        ui.div(ui.h4("Click 'Load Data' to see data"), class_="text-center mt-5")
                    ),
                    ui.panel_conditional(
                        "input.btn_calc_snapshot > 0",
                        ui.div(
                            ui.card(
                                ui.card_header("Market Radar Snapshot"),
                                output_widget("snapshot_chart"),
                                full_screen=True
                            ),
                            ui.card(
                                ui.card_header("Data Table"),
                                ui.output_data_frame("snapshot_table")
                            )
                        )
                    )
                )
            )
        ),
        ui.nav_panel("Path Analysis",
            ui.layout_sidebar(
                ui.sidebar(

                    ui.h5("Path Settings"),

                    ui.input_action_button(
                        "btn_gen_rpg",
                        "Generate Path",
                        class_="btn-primary w-100 mb-2"
                    ),

                    ui.input_select(
                        "rpg_interval",
                        "Interval",
                        choices=AVAILABLE_INTERVALS,
                        selected="1d"
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- AXES (each own row) -----
                    ui.input_select(
                        "rpg_x",
                        "X Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected=ALL_METRICS[2]
                    ),

                    ui.input_select(
                        "rpg_y",
                        "Y Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected=ALL_METRICS[3]
                    ),

                    # ----- LOG SCALE (single row) -----
                    ui.layout_columns(
                        ui.input_checkbox("rpg_x_log", "Log X"),
                        ui.input_checkbox("rpg_y_log", "Log Y"),
                        col_widths=[5, 5]
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- NUMERIC SETTINGS -----
                    ui.layout_columns(
                        ui.input_numeric(
                            "step_size",
                            "Stride",
                            value=20,
                            min=1,
                            max=50
                        ),
                        ui.input_numeric(
                            "max_points",
                            "Length",
                            value=3,
                            min=1,
                            max=200
                        ),
                        col_widths=[5, 5]
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    ui.input_selectize(
                        "rpg_symbols",
                        "Compare Symbols",
                        choices=[],
                        multiple=True
                    )
                ),

                ui.div(
                    # Hidden marker
                    ui.div(ui.output_text("rpg_ready"), class_="d-none"),
                    
                    ui.panel_conditional(
                        "input.btn_gen_rpg == 0",
                        ui.div(ui.h4("Click 'Generate Path' to see movement"), class_="text-center mt-5")
                    ),
                    ui.panel_conditional(
                        "input.btn_gen_rpg > 0",
                        ui.div(
                            ui.card(
                                ui.card_header("Path Analysis"),
                                output_widget("rpg_chart"),
                                full_screen=True
                            ),
                            ui.card(
                                ui.card_header("Path Data"),
                                ui.output_data_frame("rpg_table")
                            )
                        )
                    )
                )
            )
        )
    )

def market_radar_server(input, output, session, global_interval):
    manager = DataManager()
    engine = MetricsEngine()
    
    snapshot_data = reactive.Value(pd.DataFrame())
    rpg_data = reactive.Value(pd.DataFrame())

    logger.log("Market Radar", "INFO", "Server initialized")

    @reactive.effect
    @reactive.event(input.btn_calc_snapshot)
    async def _():
        logger.log("Market Radar", "INFO", "Snapshot calculation triggered")
        try:
            interval = input.radar_interval()
            logger.log("Market Radar", "INFO", f"Using interval: {interval}")
            
            inventory = manager.get_inventory()
            symbols = [s for s, inters in inventory.items() if interval in inters]
            logger.log("Market Radar", "INFO", f"Found {len(symbols)} symbols in inventory")
            
            if not symbols:
                ui.notification_show("No data found for this interval. Use Data Loader first.", type="warning")
                return

            with ui.Progress(min=0, max=len(symbols)) as p:
                p.set(message="Analyzing Market Data...")
                
                # 1. Prepare Benchmark once
                benchmark_df = manager.load_data(BENCHMARK_SYMBOL, interval)
                benchmark_returns = None
                if benchmark_df is not None and not benchmark_df.empty:
                    b_close = pd.to_numeric(benchmark_df['close'], errors='coerce').ffill().fillna(0)
                    benchmark_returns = b_close.pct_change().dropna()
                
                results = []
                for i, sym in enumerate(symbols):
                    df = manager.load_data(sym, interval)
                    if df is not None and not df.empty:
                        # Compute metrics for this symbol alone
                        try:
                            # Use engine to compute for single symbol
                            # We wrap it in a dict to reuse existing compute_all_metrics logic for now,
                            # or better, compute directly for this symbol.
                            single_res = engine.compute_all_metrics(
                                {sym: df}, 
                                interval=interval, 
                                benchmark_symbol=BENCHMARK_SYMBOL,
                                benchmark_returns=benchmark_returns
                            )
                            if not single_res.empty:
                                results.append(single_res.iloc[0])
                        except Exception as e:
                            logger.log("Market Radar", "ERROR", f"Error computing {sym}: {e}")
                    
                    p.set(i + 1, detail=f"Processed {sym}")
                    await reactive.flush()
                
                if not results:
                    ui.notification_show("Failed to compute metrics for any symbols.", type="error")
                    return

                res = pd.DataFrame(results)
                logger.log("Market Radar", "INFO", f"Metrics computation complete. Symbols: {len(res)}")
                
                snapshot_data.set(res)
                ui.notification_show("Market Snapshot updated!", type="success")
                
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"Snapshot error: {str(e)}")
            ui.notification_show(f"Calculation error: {str(e)}", type="error")

    @render.text
    def snapshot_ready():
        return "true" if not snapshot_data.get().empty else "false"

    @render.text
    def rpg_ready():
        return "true" if not rpg_data.get().empty else "false"
    
    @render.data_frame
    def snapshot_table():
        return snapshot_data.get()

    @render_widget
    def snapshot_chart():
        df = snapshot_data.get()
        if df.empty: return go.Figure()

        # Work on a copy to avoid side effects
        plot_df = df.copy()

        if input.drop_zeros():
            plot_df = plot_df[plot_df['volatility'] > 1e-9]

        if plot_df.empty:
            # Return empty figure with message if possible, or just empty
            fig = go.Figure()
            fig.add_annotation(text="No data matching filters", showarrow=False, font=dict(size=20))
            fig.update_layout(template="plotly_dark")
            return fig

        x, y, z = input.x_axis(), input.y_axis(), input.z_axis()
        
        # Ensure selected columns are numeric and handle NaNs for Plotly
        for col in [x, y]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0)
        
        if z != "None" and z in plot_df.columns:
            plot_df[z] = pd.to_numeric(plot_df[z], errors='coerce').fillna(0)
            # px.scatter size must be positive
            plot_df['z_marker_size'] = plot_df[z].abs() + 1e-9
            
            fig = px.scatter(
                plot_df, x=x, y=y, size='z_marker_size', color=z,
                hover_name='symbol', log_x=input.x_log(), log_y=input.y_log(),
                color_continuous_scale='Spectral_r',
                template="plotly_dark"
            )
        else:
            fig = px.scatter(
                plot_df, x=x, y=y, hover_name='symbol',
                log_x=input.x_log(), log_y=input.y_log(),
                template="plotly_dark"
            )
            
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title=METRIC_LABELS.get(x, x),
            yaxis_title=METRIC_LABELS.get(y, y),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)")
        )

        fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
        fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
        return fig

    @reactive.effect
    def _():
        interval = input.rpg_interval()
        inventory = manager.get_inventory()
        available_syms = sorted([s for s, ints in inventory.items() if interval in ints])
        ui.update_selectize("rpg_symbols", choices=available_syms, selected=available_syms[:5])

    @reactive.effect
    @reactive.event(input.btn_gen_rpg)
    async def _():
        logger.log("Market Radar", "INFO", "RPG calculation triggered")
        try:
            interval = input.rpg_interval()
            compare_symbols = list(input.rpg_symbols())
            if not compare_symbols:
                ui.notification_show("Select symbols to compare", type="error")
                return
            
            logger.log("Market Radar", "INFO", f"Comparing {len(compare_symbols)} symbols: {compare_symbols}")
            
            step_size = input.step_size()
            max_points = input.max_points()
            x_metric = input.rpg_x()
            y_metric = input.rpg_y()
            
            with ui.Progress(min=0, max=len(compare_symbols)) as p:
                p.set(message="Calculating trajectories...")
                combined_df = pd.DataFrame()
                window_size = step_size * max_points + 1
                
                for i, sym in enumerate(compare_symbols):
                    # DataManager.load_data now uses LRU cache
                    df = manager.load_data(sym, interval)
                    if df is not None and not df.empty:
                        key_x = get_metric_key(x_metric)
                        key_y = get_metric_key(y_metric)
                        
                        sx = engine.calculate_rolling_metric(df, key_x, window=window_size, step=step_size, interval=interval)
                        sy = engine.calculate_rolling_metric(df, key_y, window=window_size, step=step_size, interval=interval)
                        
                        if not sx.dropna().empty and not sy.dropna().empty:
                            common_idx = sx.index.intersection(sy.index)
                            sx, sy = sx.loc[common_idx], sy.loc[common_idx]
                            
                            if len(sx) > max_points:
                                sx, sy = sx.iloc[-max_points:], sy.iloc[-max_points:]
                                
                            # Build trajectory rows
                            order = np.linspace(0.2, 1.0, len(sx))
                            row_data = {
                                'X_Value': sx.values,
                                'Y_Value': sy.values,
                                'Symbol': sym,
                                'Order': order,
                                'Marker_Size': (order * 15 + 5)
                            }
                            combined_df = pd.concat([combined_df, pd.DataFrame(row_data)])
                    p.set(i + 1, detail=f"Trajectory for {sym}")
                    await reactive.flush()
                
                logger.log("Market Radar", "INFO", f"Trajectory gen complete. Total rows: {len(combined_df)}")
                rpg_data.set(combined_df)
                ui.notification_show("Trajectory update complete!", type="success")
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"RPG error: {str(e)}")
            ui.notification_show(f"RPG calculation failed: {str(e)}", type="error")

    @render.data_frame
    def rpg_table():
        df = rpg_data.get()
        if df.empty: return None
        return df

    @render_widget
    def rpg_chart():
        df = rpg_data.get()
        if df.empty: return go.Figure()
        
        fig = px.line(
            df, x='X_Value', y='Y_Value', color='Symbol',
            log_x=input.rpg_x_log(), log_y=input.rpg_y_log(),
            template="plotly_dark", line_shape='spline'
        )
        # Add markers for status
        for sym in df['Symbol'].unique():
            sym_df = df[df['Symbol'] == sym]
            fig.add_trace(go.Scatter(
                x=sym_df['X_Value'], y=sym_df['Y_Value'],
                mode='markers',
                marker=dict(size=sym_df['Marker_Size'], opacity=sym_df['Order']),
                name=sym, showlegend=False
            ))
        
        # Hide legend and set proper axis titles
        x_label = METRIC_LABELS.get(input.rpg_x(), input.rpg_x())
        y_label = METRIC_LABELS.get(input.rpg_y(), input.rpg_y())
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title=x_label,
            yaxis_title=y_label,
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)")
        )
        return fig

def get_metric_key(m):
    return 'return' if m == 'metric_return' else m
