from shiny import ui, render, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from src.data import DataManager
from src.metrics import MetricsEngine
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
from scipy.stats import gaussian_kde, rankdata


def _sanitize(data):
    """Replace inf/nan with 0 to prevent Plotly JSON serialization errors."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.replace([np.inf, -np.inf], np.nan).fillna(0)
    elif isinstance(data, np.ndarray):
        d = np.where(np.isfinite(data), data, 0)
        return d
    return data

def calculate_rolling_zscore(series, window):
    rolling_mean = series.ewm(span=window).mean()
    rolling_std = series.ewm(span=window).std().replace(0, 1e-9)
    return (series - rolling_mean) / rolling_std

def pair_radar_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Pair Analysis"),
            ui.input_action_button("btn_gen_pair", "Generate Radar", class_="btn-primary w-100 mt-3"),
            ui.input_select("pair_interval", "Timeframe", choices=[], selected="1h"),
            ui.input_selectize("symbol_a", "Asset A", choices=[]),
            ui.input_selectize("symbol_b", "Asset B", choices=[]),
            ui.input_select("pair_mode", "Generation Mode", 
                           choices={"ratio": "Ratio", "spread": "Spread"}, selected="spread"),
            ui.hr(),
            ui.h5("Chart Controls"),
            ui.input_numeric("rolling_window", "Rolling Window", value=50, min=5),
            ui.input_numeric("pair_window", "Window", value=500, min=30),
        ),
        ui.navset_card_pill(
            ui.nav_panel("Dashboard",
                ui.card(
                    ui.card_header("Pair Diagnostics"),
                    ui.output_ui("pair_metrics_viz")
                ),
                ui.card(
                    ui.card_header("Statistical Analytics Dashboard"),
                    output_widget("pair_main_chart"),
                    full_screen=True
                ),
            ),
            ui.nav_panel("Copula Analysis",
                ui.row(
                    ui.column(
                        3,
                        ui.card(
                            ui.card_header("Setup"),
                            ui.input_radio_buttons(
                                "copula_mode",
                                "Copula Data Mode",
                                choices={
                                    "price": "Normal Price",
                                    "log_rets": "Log Returns"
                                },
                                selected="price",
                                inline=True
                            ),
                            ui.input_numeric(
                                "r_window",
                                "Group Return Window",
                                value=10,
                                min=1
                            ),

                            full_screen=False
                        )
                    ),
                    ui.column(
                        9,
                        ui.card(
                            ui.card_header("Copula Fit (Dependency Mapping)"),
                            output_widget("pair_copula_chart"),
                            full_screen=True
                        )
                    )
                )
            ),
            ui.nav_panel("Asset Comparison",
                ui.card(
                    ui.card_header("Price Comparison"),
                    output_widget("pair_comp_chart"),
                    full_screen=True
                )
            ),
            id="pair_tabs"
        )
    )

def pair_radar_server(input, output, session, global_interval):
    manager = DataManager()
    pair_data = reactive.Value(pd.DataFrame())
    metrics_res = reactive.Value({})

    @reactive.effect
    def populate_selectors():
        inventory = manager.get_inventory()
        if not inventory:
            return
        
        all_syms = sorted(inventory.keys())
        ui.update_selectize("symbol_a", choices=all_syms, selected="BTCUSDT")
        ui.update_selectize("symbol_b", choices=all_syms, selected="ETHUSDT")
        
        intervals = sorted(list(set(i for ivs in inventory.values() for i in ivs)))
        ui.update_select("pair_interval", choices=intervals, selected=global_interval.get())

    @reactive.Effect
    @reactive.event(input.btn_gen_pair)
    def _generate_radar():
        sym_a = input.symbol_a()
        sym_b = input.symbol_b()
        interval = input.pair_interval()
        mode = input.pair_mode()
        window = input.rolling_window()

        if not sym_a or not sym_b:
            return

        with ui.Progress(min=0, max=3) as p:
            p.set(1, message="Loading data...")
            df_a = manager.load_data(sym_a, interval)
            df_b = manager.load_data(sym_b, interval)
            
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                ui.notification_show("Data missing for symbols", type="error")
                return

            p.set(2, message="Generating synthetic series...")
            df_a = df_a.set_index("open_time")
            df_b = df_b.set_index("open_time")
            common = df_a.index.intersection(df_b.index)
            
            if len(common) < window:
                ui.notification_show("Insufficient common data", type="warning")
                return
            common = common[-(window + input.pair_window()):]
            df_a = df_a.loc[common]
            df_b = df_b.loc[common]

            # Prepare prices for regression and synthetic series
            # Only use log for price columns, avoid volume/quoteVolume (can be 0)
            price_cols = ['open', 'high', 'low', 'close']
            
            # Ensure no zeroes/negatives before log
            df_a[price_cols] = df_a[price_cols].clip(lower=1e-9)
            df_b[price_cols] = df_b[price_cols].clip(lower=1e-9)

            # Store raw prices for ratio mode before log-transform
            raw_a = df_a.copy()
            raw_b = df_b.copy()

            df_a[price_cols] = np.log(df_a[price_cols])
            df_b[price_cols] = np.log(df_b[price_cols])

            synthetic = pd.DataFrame(index=common)
            
            slope, intercept, r_val, p_val, std_err = stats.linregress(df_b["close"], df_a["close"])
            beta = slope
            
            if mode == "ratio":
                synthetic["open"] = raw_a["open"] / raw_b["open"]
                synthetic["high"] = raw_a["high"] / raw_b["high"]
                synthetic["low"] = raw_a["low"] / raw_b["low"]
                synthetic["close"] = raw_a["close"] / raw_b["close"]

            else: # spread
                synthetic["open"] = df_a["open"] - (intercept + beta * df_b["open"])
                synthetic["high"] = df_a["high"] - (intercept + beta * df_b["high"])
                synthetic["low"] = df_a["low"] - (intercept + beta * df_b["low"])
                synthetic["close"] = df_a["close"] - (intercept + beta * df_b["close"])

            synthetic["low"] = synthetic.min(axis=1)
            synthetic["high"] = synthetic.max(axis=1)
                
            # Store raw prices and returns for Comparison and Copula
            synthetic["price_a"] = df_a["close"]
            synthetic["price_b"] = df_b["close"]
            synthetic["log_ret_a"] = df_a["close"].diff()
            synthetic["log_ret_b"] = df_b["close"].diff()
            vol_Ratio = synthetic["log_ret_a"].std() / synthetic["log_ret_b"].std()

            # Cumulative Returns (indexed to 1.0)
            synthetic["cum_ret_a"] = np.exp(df_a["close"] - df_a["close"].iloc[0])
            synthetic["cum_ret_b"] = np.exp(df_b["close"] - df_b["close"].iloc[0])

            p.set(3, message="Computing pair metrics...")

            y = df_a["close"].values
            x = df_b["close"].values

            residuals = y - (slope * x + intercept)

            adf_stat, adf_p, _, _, _, _ = adfuller(residuals)

            spread = residuals[-window:]
            spread_vol = np.std(spread) * np.sqrt(MetricsEngine.get_annual_scaling(interval))

            spread_lag = spread[:-1]
            spread_ret = np.diff(spread)
            lambda_coef = np.polyfit(spread_lag, spread_ret, 1)[0]
            half_life = -np.log(2) / lambda_coef if lambda_coef < 0 else np.nan

            roll_window = 60
            rolling_beta = []
            for i in range(roll_window, len(y)):
                b, _, _, _, _ = stats.linregress(x[i-roll_window:i], y[i-roll_window:i])
                rolling_beta.append(b)

            beta_stability = np.std(rolling_beta) if rolling_beta else np.nan

            # Local zscore for SignalRate metric
            z_local = calculate_rolling_zscore(synthetic["close"], window)
            z_vals = z_local.dropna().values
            signal_rate = np.sum(np.abs(z_vals) > 2) / len(z_vals) if len(z_vals) > 0 else 0

            r2 = r_val**2

            met = {
                "HedgeRatio": slope,
                "VolRatio": vol_Ratio,
                "HalfLife": half_life,
                "SpreadVol": spread_vol,
                "BetaStability": beta_stability,
                "SignalRate": signal_rate,
                "ADF_P": adf_p,
                "R2": r2
            }

            pair_data.set(synthetic)
            metrics_res.set(met)

    @render_widget
    def pair_main_chart():
        df = pair_data.get()
        if df.empty:
            return None

        df.index = pd.to_datetime(df.index)
        df.index = df.index.strftime("%Y-%m-%d %H:%M")
        window = input.rolling_window()

        df = df.tail(input.pair_window() + window)


        # Recalculate rolling stats in real-time
        df['zscore'] = calculate_rolling_zscore(df['close'], window)
        df['ema'] = df['close'].ewm(span=window, adjust=False).mean()
        df['ema_std'] = df['close'].ewm(span=window, adjust=False).std()
        df['bb_upper'] = df['ema'] + 2 * df['ema_std']
        df['bb_lower'] = df['ema'] - 2 * df['ema_std']
        
        # Correlations (Ret & Price)
        df['corr_pearson'] = df["log_ret_a"].rolling(window).corr(df["log_ret_b"])
        df['corr_pearson_price'] = df["price_a"].rolling(window).corr(df["price_b"])

        # Kendall Tau (Corrected logic)
        def _roll_kendall(s1, s2, w):
            res = np.full(len(s1), np.nan)
            v1 = s1.values
            v2 = s2.values
            for i in range(w, len(s1)+1):
                res[i-1] = stats.kendalltau(v1[i-w:i], v2[i-w:i])[0]
            return res

        df['corr_kendall'] = _roll_kendall(df["log_ret_a"], df["log_ret_b"], window)
        df['corr_kendall_price'] = _roll_kendall(df["price_a"], df["price_b"], window)

        # Truncate for display
        df = df.tail(input.pair_window())

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.15]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Pair Price",
            increasing_line_color="lightgray",
            decreasing_line_color="#ff4b4b"
        ), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema'],
            name="Basis",
            line=dict(color="rgba(255,0,0,0.8)", width=2),
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_upper'],
            name="BB Upper",
            line=dict(color="rgba(0,255,255,0.5)", width=1),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_lower'],
            name="BB Lower",
            line=dict(color="rgba(0,255,255,0.5)", width=1),
            fill='tonexty',
            fillcolor="rgba(0,255,255,0.1)"
        ), row=1, col=1)

        # Z-score
        z = _sanitize(df['zscore'])
        fig.add_trace(go.Scatter(
            x=df.index,
            y=z,
            name="Z-score",
            line=dict(color="#00ffff", width=1.5),
            fill='tozeroy',
            fillcolor="rgba(0,255,255,0.1)"
        ), row=2, col=1)

        fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=1)

        # Correlation plots
        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_pearson']),
            name="Pearson (Ret)",
            line=dict(color="#FFD700", width=1.5)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_kendall']),
            name="Kendall (Ret)",
            line=dict(color="#00FA9A", width=1.5)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_pearson_price']),
            name="Pearson (Price)",
            line=dict(color="#FFA500", width=1.5)
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_kendall_price']),
            name="Kendall (Price)",
            line=dict(color="#FF69B4", width=1.5)
        ), row=4, col=1)

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5
            ),
            template="plotly_dark",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            height=800,
            width=1500,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_xaxes(rangeslider_visible=False)

        for i in [1, 2, 3, 4]:
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=False, row=i, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=True,
                            zerolinecolor="rgba(255,255,255,0.2)", row=i, col=1)

        return fig

    @render.ui
    def pair_metrics_viz():
        met = metrics_res.get()
        if not met:
            return None

        # Define all metrics
        metrics = [
            "HedgeRatio",
            "VolRatio",
            "ADF_P",
            "R2",
            "HalfLife",
            "SpreadVol",
            "BetaStability",
            # "SignalRate"
        ]

        boxes = []
        for m in metrics:
            v = met.get(m)
            if v is not None:
                boxes.append(
                    ui.value_box(
                        title=m,
                        value=f"{v:.2f}",
                        icon="bar-chart",
                        color="cyan",
                        width=2
                    )
                )

        # Put all boxes in a single row using layout_columns for better scaling
        return ui.layout_columns(*boxes, fill=False)

                        
    # @render_widget
    # def pair_copula_chart():
    #     df = pair_data.get()
    #     if df.empty or "price_a" not in df.columns or "price_b" not in df.columns:
    #         return None

    #     # Convert back from log10 to original prices
    #     df = np.exp(df)
    #     df = df.tail(input.pair_window())
    #     x = df["price_a"].values
    #     y = df["price_b"].values

    #     # Joint KDE
    #     values = np.vstack([x, y])
    #     kde = gaussian_kde(values)

    #     # Conditional P(A|B) and P(B|A)
    #     px_given_y = []
    #     py_given_x = []
    #     for xi, yi in zip(x, y):
    #         pxy = kde([xi, yi])[0]
    #         py = np.mean([kde([xj, yi])[0] for xj in x])
    #         px = np.mean([kde([xi, yj])[0] for yj in y])
    #         px_given_y.append(pxy / py if py > 0 else 0)
    #         py_given_x.append(pxy / px if px > 0 else 0)

    #     px_prob = (np.array(px_given_y) - np.min(px_given_y)) / (np.max(px_given_y) - np.min(px_given_y) + 1e-9)
    #     py_prob = (np.array(py_given_x) - np.min(py_given_x)) / (np.max(py_given_x) - np.min(py_given_x) + 1e-9)

    #     # Rank-based probabilities for axes
    #     rank_x = rankdata(x) / len(x)
    #     rank_y = rankdata(y) / len(y)

    #     fig = go.Figure()

    #     # Scatter: all points
    #     fig.add_trace(go.Scatter(
    #         x=rank_x,
    #         y=rank_y,
    #         mode='markers',
    #         marker=dict(
    #             size=7,
    #             color='cyan',
    #             opacity=0.7,
    #             line=dict(width=1, color='rgba(255,255,255,0.3)')
    #         ),
    #         text=[f"A: ${pa:,.2f}<br>B: ${pb:,.2f}" for pa, pb in zip(x, y)],
    #         hovertemplate=(
    #             "<b>Conditional Copula</b><br>"
    #             "Rank A: %{x:.2f}<br>"
    #             "Rank B: %{y:.2f}<br>"
    #             "P(A | B): %{customdata[0]:.3f}<br>"
    #             "P(B | A): %{customdata[1]:.3f}<br>"
    #             "%{text}<extra></extra>"
    #         ),
    #         customdata=np.vstack([px_prob, py_prob]).T,
    #         name="All Points"
    #     ))

    #     # Highlight current/latest price
    #     fig.add_trace(go.Scatter(
    #         x=[rank_x[-1]],
    #         y=[rank_y[-1]],
    #         mode='markers',
    #         marker=dict(
    #             size=10.5,
    #             color='red',
    #             line=dict(width=1, color='white')
    #         ),
    #         text=[f"Current A: ${x[-1]:,.4f}<br>Current B: ${y[-1]:,.4f}"],
    #         hovertemplate="%{text}<extra></extra>",
    #         name="Current Price"
    #     ))

    #     fig.update_layout(
    #         template="plotly_dark",
    #         showlegend=False,
    #         paper_bgcolor="#0b3d91",
    #         plot_bgcolor="#0b3d91",
    #         font=dict(family="Space Mono", color="white"),
    #         xaxis_title=f"Rank P({input.symbol_a()})",
    #         yaxis_title=f"Rank P({input.symbol_b()})",
    #         margin=dict(l=60, r=60, t=60, b=60),
    #         height=600,
    #         xaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[-0.05, 1.05], zeroline=False),
    #         yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[-0.05, 1.05], zeroline=False)
    #     )

    #     return fig

    @render_widget
    def pair_copula_chart():
        df = pair_data.get()
        if df.empty:
            return None

        mode = input.copula_mode()
        df_plot = df.tail(input.pair_window()).copy()

        if mode == "price":
            x = np.exp(df_plot["price_a"].values)
            y = np.exp(df_plot["price_b"].values)
            title_suffix = " (Price Levels)"
        else:
            df_plot = df_plot.dropna(subset=["log_ret_a", "log_ret_b"])
            x = df_plot["log_ret_a"].rolling(input.r_window()).mean().dropna().values
            y = df_plot["log_ret_b"].rolling(input.r_window()).mean().dropna().values
            title_suffix = " (Log Returns)"

        if len(x) < 2:
            return None

        rank_x = rankdata(x) / (len(x) + 1)
        rank_y = rankdata(y) / (len(y) + 1)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rank_x,
            y=rank_y,
            mode='markers',
            marker=dict(
                size=7, 
                color='cyan', 
                opacity=0.6,
                line=dict(width=0.5, color='rgba(255,255,255,0.2)')
            ),
            text=[f"A: {v1:,.4f}<br>B: {v2:,.4f}" for v1, v2 in zip(x, y)],
            hovertemplate=(
                "Rank A: %{x:.2f}<br>"
                "Rank B: %{y:.2f}<br>"
                "%{text}<extra></extra>"
            ),
            name="Rank Scatter"
        ))

        # Path of last 10 steps
        if len(rank_x) >= 10:
            fig.add_trace(go.Scatter(
                x=rank_x[-10:],
                y=rank_y[-10:],
                mode='lines+markers',
                line=dict(color='yellow', width=1.5, dash='dot'),
                marker=dict(size=4, color='yellow'),
                name="Last 10 Ticks",
                hoverinfo='skip'
            ))

        # Current point
        fig.add_trace(go.Scatter(
            x=[rank_x[-1]],
            y=[rank_y[-1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='cross', line=dict(width=2, color='white')),
            text=[f"Current A: {x[-1]:,.4f}<br>Current B: {y[-1]:,.4f}"],
            hovertemplate="%{text}<extra></extra>",
            name="Current"
        ))

        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            xaxis_title=f"Rank {input.symbol_a()}{title_suffix}",
            yaxis_title=f"Rank {input.symbol_b()}{title_suffix}",
            height=800,
            width=1090,
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[-0.02, 1.02], zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[-0.02, 1.02], zeroline=False),
            font=dict(family="Space Mono")
        )

        return fig

    @render_widget
    def pair_comp_chart():
        df = pair_data.get()
        if df.empty or "cum_ret_a" not in df.columns or "cum_ret_b" not in df.columns:
            return None

        window = input.rolling_window()

        df_plot = df.tail(input.pair_window() + window).copy()
        df_plot.index = pd.to_datetime(df_plot.index)
        df_plot.index = df_plot.index.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate real-time correlations
        df_plot['corr_pearson'] = df_plot["log_ret_a"].rolling(window).corr(df_plot["log_ret_b"])
        df_plot['corr_pearson_price'] = df_plot["price_a"].rolling(window).corr(df_plot["price_b"])

        # Optimized Rolling Kendall for Comp Chart
        def _roll_kendall(s1, s2, w):
            res = np.full(len(s1), np.nan)
            v1 = s1.values
            v2 = s2.values
            for i in range(w, len(s1)+1):
                res[i-1] = stats.kendalltau(v1[i-w:i], v2[i-w:i])[0]
            return res

        df_plot['corr_kendall'] = _roll_kendall(df_plot["log_ret_a"], df_plot["log_ret_b"], window)
        df_plot['corr_kendall_price'] = _roll_kendall(df_plot["price_a"], df_plot["price_b"], window)

        # Truncate for display
        df_plot = df_plot.tail(input.pair_window())

        # Use pre-calculated log returns for scaling comparison
        ret_a = df_plot["log_ret_a"].dropna()
        ret_b = df_plot["log_ret_b"].dropna()

        # Calculate beta (vol ratio) for better visual scaling
        beta = ret_a.std() / ret_b.std()

        # Scale cumulative return of A to match B visually
        cum_ret_a_scaled = (ret_a / beta).cumsum()
        cum_ret_b_scaled = ret_b.cumsum()

        # # Normalize prices for visual comparison on secondary axis
        # slope, intercept, r_val, p_val, std_err = stats.linregress(df_plot["price_a"], df_plot["price_b"])
        # price_a_norm = intercept + slope * df_plot["price_a"]
        # price_b_norm = df_plot["price_b"]
    
        # Create figure with 3 rows: Performance, Corr (Return), Corr (Price)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}]]
        )

        # --- Row 1: Performance Comparison ---
        # Vol scaled
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=cum_ret_a_scaled,
            name=f"{input.symbol_a()} Vol Scaled",
            line=dict(color="#FFD60A", width=1),
            opacity=1
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=cum_ret_b_scaled,
            name=f"{input.symbol_b()} Vol Scaled",
            line=dict(color="#FF006E", width=1),
            opacity=1
        ), row=1, col=1, secondary_y=False)

        # # Cointegration scaled price
        # fig.add_trace(go.Scatter(
        #     x=df_plot.index,
        #     y=price_a_norm,
        #     name=f"{input.symbol_a()} Coin Price",
        #     line=dict(color="#F8F9FA", width=2, dash="dot"),
        #     opacity=1
        # ), row=1, col=1, secondary_y=True)

        # fig.add_trace(go.Scatter(
        #     x=df_plot.index,
        #     y=price_b_norm,
        #     name=f"{input.symbol_b()} Coin Price",
        #     line=dict(color="#FB5607", width=2, dash="dot"),
        #     opacity=1
        # ), row=1, col=1, secondary_y=True)

        # --- Row 2: Correlation (Return) ---
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=_sanitize(df_plot['corr_pearson']),
            name="Pearson (Ret)",
            line=dict(color="#FFD700", width=1)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=_sanitize(df_plot['corr_kendall']),
            name="Kendall (Ret)",
            line=dict(color="#00FA9A", width=1)
        ), row=2, col=1)

        # --- Row 3: Correlation (Price) ---
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=_sanitize(df_plot['corr_pearson_price']),
            name="Pearson (Price)",
            line=dict(color="#FFA500", width=1)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=_sanitize(df_plot['corr_kendall_price']),
            name="Kendall (Price)",
            line=dict(color="#FF69B4", width=1)
        ), row=3, col=1)

        # Layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            height=800,
            width=1500,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5
            )
        )

        fig.update_yaxes(row=1, col=1, secondary_y=False)
        fig.update_yaxes(row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(row=2, col=1)
        fig.update_yaxes(row=3, col=1)

        for i in [1, 2, 3]:
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=False, row=i, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=True, 
                            zerolinecolor="rgba(255,255,255,0.2)", row=i, col=1)
            
        return fig