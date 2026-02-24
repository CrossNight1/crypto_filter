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
import numpy as np
import plotly.graph_objects as go
from scipy import stats


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
            ui.input_select("pair_interval", "Timeframe", choices=[], selected="1h"),
            ui.input_selectize("symbol_a", "Asset A", choices=[]),
            ui.input_selectize("symbol_b", "Asset B", choices=[]),
            ui.input_select("pair_mode", "Generation Mode", 
                           choices={"ratio": "Ratio", "spread": "Spread"}),
            ui.input_numeric("rolling_window", "Rolling Window", value=30, min=5),
            ui.input_numeric("window", "Window", value=700, min=5),
            ui.input_action_button("btn_gen_pair", "Generate Radar", class_="btn-primary w-100 mt-3"),
        ),
        ui.div(
            ui.card(
                ui.card_header("Statistical Analytics Dashboard"),
                output_widget("pair_main_chart"),
                full_screen=True
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Pair Diagnostics"),
                    ui.output_table("pair_metrics_viz")
                ),
                ui.card(
                    ui.card_header("Copula Fit (Dependency Mapping)"),
                    output_widget("pair_copula_chart"),
                    full_screen=True
                ),
                col_widths=[5, 7]
            )
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
                
            df_a = df_a.loc[common]
            df_b = df_b.loc[common]

            df_a = np.log(df_a)
            df_b = np.log(df_b)

            synthetic = pd.DataFrame(index=common)
            slope, intercept, r_val, p_val, std_err = stats.linregress(df_b["close"], df_a["close"])
            beta = slope
            
            if mode == "ratio":
                synthetic["open"] = df_a["open"] / (beta * df_b["open"])
                synthetic["high"] = df_a["high"] / (beta * df_b["high"])
                synthetic["low"] = df_a["low"] / (beta * df_b["low"])
                synthetic["close"] = df_a["close"] / (beta * df_b["close"])

            else: # spread
                synthetic["open"] = df_a["open"] - beta * df_b["open"]
                synthetic["high"] = df_a["high"] - beta * df_b["high"]
                synthetic["low"] = df_a["low"] - beta * df_b["low"]
                synthetic["close"] = df_a["close"] - beta * df_b["close"]
                
            # Rolling stats
            synthetic["zscore"] = calculate_rolling_zscore(synthetic["close"], window)
            synthetic["corr_pearson_price"] = df_a["close"].rolling(window).corr(df_b["close"])
            synthetic["corr_kendall_price"] = df_a["close"].rolling(window).apply(lambda x: x.corr(df_b["close"].loc[x.index], method='kendall') if len(x) == window else np.nan)

            synthetic["corr_pearson"] = df_a["close"].diff().dropna().rolling(window).corr(df_b["close"].diff().dropna())
            synthetic["corr_kendall"] = df_a["close"].diff().dropna().rolling(window).apply(lambda x: x.corr(df_b["close"].diff().dropna().loc[x.index], method='kendall') if len(x) == window else np.nan)
            
            # Store raw prices for Copula (per user request: "just use normal price")
            synthetic["price_a"] = df_a["close"]
            synthetic["price_b"] = df_b["close"]

            p.set(3, message="Computing pair metrics...")

            y = df_a["close"].values
            x = df_b["close"].values

            residuals = y - (slope * x + intercept)

            adf_stat, adf_p, _, _, _, _ = adfuller(residuals)

            spread = residuals[-window:]
            spread_vol = np.std(spread)

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

            z = synthetic["zscore"].dropna().values
            signal_rate = np.sum(np.abs(z) > 2) / len(z) if len(z) > 0 else 0

            r2 = r_val**2

            met = {
                "HedgeRatio": slope,
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
        df = df.tail(input.window())

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=[0.5, 0.15, 0.15, 0.15]
        )

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

        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_pearson']),
            name="Pearson (Return)",
            line=dict(color="#FFD700", width=1.5)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_kendall']),
            name="Kendall (Return)",
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

        order = ["HedgeRatio","ADF_P","R2","HalfLife","SpreadVol","BetaStability","SignalRate"]

        df = pd.DataFrame({
            "Metric": order,
            "Value": [met.get(k, None) for k in order]
        })

        df["Value"] = df["Value"].round(6)

        # Convert DataFrame to HTML table
        html_table = df.to_html(index=False, classes="table table-dark table-striped")

        return ui.HTML(html_table)

    # @render_widget
    # def pair_copula_chart():
    #     df = pair_data.get()
    #     if df.empty or "price_a" not in df.columns or "price_b" not in df.columns:
    #         return None

    #     # Convert back from log10 to original prices
    #     df = np.exp(df)
    #     df = df.tail(input.window())
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
        if df.empty or "price_a" not in df.columns or "price_b" not in df.columns:
            return None

        df = np.exp(df)
        df = df.tail(input.window())

        x = df["price_a"].values
        y = df["price_b"].values

        rank_x = rankdata(x) / len(x)
        rank_y = rankdata(y) / len(y)

        fig = go.Figure()

        # All points
        fig.add_trace(go.Scatter(
            x=rank_x,
            y=rank_y,
            mode='markers',
            marker=dict(size=7, color='cyan', opacity=0.6),
            text=[f"A: {pa:,.2f}<br>B: {pb:,.2f}" for pa, pb in zip(x, y)],
            hovertemplate=(
                "Rank A: %{x:.2f}<br>"
                "Rank B: %{y:.2f}<br>"
                "%{text}<extra></extra>"
            ),
            name="Rank Scatter"
        ))

        # Path of last n steps
        if len(rank_x) >= 10:
            fig.add_trace(go.Scatter(
                x=rank_x[-10:],
                y=rank_y[-10:],
                mode='lines+markers',
                line=dict(color='yellow', width=1),
                marker=dict(size=4, color='yellow'),
                hoverinfo='skip'
            ))

        # Current point
        fig.add_trace(go.Scatter(
            x=[rank_x[-1]],
            y=[rank_y[-1]],
            mode='markers',
            marker=dict(size=11, color='red'),
            text=[f"Current A: {x[-1]:,.2f}<br>Current B: {y[-1]:,.2f}"],
            hovertemplate="%{text}<extra></extra>",
            name="Current"
        ))

        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            xaxis_title=f"Rank P({input.symbol_a()})",
            yaxis_title=f"Rank P({input.symbol_b()})",
            height=600,
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[-0.05, 1.05], zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[-0.05, 1.05], zeroline=False)
        )

        return fig