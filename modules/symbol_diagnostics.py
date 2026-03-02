from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
from shinywidgets import output_widget, render_widget
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import AVAILABLE_INTERVALS, BENCHMARK_SYMBOL, METRIC_LABELS, MANDATORY_CRYPTO
from src.logger import logger
from ml_engine.labeling.labeler import Labeler
from ml_engine.analysis.correlation import DecompositionEngine
from ml_engine.data.bars import construct_volume_bars, construct_dollar_bars, calibrate_bar_threshold

# --- GARCH HELPER ---
def garch_neg_log_likelihood(params, returns):
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
    log_likelihood = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    return -log_likelihood

def fit_garch(returns):
    # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
    bounds = ((1e-6, None), (1e-6, 1), (1e-6, 1))
    # Initial guess
    initial_params = [np.var(returns)*0.01, 0.1, 0.8]
    
    res = minimize(garch_neg_log_likelihood, initial_params, args=(returns,),
                   bounds=bounds, method='L-BFGS-B')
    return res.x

def forecast_garch(params, returns, steps=10, ann_factor=1):
    omega, alpha, beta = params
    # Last sigma2
    sigma2_t = np.var(returns) # simplistic start if rebuilding full path is slow
    # Better: rebuild last variance
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    last_sigma2 = sigma2[-1]
    last_ret2 = returns[-1]**2
    
    forecasts = []
    current_sigma2 = omega + alpha * last_ret2 + beta * last_sigma2
    
    # Long run variance = omega / (1 - alpha - beta)
    var_long = omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.var(returns)

    for _ in range(steps):
        forecasts.append(np.sqrt(current_sigma2) * np.sqrt(ann_factor))
        # E[r_{t+k}^2] = sigma_{t+k}^2
        current_sigma2 = omega + alpha * current_sigma2 + beta * current_sigma2
        # Actually E[sigma^2_{t+1}] = omega + (alpha+beta)*sigma^2_t
        # So we project variance:
        # sigma^2_{t+k} = var_long + (alpha+beta)^(k-1) * (sigma^2_{t+1} - var_long)
    
    # Re-calculate correct projection loop
    fc_vars = []
    curr = omega + alpha * last_ret2 + beta * last_sigma2
    for _ in range(steps):
        fc_vars.append(curr)
        curr = omega + (alpha + beta) * curr
        
    return np.sqrt(fc_vars) * np.sqrt(ann_factor)

# --- STYLING CONSTANTS ---
THEME_BG = "#0b3d91"
THEME_FONT = "Space Mono"
THEME_GRID = "rgba(255, 255, 255, 0.3)"
THEME_ZERO = "rgba(255, 255, 255, 0.5)"
THEME_TEXT = "white"

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font=dict(family=THEME_FONT, color=THEME_TEXT),
        xaxis=dict(gridcolor=THEME_GRID, zerolinecolor=THEME_ZERO),
        yaxis=dict(gridcolor=THEME_GRID, zerolinecolor=THEME_ZERO),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def symbol_diagnostics_ui():
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button("btn_run_diag", "Run Diagnostics", class_="btn-primary w-100 mt-2"),
                ui.input_selectize("diag_symbol", "Select Symbol", choices=[], selected="BTCUSDT", multiple=False),
                ui.input_select("diag_interval", "Interval", choices=AVAILABLE_INTERVALS, selected="1h"),
                ui.input_numeric("metric_window", "Metrics Window", value=10, min=20, max=500),
                ui.input_numeric("diag_window", "Analysis Window", value=100, min=20, max=500),
            ),
            
            ui.navset_card_underline(
                ui.nav_panel(
                    "Standard Analysis",
                    ui.div(
                        # Hidden marker
                        ui.div(ui.output_text("diag_ready"), class_="d-none"),
                        
                        ui.panel_conditional(
                            "input.btn_run_diag > 0",
                            
                            # 1. Performance Overview
                            ui.card(
                                ui.card_header("Performance Overview"),
                                ui.layout_columns(
                                    ui.value_box("CVaR", ui.output_text("val_cvar"), theme="primary"),
                                    ui.value_box("Volatility", ui.output_text("val_vol"), theme="blue"),
                                    ui.value_box("Omega Ratio", ui.output_text("val_omega"), theme="red"),
                                    ui.value_box("Avg Drawdown", ui.output_text("val_avgdd"), theme="orange"),
                                    ui.value_box("Beta", ui.output_text("val_beta"), theme="info"),
                                    ui.value_box("Alpha", ui.output_text("val_alpha"), theme="success"),
                                )
                            ),
                            
                            # 2. Metrics & Exposure
                            ui.card(
                                ui.card_header("Market-Neutral Analysis"),
                                ui.layout_columns(
                                    ui.div(
                                        output_widget("plot_metrics")
                                    ),
                                    ui.div(
                                        output_widget("plot_mn_cum_ret")
                                    ),
                                    col_widths=[6, 6]
                                )
                            ),
                            
                            # 3. Forecast
                            ui.card(
                                ui.card_header("Forecast (Forward 10 periods)"),
                                ui.layout_columns(
                                    ui.div(
                                        output_widget("plot_forecast_price")
                                    ),
                                    ui.div(
                                        output_widget("plot_forecast_vol")
                                    ),
                                    col_widths=[6, 6]
                                )
                            ),
                            
                            # # 4. Cointegration
                            # ui.card(
                            #     ui.card_header("Cointegration & Correlation"),
                            #     output_widget("plot_coint")
                            # )
                        )
                    )
                ),
                ui.nav_panel(
                    "Financial Bars",
                    ui.div(
                        ui.row(
                            ui.column(
                                4,
                                ui.card(
                                    ui.card_header("Structural Engineering"),
                                    ui.input_select("diag_bar_type", "Active Bar Type", choices=["Time Bars", "Volume Bars", "Dollar Bars"], selected="Dollar Bars"),
                                    ui.layout_columns(
                                        ui.input_numeric("diag_vol_th", "Volume Threshold", value=10000, min=1),
                                        ui.input_numeric("diag_dollar_th", "Dollar Threshold", value=1000000, min=1),
                                    ),
                                    ui.layout_columns(
                                        ui.input_action_button("btn_generate_bars", "Generate All", class_="btn-primary w-100"),
                                        ui.input_action_button("btn_diag_auto_calibrate", "Auto Calibrate", class_="btn-primary w-100"),
                                    ),
                                    ui.hr(),
                                    ui.output_table("diag_engineering_stats_table")
                                )
                            ),
                            ui.column(
                                8,
                                ui.card(
                                    ui.card_header("Return Distribution Comparison"),
                                    output_widget("diag_engineering_dist_plot"),
                                    full_screen=True
                                )
                            )
                        ),
                        ui.row(
                            ui.column(
                                12,
                                ui.card(
                                    ui.card_header("Bar OHLCV Viewer"),
                                    output_widget("plot_financial_bars"),
                                    full_screen=True
                                )
                            )
                        )
                    )
                ),
                id="tab_diag"
            )
        )
    )
    
def symbol_diagnostics_server(input, output, session, global_interval):
    manager = DataManager()
    engine = MetricsEngine()
    
    diag_data = reactive.Value({})
    engineering_results_cache = reactive.Value(None)

    data_info = reactive.Value({"global": {"oldest": "-", "latest": "-"},
                                "symbol": {"oldest": "-", "latest": "-"}})
    
    def get_timestamps(symbol, interval):
        # Disable auto_sync for metadata checks to prevent startup data fetching
        df = manager.load_data(symbol, interval, auto_sync=False)
        if df is not None and not df.empty and 'open_time' in df.columns:
            ts = pd.to_datetime(df['open_time'])
            return {"oldest": str(ts.min()), "latest": str(ts.max())}
        return {"oldest": "-", "latest": "-"}
    
    @reactive.Effect
    def populate_symbols():
        inventory = manager.get_inventory()
        all_syms = sorted(inventory.keys())
        ui.update_selectize("diag_symbol", choices=all_syms, selected="BTCUSDT", server=True)
        
        # Set benchmark/global timestamps once
        global_ts = get_timestamps(BENCHMARK_SYMBOL, input.diag_interval())
        data_info.set({"global": global_ts, "symbol": {"oldest": "-", "latest": "-"}})
    
    @reactive.Effect
    @reactive.event(input.diag_symbol)
    def update_symbol_ts():
        if input.diag_symbol():
            symbol_ts = get_timestamps(input.diag_symbol(), input.diag_interval())
            current_global = data_info.get().get("global", {"oldest": "-", "latest": "-"})
            data_info.set({"global": current_global, "symbol": symbol_ts})
    
    @render.ui
    def data_status():
        d = data_info.get()
        return ui.HTML(f"""
            <div style="font-size: 0.7rem; opacity: 0.7; color: white;">
                <div>Global Data: <div>
                <div>{d['global']['oldest']}</div>
                <div>{d['global']['latest']}</div>
                <br>
                <div>Symbol Data: <div>
                <div>{d['symbol']['oldest']}</div>
                <div>{d['symbol']['latest']}</div>
            </div>
        """)

    @reactive.Effect
    @reactive.event(input.btn_run_diag)
    def _():
        symbol = input.diag_symbol()
        interval = input.diag_interval()
        window = input.diag_window()
        metric_window = input.metric_window()
        
        if not symbol:
            ui.notification_show("Please select a symbol", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            p.set(10, message="Loading Data...")
            
            # 1. Load Data
            df = manager.load_data(symbol, interval)
            bench_df = manager.load_data(BENCHMARK_SYMBOL, interval)
            
            if df is None or df.empty or len(df) < window:
                ui.notification_show("Insufficient data for analysis", type="error")
                return
            
             # Update Symbol Date Info
            if 'open_time' in df.columns:
                last_ts_sym = pd.to_datetime(df['open_time']).max()
                curr_info = data_info.get()
                curr_info['symbol'] = str(last_ts_sym)
                data_info.set(curr_info)

            # Clean & Prepare
            prices = pd.to_numeric(df['close'], errors='coerce').ffill().values
            log_rets = np.diff(np.log(prices))
            log_rets = np.nan_to_num(log_rets)
            
            # Load Benchmark
            bench_rets = None
            if bench_df is not None and not bench_df.empty:
                b_prices = pd.to_numeric(bench_df['close'], errors='coerce').ffill().values
                bench_rets = np.diff(np.log(b_prices))
                # Align lengths
                min_len = min(len(log_rets), len(bench_rets))
                log_rets_aligned = log_rets[-min_len:]
                bench_rets_aligned = bench_rets[-min_len:]
            
            p.set(30, message="Calculating Performance...")
            
            # 2. Performance Metrics
            res_sharpe = engine.calculate_sharpe_ratio(log_rets, interval=interval)
            res_sortino = engine.calculate_sortino_ratio(log_rets, interval=interval)
            res_maxdd = engine.calculate_max_drawdown(prices)
            res_avgdd = engine.calculate_avg_drawdown(prices)
            
            # CVaR (Conditional Value at Risk) at 95% confidence level
            var_threshold = np.percentile(log_rets, 5)  # 5th percentile (95% confidence)
            cvar = log_rets[log_rets <= var_threshold].mean()
            
            # Volatility (annualized)
            ann_factor = engine.get_annual_scaling(interval)
            volatility = np.std(log_rets) * np.sqrt(ann_factor)
            
            # Omega Ratio (ratio of gains to losses relative to threshold, using 0 as threshold)
            threshold = 0
            gains = log_rets[log_rets > threshold].sum()
            losses = np.abs(log_rets[log_rets <= threshold].sum())
            omega_ratio = gains / losses if losses != 0 else 0
            
            
            # 3. Metrics Snapshot
            latest_metrics = engine.calculate_all_indicators(
                df.iloc[-window * 2:], 
                benchmark_returns=np.log(bench_df['close']).diff()[-window * 2:],
                interval=interval,
                window=int(metric_window) 
            )
            
            p.set(50, message="Calculating Exposure...")
            
            # 4. Exposure (Beta)
            beta, alpha, r2 = 0, 0, 0
            if bench_rets is not None:
                beta_, alpha_, r2 = engine.calculate_beta_alpha(log_rets_aligned, bench_rets_aligned)
                
            # Factor Decomp - Load ALL symbols for comprehensive analysis
            market_data = {}
            for s in MANDATORY_CRYPTO:
                d = manager.load_data(s, interval)
                if d is not None:
                    market_data[s] = pd.to_numeric(d['close'], errors='coerce').pct_change()
            
            factor_df = pd.DataFrame(market_data).ffill().fillna(0).tail(window)
            mn_cum_ret = pd.Series()
            
            if factor_df.shape[1] > 2:
                decomp_res = DecompositionEngine.k_factor_decompose(factor_df, k=5)
                sym_series = pd.Series(log_rets[-window:], index=factor_df.index)
                
                # Market Neutral Calculation: Regress sym_series against PC1
                pc1 = decomp_res['factor_returns']['PC1'].values.reshape(-1, 1)
                y = sym_series.values
                
                lr = LinearRegression()
                lr.fit(pc1, y)
                residuals = y - lr.predict(pc1)
                mn_cum_ret = pd.Series(np.cumsum(residuals), index=factor_df.index)
            
            p.set(70, message="Forecasting...")
            
            # 5. Forecast
            # ARIMA for Price
            try:
                history_price = prices[-window:]
                model = ARIMA(history_price, order=(5,1,5))
                model_fit = model.fit()
                fc_res = model_fit.get_forecast(steps=10)
                fc_mean = fc_res.predicted_mean
                fc_ci = fc_res.conf_int(alpha=0.05)
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"ARIMA failed: {e}")
                fc_mean, fc_ci = np.zeros(10), np.zeros((10, 2))

            # GARCH Volatility Forecast
            try:
                garch_params = fit_garch(log_rets[-window:])
                vol_forecast_vals = forecast_garch(garch_params, log_rets[-window:], steps=10, ann_factor=MetricsEngine.get_annual_scaling(interval))
                
                # Historical vol for context
                hist_vol = pd.Series(log_rets[-window:]**2).ewm(alpha=0.06).mean()
                hist_vol = np.sqrt(hist_vol) * np.sqrt(MetricsEngine.get_annual_scaling(interval))
                hist_vol = hist_vol.values
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"GARCH failed: {e}")
                vol_forecast_vals = np.zeros(10)
                vol_ci = np.zeros((10, 2))
                hist_vol = np.zeros(window)

            p.set(80, message="Regime Classification...")
            
            # 6. Regime (Labeler - Trend)
            # Use smaller window for labeling loop or just label whole series
            l_algo = Labeler(amplitude_threshold=0.01, max_inactive_period=10) # 1% move, 10 bars inactive
            lbl_df = l_algo.label(prices[-window:])
            curr_lbl_val = lbl_df['label'].iloc[-1]
            if curr_lbl_val == 1: curr_regime = "Uptrend"
            elif curr_lbl_val == -1: curr_regime = "Downtrend"
            else: curr_regime = "Sideways/Neutral"
            
            p.set(90, message="Relationships...")
            
            # Relationships (Deprecated)
            corrs = pd.Series()
            coint_scores = []
            zscore_spreads = []

            col_droped = ["volatility"]
            latest_metrics = latest_metrics.drop(columns=col_droped)
            latest_metrics = latest_metrics.dropna(axis=1).dropna(axis=0)
            
            from scipy.stats import zscore
            for col in latest_metrics.columns:
                latest_metrics[col] = zscore(latest_metrics[col])

            # Pack Data
            data_pack = {
                "sharpe": res_sharpe,
                "sortino": res_sortino,
                "maxdd": res_maxdd,
                "avgdd": res_avgdd,
                "cvar": cvar,
                "volatility": volatility,
                "omega": omega_ratio,
                "metrics_df": latest_metrics.tail(1).T.reset_index(),
                "beta": beta_,
                "alpha": alpha_,
                "mn_cum_ret": mn_cum_ret if not mn_cum_ret.empty else pd.Series(),
                "fc_price": {"hist": history_price, "fc": fc_mean, "ci": fc_ci},
                "fc_vol": {"hist": hist_vol, "fc": vol_forecast_vals},
                "regime": {"status": curr_regime, "labels": lbl_df['label'].values, "prices": lbl_df['price'].values}
            }
            
            diag_data.set(data_pack)
            p.set(100, message="Complete")

    # ----- RENDERERS -----
    
    @render.text
    def diag_ready():
        return "true" if diag_data.get() else "false"

    @render.text
    def val_sharpe(): return f"{diag_data.get().get('sharpe', 0):.2f}" if diag_data.get() else "-"
    @render.text
    def val_sortino(): return f"{diag_data.get().get('sortino', 0):.2f}" if diag_data.get() else "-"
    @render.text
    def val_maxdd(): return f"{diag_data.get().get('maxdd', 0)*100:.2f}%" if diag_data.get() else "-"
    @render.text
    def val_avgdd(): return f"{diag_data.get().get('avgdd', 0)*100:.2f}%" if diag_data.get() else "-"
    @render.text
    def val_winrate(): return f"{diag_data.get().get('winrate', 0)*100:.1f}%" if diag_data.get() else "-"

    @render.text
    def val_beta():
        d = diag_data.get()
        return f"{d.get('beta', 0):.3f}" if d else "-"
    
    @render.text
    def val_alpha():
        d = diag_data.get()
        return f"{d.get('alpha', 0):.4f}" if d else "-"
    
    @render.text
    def val_cvar():
        d = diag_data.get()
        return f"{d.get('cvar', 0)*100:.2f}%" if d else "-"
    
    @render.text
    def val_vol():
        d = diag_data.get()
        return f"{d.get('volatility', 0)*100:.2f}%" if d else "-"
    
    @render.text
    def val_omega():
        d = diag_data.get()
        return f"{d.get('omega', 0):.2f}" if d else "-"

    @render_widget
    def plot_metrics():
        d = diag_data.get()
        if not d or d['metrics_df'].empty: return None
        
        df = d['metrics_df']
        if len(df.columns) < 2: return None

        df.columns = ["Metric", "Value"]
        # Filter numeric only
        df = df[pd.to_numeric(df['Value'], errors='coerce').notnull()].head(10)
        df['Value'] = pd.to_numeric(df['Value'])
        df['Metric'] = df['Metric'].map(METRIC_LABELS)
        
        # Radial bar chart (polar)
        fig = go.Figure(go.Barpolar(
            r=df['Value'],
            theta=df['Metric'],
            marker=dict(
                color=df['Value'],
                colorscale='Spectral_r',
                showscale=True,
                colorbar=dict(
                    title="Norm Value"
                )
            ),
            opacity=0.8
        ))

        
        fig.update_layout(
            polar=dict(
                bgcolor=THEME_BG,
                radialaxis=dict(visible=True, gridcolor=THEME_GRID),
                angularaxis=dict(gridcolor=THEME_GRID)
            ),
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_BG,
            font=dict(family=THEME_FONT, color=THEME_TEXT),
            showlegend=False,
            height=450,
            width=700
        )
        return apply_theme(fig)

    @render_widget
    def plot_mn_cum_ret():
        d = diag_data.get()
        if not d or d['mn_cum_ret'].empty: return None
        
        series = d['mn_cum_ret']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(series)),
            y=series.values,
            mode='lines',
            line=dict(color='cyan', width=2),
            name='MN Cum Ret'
        ))
        fig.update_layout(
            title="Market-Neutral Cumulative Return (Ex-PC1)",
            xaxis_title="Periods",
            yaxis_title="Cumulative Return",
            height=450, width=700)
        return apply_theme(fig)

    @render_widget
    def plot_forecast_price():
        d = diag_data.get()
        if not d: return go.Figure()
        
        hist = d['fc_price']['hist']
        fc = d['fc_price']['fc']
        ci = d['fc_price']['ci']
        
        x_hist = np.arange(len(hist))
        x_fc = np.arange(len(hist), len(hist) + len(fc))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_hist, y=hist, name="History", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=x_fc, y=fc, name="Forecast", line=dict(color="lime", dash="dot", width=1)))
        
        # CI
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fc, x_fc[::-1]]),
            y=np.concatenate([ci[:,1], ci[:,0][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='(95%) CI'
        ))
        
        fig.update_layout(
            template="plotly_dark", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), height=350, width=700)
        return apply_theme(fig)

    @render_widget
    def plot_forecast_vol():
        d = diag_data.get()
        if not d: return go.Figure()
        
        hist = d['fc_vol']['hist']
        fc = d['fc_vol']['fc']

        diff = hist[-1] - fc[0]
        
        x_hist = np.arange(len(hist))
        x_fc = np.arange(len(hist), len(hist) + len(fc))
        fc = fc + diff

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_hist, y=hist, name="Volatility (GARCH)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=x_fc, y=fc, name="Forecast", line=dict(color="red", dash="dot", width=1)))
        
        # Add CI if available (±1 std)
        fc_ci = np.full(len(fc), np.std(hist))

        if fc_ci is not None:
            steps = np.arange(1, len(fc) + 1)

            k = 0.2
            scale = k * np.sqrt(steps)

            upper = fc + (fc_ci - fc) * scale
            lower = fc - (fc_ci - fc) * scale

            fig.add_trace(go.Scatter(
                x=np.concatenate([x_fc, x_fc[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 99, 71, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='(95%) CI'
            ))

        fig.update_layout(
            template="plotly_dark", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), height=350, width=700)
        return apply_theme(fig)


    @reactive.calc
    @reactive.event(input.btn_generate_bars)
    def engineering_result():
        symbol = input.diag_symbol()
        interval = input.diag_interval()
        
        if not symbol:
            return None

        with ui.Progress(min=0, max=100) as p:
            p.set(10, message="Loading Data...")
            df = manager.load_data(symbol, interval)
            if df is None or df.empty: return None
            
            if 'open_time' in df.columns:
                df = df.set_index(pd.to_datetime(df['open_time']))
            
            p.set(30, message="Generating Time Bars...")
            time_df = df.copy()
            time_df['ret'] = np.log(time_df['close'] / time_df['close'].shift(1))
            
            p.set(50, message="Generating Volume Bars...")
            vol_df = construct_volume_bars(df, input.diag_vol_th())
            if not vol_df.empty:
                vol_df['ret'] = np.log(vol_df['close'] / vol_df['close'].shift(1))
                
            p.set(70, message="Generating Dollar Bars...")
            dollar_df = construct_dollar_bars(df, input.diag_dollar_th())
            if not dollar_df.empty:
                dollar_df['ret'] = np.log(dollar_df['close'] / dollar_df['close'].shift(1))
            
            p.set(100, message="Complete")
            
            res = {
                "time": time_df,
                "volume": vol_df,
                "dollar": dollar_df,
                "ticker": symbol
            }
            engineering_results_cache.set(res)
            return res

    @render_widget
    def diag_engineering_dist_plot():
        res = engineering_result()
        if res is None: return None

        datasets = []
        labels = []
        
        for k in ["time", "volume", "dollar"]:
            df = res.get(k)
            if df is not None and not df.empty and 'ret' in df.columns:
                rets = df['ret'].dropna()
                if len(rets) > 10:
                    datasets.append(rets.values)
                    labels.append(k.capitalize() + " Bars")
                    
        if not datasets: return None

        colors = pc.diverging.Spectral_r[:len(datasets)]

        fig = ff.create_distplot(
            datasets,
            labels,
            bin_size=.001,
            show_hist=True,
            show_curve=True,
            colors=colors
        )

        fig.update_layout(
            template="plotly_dark",
            height=460,
            width=1000,
            margin=dict(t=40, b=60, l=30, r=30),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            xaxis_title="Log Returns",
            yaxis_title="Density"
        )
        return apply_theme(fig)

    @render.table
    def diag_engineering_stats_table():
        res = engineering_result()
        if res is None: return None
        
        stats_list = []
        for k in ["time", "volume", "dollar"]:
            df = res.get(k)
            if df is not None and not df.empty:
                rets = df['ret'].dropna()
                stats_list.append({
                    "Bar Type": k.capitalize(),
                    "Count": len(df),
                    "Skew": skew(rets),
                    "Kurtosis": kurtosis(rets, fisher=True)
                })
        
        df_stats = pd.DataFrame(stats_list)
        return (
            df_stats.style
            .hide(axis="index")
            .format({"Skew": "{:.4f}", "Kurtosis": "{:.4f}"})
            .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("color", "black"), ("background-color", "#FCC780"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("border", "1px solid #1a4da3")]}
            ])
        )

    @reactive.Effect
    @reactive.event(input.btn_diag_auto_calibrate)
    async def _auto_calibrate_diag():
        symbol = input.diag_symbol()
        interval = input.diag_interval()
        
        if not symbol:
            ui.notification_show("Please select a symbol", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            p.set(20, message="Loading Data...")
            df = manager.load_data(symbol, interval)
            if df is None or df.empty:
                ui.notification_show("No data available", type="error")
                return
            
            if 'open_time' in df.columns:
                df = df.set_index(pd.to_datetime(df['open_time']))

            p.set(40, message="Calibrating Volume...")
            opt_vol = calibrate_bar_threshold(df, "Volume Bars")
            if opt_vol:
                ui.update_numeric("diag_vol_th", value=int(opt_vol))
                
            p.set(70, message="Calibrating Dollar...")
            opt_dollar = calibrate_bar_threshold(df, "Dollar Bars")
            if opt_dollar:
                ui.update_numeric("diag_dollar_th", value=int(opt_dollar))
            
            p.set(100, message="Complete")
            ui.notification_show(f"✓ Calibrated! Vol: {opt_vol:,}, Dollar: {opt_dollar:,}", type="message")

    @render_widget
    def plot_financial_bars():
        res = engineering_results_cache.get()
        if res is None: return None
        
        b_type = input.diag_bar_type()
        target = "time" if b_type == "Time Bars" else ("volume" if b_type == "Volume Bars" else "dollar")
        df = res.get(target)
        
        if df is None or df.empty: return None

        df.index = pd.to_datetime(df.index)
        df.index = df.index.strftime("%Y-%m-%d %H:%M")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC",
            increasing_line_color="lightgray",
            decreasing_line_color="#ff4b4b"
        )])
        
        fig.update_layout(
            showlegend=False,
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
            height=600,
            width=1500,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_xaxes(rangeslider_visible=False, type='category')
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=False)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.2)")

        fig = apply_theme(fig)
        fig.update_layout(
            margin=dict(l=50, r=100, t=50, b=50)
        )
        return fig
