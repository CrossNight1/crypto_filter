from shiny import ui, render, reactive, req
import faicons as fa
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import asyncio
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from numba import njit
from scipy.stats import probplot
import matplotlib.pyplot as plt
import matplotlib
import plotly.colors as pc
matplotlib.use('Agg') # Safe for web apps

from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import METRIC_LABELS, ALL_METRICS, BENCHMARK_SYMBOL, DEFAULT_FEATURES
from ml_engine.modeling.factory import ModelFactory
from ml_engine.modeling.feature_selection import FeatureSelector
from ml_engine.predictive.predictor import Predictor
from ml_engine.labeling.labeler import Labeler, TripleBarrierLabeler, StationarityLabeler, CombinedLabeler, TailSetLabeler
from src.logger import logger
from src.backtest import BacktestEngine

def predictive_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Predictive Analysis"),
            ui.input_action_button("btn_run_analysis", "Run Analysis", class_="btn-primary w-100 mt-3"),
            ui.input_select("analysis_goal", "Analysis Goal", ["Directional", "Return Value"]),
            ui.output_ui("symbol_selection_ui"),
            ui.input_select("interval", "Interval", choices=["1h", "4h", "1d"]),
            ui.output_ui("model_select_ui"),
            ui.accordion(
                ui.accordion_panel(
                    "Model Parameters",

                    ui.panel_conditional(
                        "typeof input.reg_type !== 'undefined' && (input.reg_type.includes('Random Forest') || input.reg_type.includes('XGB'))",
                        ui.input_slider("rf_max_depth", "Max Depth", min=2, max=50, value=20)
                    ),

                    ui.panel_conditional(
                        "input.analysis_goal == 'Return Value'",
                        ui.input_numeric("fwd_window", "Prediction Window", value=10, min=1, max=200)
                    ),

                    ui.input_slider("test_size", "Test Set Ratio (%)", min=0, max=50, value=20, step=5),
                    ui.input_switch("standardize", "Standardize Features", value=True),
                    ui.input_switch("remove_outliers", "Remove Outliers", value=False),
                ),
                open=True
            ),
        ),
        ui.navset_card_underline(
            # ================= DATA ENGINEERING =================
            ui.nav_panel(
                "Data Engineering",
                ui.div(
                    ui.row(
                        ui.column(
                            4,
                            ui.card(
                                ui.card_header("Structural Engineering"),
                                ui.layout_columns(
                                    ui.input_action_button(
                                        "btn_run_engineering",
                                        "Apply Engineering",
                                        class_="btn-primary w-100"
                                    ),
                                    ui.input_action_button(
                                        "btn_auto_calibrate",
                                        "Auto Calibrate",
                                        class_="btn-primary w-100"
                                    )
                                ),
                                ui.layout_columns(
                                    ui.h6("Volume Threshold"),
                                    ui.input_numeric("vol_bar_th", None, value=10000, min=1),
                                    col_widths=[6, 6]
                                ),
                                ui.layout_columns(
                                    ui.h6("Dollar Threshold"),
                                    ui.input_numeric("dollar_bar_th", None, value=1000000000, min=1),
                                    col_widths=[6, 6]
                                ),
                                ui.layout_columns(
                                    ui.h6("Structure Analysis"),
                                    ui.input_select("bar_type", None, choices=["Time Bars", "Volume Bars", "Dollar Bars"], selected="Dollar Bars"),
                                    col_widths=[6, 6]
                                ),
                                ui.output_table("engineering_stats_table")
                            )
                        ),
                        ui.column(
                            8,
                            ui.card(
                                ui.card_header("Return Distribution Comparison"),
                                output_widget("engineering_dist_plot"),
                                full_screen=True
                            )
                        )
                    ),                  
                    ui.row(
                        ui.column(
                            4,
                            ui.card(
                                ui.card_header("Labeling Settings"),
                                ui.input_action_button(
                                    "btn_run_labeling",
                                    "Apply Labeling",
                                    class_="btn-primary w-100"
                                ),
                                ui.input_select(
                                    "labeler_type",
                                    "Labeler",
                                    choices=["Combine", "Trend", "BoxRange", "Regime", "TailSet"],
                                    selected="BoxRange"
                                ),

                                ui.panel_conditional(
                                    "input.labeler_type == 'Trend' || input.labeler_type == 'Combine'",
                                    ui.row(
                                        ui.column(6, ui.input_numeric("amp_th_bps", "Amplitude (bps)", value=100)),
                                        ui.column(6, ui.input_numeric("max_inactive", "Max Inactive", value=10))
                                    )
                                ),

                                ui.panel_conditional(
                                    "input.labeler_type == 'BoxRange' || input.labeler_type == 'Combine'",
                                    ui.row(
                                        ui.column(3, ui.input_numeric("vol_window", "Vol Window", value=10)),
                                        ui.column(3, ui.input_numeric("upper_mult", "Upper Mult", value=2)),
                                        ui.column(3, ui.input_numeric("lower_mult", "Lower Mult", value=2)),
                                        ui.column(3, ui.input_numeric("max_holding", "Max Period", value=10))
                                    )
                                ),

                                ui.panel_conditional(
                                    "input.labeler_type == 'Regime'",
                                    ui.row(
                                        ui.column(6, ui.input_numeric("stat_window", "Stationarity Window", value=20)),
                                        ui.column(6, ui.input_slider("vote_th", "Vote Threshold", min=0.1, max=1.0, value=0.4, step=0.05))
                                    )
                                ),
                                ui.panel_conditional(
                                    "input.labeler_type == 'TailSet'",
                                    ui.row(
                                        ui.column(6, ui.input_numeric("tail_window", "Window", value=1)),
                                        ui.column(6, ui.input_numeric("tail_threshold", "Multiplier Threshold", value=1.0, step=0.1))
                                    )
                                ),                                
                                ui.output_table("dist_train_table")
                            )
                        ),
                        ui.column(
                            8,
                            ui.card(
                                ui.card_header("Labeling Preview"),
                                output_widget("label_chart"),
                                full_screen=True
                            )
                        )
                    ),
                    ui.row(
                        ui.column(
                            4,
                            ui.card(
                                ui.card_header("Feature Engineering"),
                                ui.input_action_button("btn_run_feature_analysis", "Analyze & Filter Features", class_="btn-primary w-100"),
                                ui.layout_columns(
                                    ui.input_selectize("eng_features", "Candidate Features", choices=ALL_METRICS, multiple=True, selected=DEFAULT_FEATURES),
                                    ui.input_numeric("eng_lookback", "Lookback", value=20, min=2),
                                    ui.input_numeric("eng_min_samples", "Min Samples", value=100, min=10),
                                    ui.input_numeric("vif_th", "VIF Threshold", value=10.0, min=1.1, step=0.5),
                                    col_widths=[12, 4, 4, 4]
                                ),
                                ui.output_table("feature_stats_table")
                            )
                        ),
                        ui.column(
                            8,
                            ui.card(
                                ui.card_header("Feature Distribution"),
                                output_widget("feature_dist_plot"),
                                full_screen=True
                            )
                        )
                    )
                ),
                value="Data Engineering"
            ),



            # ================= PERFORMANCE =================
            ui.nav_panel(
                "Performance",
                ui.div(

                    ui.output_ui("kpi_row"),

                    ui.row(
                        ui.column(
                            12,
                            ui.card(ui.h6("Live Predictions"), ui.output_ui("live_predictions_table")),
                            ui.card(ui.h6("Classification Report"), ui.output_table("classification_report_table")),
                        )
                    ),

                    ui.row(
                        ui.column(6, ui.card(ui.h6("Confusion Matrix / Prediction Fit"), output_widget("cm_or_fit_plot"))),
                        ui.column(6, ui.card(ui.h6("Feature Importance"), output_widget("importance_plot")))
                    )
                )
            ),

            # ================= VISUALIZE =================
            ui.nav_panel(
                "Visualize",
                ui.div(

                    ui.row(
                        ui.column(
                            12,
                            ui.card(
                                ui.card_header("Cumulative PnL"),
                                output_widget("pnl_plot")
                            )
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.card(
                                ui.card_header("Strategy Performance Metrics"),
                                ui.output_table("strategy_metrics_table")
                            )
                        )
                    ),

                    ui.row(
                        ui.column(
                            12,
                            ui.card(
                                ui.card_header("Predicted Labeling"),
                                output_widget("predicted_label_chart"),
                                full_screen=True
                            )
                        )
                    ),
                    
                    ui.row(
                        ui.column(6, ui.card(
                            ui.card_header("Primary Model: SHAP Mean Importance"),
                            output_widget("shap_summary_primary"),
                            full_screen=True
                        )),
                        ui.column(6, ui.card(
                            ui.card_header("Meta Model: SHAP Mean Importance"),
                            output_widget("shap_summary_meta"),
                            full_screen=True
                        ))
                    ),
                    
                    ui.row(
                        ui.column(6, ui.card(
                            ui.card_header("Global Impact: SHAP Beeswarm"),
                            output_widget("shap_beeswarm_plot"),
                            full_screen=True
                        )),
                        ui.column(6, ui.card(
                            ui.card_header("SHAP Local Explanation"),
                            output_widget("shap_local_plot"),
                            full_screen=True
                        ))
                    )
                )
            )
        , id="main_navset")
    )

def predictive_server(input, output, session):
    manager = DataManager()
    engine = MetricsEngine()
    
    # Reactive state
    inventory = reactive.Value({})
    results = reactive.Value(None)
    engineered_data = reactive.Value({"volume": None, "dollar": None, "ticker": None, "interval": None})
    eng_trigger = reactive.Value(0)
    label_trigger = reactive.Value(0)
    vol_th_sync = reactive.Value(None)
    dollar_th_sync = reactive.Value(None)
    labeled_data = reactive.Value({})  # Cache for labeled data: {ticker_interval_labeler: {"df": df, "y": y_series, "params": {...}}}
    filtered_features = reactive.Value(None) # Cache for feature filtering results from Data Engineering

    @njit
    def _volume_bar_core(open_, high_, low_, close_, volume_, threshold):
        n = len(close_)
        
        o_list = []
        h_list = []
        l_list = []
        c_list = []
        v_list = []
        t_list = []
        
        cum_vol = 0.0
        o = h = l = c = 0.0
        start_idx = 0
        
        for i in range(n):
            if cum_vol == 0.0:
                o = open_[i]
                h = high_[i]
                l = low_[i]
                start_idx = i
            
            cum_vol += volume_[i]
            if high_[i] > h: h = high_[i]
            if low_[i] < l: l = low_[i]
            c = close_[i]
            
            if cum_vol >= threshold:
                o_list.append(o)
                h_list.append(h)
                l_list.append(l)
                c_list.append(c)
                v_list.append(cum_vol)
                t_list.append(start_idx)
                cum_vol = 0.0
        
        return t_list, o_list, h_list, l_list, c_list, v_list


    @njit
    def _dollar_bar_core(open_, high_, low_, close_, volume_, threshold):
        n = len(close_)
        
        o_list = []
        h_list = []
        l_list = []
        c_list = []
        d_list = []
        v_list = []
        t_list = []
        
        cum_dollar = 0.0
        cum_vol = 0.0
        o = h = l = c = 0.0
        start_idx = 0
        
        for i in range(n):
            if cum_dollar == 0.0:
                o = open_[i]
                h = high_[i]
                l = low_[i]
                start_idx = i
            
            dollar_val = close_[i] * volume_[i]
            cum_dollar += dollar_val
            cum_vol += volume_[i]
            
            if high_[i] > h: h = high_[i]
            if low_[i] < l: l = low_[i]
            c = close_[i]
            
            if cum_dollar >= threshold:
                o_list.append(o)
                h_list.append(h)
                l_list.append(l)
                c_list.append(c)
                d_list.append(cum_dollar)
                v_list.append(cum_vol)
                t_list.append(start_idx)
                cum_dollar = 0.0
                cum_vol = 0.0
        
        return t_list, o_list, h_list, l_list, c_list, d_list, v_list


    def construct_volume_bars(df, threshold):
        open_ = df["open"].values.astype(np.float64)
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        close_ = df["close"].values.astype(np.float64)
        volume_ = df["volume"].values.astype(np.float64)
        
        t, o, h, l, c, v = _volume_bar_core(open_, high_, low_, close_, volume_, threshold)
        
        if len(o) == 0:
            return pd.DataFrame()
        
        idx = df.index[np.array(t)]
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
        return out


    def construct_dollar_bars(df, threshold):
        open_ = df["open"].values.astype(np.float64)
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        close_ = df["close"].values.astype(np.float64)
        volume_ = df["volume"].values.astype(np.float64)
        
        t, o, h, l, c, d, v = _dollar_bar_core(open_, high_, low_, close_, volume_, threshold)
        
        if len(o) == 0:
            return pd.DataFrame()
        
        idx = df.index[np.array(t)]
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "dollar_vol": d}, index=idx)
        return out


    @reactive.Effect
    @reactive.event(input.main_navset)
    def auto_run_engineering():
        if input.main_navset() == "Data Engineering":
            ticker = input.selected_ticker() if input.analysis_goal() == "Directional" else (list(input.selected_tickers())[0] if input.selected_tickers() else None)
            interval = input.interval()
            cache = engineered_data.get()
            
            if ticker and (cache["volume"] is None or cache["ticker"] != ticker or cache["interval"] != interval):
                logger.log("Predictive", "INFO", f"Auto-triggering engineering for {ticker} ({interval})")
                eng_trigger.set(eng_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(input.btn_run_engineering)
    def btn_trigger_eng():
        eng_trigger.set(eng_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(eng_trigger)
    def _execute_engineering_on_trigger():
        # Force execution of the lazy reactive.calc to update engineered_data cache
        if eng_trigger.get() > 0:
            logger.log("Predictive", "INFO", "Executing scheduled engineering calculation")
            engineering_result()
            
            # Also auto-trigger labeling if we are not on Time Bars
            if input.bar_type() != "Time Bars":
                logger.log("Predictive", "INFO", "Auto-triggering labeling after engineering")
                label_trigger.set(label_trigger.get() + 1)
    @reactive.Effect
    @reactive.event(input.vol_bar_th)
    def _reset_vol_sync():
        vol_th_sync.set(None)

    @reactive.Effect
    @reactive.event(input.dollar_bar_th)
    def _reset_dollar_sync():
        dollar_th_sync.set(None)

    @reactive.Effect
    @reactive.event(input.btn_auto_calibrate)
    async def auto_calibrate_thresholds():
        """Auto-calibrate Volume and Dollar thresholds to minimize normality loss"""
        # Get current ticker and interval
        goal = input.analysis_goal()
        if goal == "Directional":
            ticker = input.selected_ticker()
        else:
            ts = list(input.selected_tickers())
            if not ts:
                ui.notification_show("Please select a ticker first.", type="warning")
                return
            ticker = ts[0]
        
        interval = input.interval()
        df = manager.load_data(ticker, interval)
        
        if df is None or df.empty:
            ui.notification_show(f"No data available for {ticker}.", type="error")
            return
        
        if 'open_time' in df.columns:
            df = df.set_index(pd.to_datetime(df['open_time']))
        
        ui.notification_show(f"Calibrating thresholds for {ticker}", duration=5)
                
        def calibration_loss(returns, N_min=10000):
            if len(returns) < 30:
                return 1e10

            N = len(returns)
            s = skew(returns)
            k = kurtosis(returns) - 3

            return (
                s**2 +
                1000 * max(0, (N_min - N)/N_min)**2
            )
        
        def objective_volume(vol_threshold):
            """Objective function for volume bars"""
            try:
                vol_df = construct_volume_bars(df, int(vol_threshold[0]))
                if vol_df.empty or len(vol_df) < 10:
                    return 1e10
                returns = np.log(vol_df['close'] / vol_df['close'].shift(1)).dropna()
                loss = calibration_loss(returns)
                return loss
            except Exception as e:
                return 1e10
        
        def objective_dollar(dollar_threshold):
            """Objective function for dollar bars"""
            try:
                dollar_df = construct_dollar_bars(df, int(dollar_threshold[0]))
                if dollar_df.empty or len(dollar_df) < 10:
                    return 1e10
                returns = np.log(dollar_df['close'] / dollar_df['close'].shift(1)).dropna()
                loss = calibration_loss(returns)
                return loss
            except Exception as e:
                return 1e10

        def objective_volume_log(x):
            th = int(10**x[0])
            return objective_volume([th])

        def objective_dollar_log(x):
            th = int(10**x[0])
            return objective_dollar([th])
        
        # Optimize Volume threshold with multiple starting points
        logger.log("Predictive", "INFO", "Optimizing Volume threshold...")
        print(f"\n=== Optimizing Volume Threshold for {ticker} ===")
        
        best_vol_result = None
        best_vol_loss = 1e10
        
        for x0 in [1000, 5000, 10000, 50000]:
            result = minimize(
                objective_volume_log,
                x0=[np.log10(x0)],
                bounds=[(np.log10(100), np.log10(1000000000))],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol':1e-3, 'fatol':1e-3}
            )
            if result.fun < best_vol_loss:
                best_vol_loss = result.fun
                best_vol_result = result
        
        optimal_vol = int(10**best_vol_result.x[0])
        
        # Optimize Dollar threshold with multiple starting points
        logger.log("Predictive", "INFO", "Optimizing Dollar threshold...")        
        best_dollar_result = None
        best_dollar_loss = 1e10
        
        for x0 in [100000000, 500000000, 1000000000, 5000000000]:
            result = minimize(
                objective_dollar_log,
                x0=[np.log10(x0)],
                bounds=[(np.log10(1000), np.log10(100000000000))],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol':1e-3, 'fatol':1e-3}
            )
            if result.fun < best_dollar_loss:
                best_dollar_loss = result.fun
                best_dollar_result = result

        optimal_dollar = int(10**best_dollar_result.x[0])
        
        # Update sync values for immediate reactive update
        vol_th_sync.set(optimal_vol)
        dollar_th_sync.set(optimal_dollar)
        
        # Update UI inputs
        ui.update_numeric("vol_bar_th", value=optimal_vol)
        ui.update_numeric("dollar_bar_th", value=optimal_dollar)
        
        ui.notification_show(
            f"✓ Calibration complete! Volume: {optimal_vol:,}, Dollar: {optimal_dollar:,}",
            type="message",
            duration=10
        )
        logger.log("Predictive", "INFO", f"Calibrated thresholds - Vol: {optimal_vol}, Dollar: {optimal_dollar}")
        
        # Trigger engineering with new thresholds
        eng_trigger.set(eng_trigger.get() + 1)

    @reactive.calc
    @reactive.event(eng_trigger)
    def engineering_result():
        # Get Current Symbol and Interval context
        goal = input.analysis_goal()
        if goal == "Directional":
            ticker = input.selected_ticker()
        else:
            ts = list(input.selected_tickers())
            if not ts: return None
            ticker = ts[0]
            
        interval = input.interval()
        df = manager.load_data(ticker, interval)
        if df is None or df.empty: return None
        
        # Consistent Indexing
        if 'open_time' in df.columns:
            df = df.set_index(pd.to_datetime(df['open_time']))
            
        # 1. Time Bars (Original)
        time_df = df.copy()
        time_df['ret'] = np.log(time_df['close'] / time_df['close'].shift(1))
        
        # 2. Volume Bars
        v_sync = vol_th_sync.get()
        vol_th = v_sync if v_sync is not None else input.vol_bar_th()
        vol_df = construct_volume_bars(df, vol_th)
        if not vol_df.empty:
            vol_df['ret'] = np.log(vol_df['close'] / vol_df['close'].shift(1))
            
        # 3. Dollar Bars
        d_sync = dollar_th_sync.get()
        dollar_th = d_sync if d_sync is not None else input.dollar_bar_th()
        dollar_df = construct_dollar_bars(df, dollar_th)
        if not dollar_df.empty:
            dollar_df['ret'] = np.log(dollar_df['close'] / dollar_df['close'].shift(1))
            
        # Update Cache for run_analysis
        engineered_data.set({
            "volume": vol_df,
            "dollar": dollar_df,
            "ticker": ticker,
            "interval": interval
        })
            
        return {
            "time": time_df,
            "volume": vol_df,
            "dollar": dollar_df,
            "ticker": ticker
        }

    @reactive.calc
    @reactive.event(input.btn_run_feature_analysis)
    def feature_analysis_result():
        # Requires engineering to have run
        eng = engineering_result()
        if eng is None:
            ui.notification_show("Please click 'Apply Engineering' first.", type="error")
            return None
            
        # Get selected bar type
        b_type = input.bar_type()
        target_key = "time" if b_type == "Time Bars" else ("volume" if b_type == "Volume Bars" else "dollar")
        df = eng.get(target_key)
        
        if df is None or df.empty:
            ui.notification_show(f"No data for {b_type}. Apply engineering first.", type="error")
            return None
            
        # Select features
        features = list(input.eng_features())
        if not features:
            ui.notification_show("Please select at least one feature.", type="warning")
            return None
            
        # Params from UI
        lookback = input.eng_lookback()
        min_s = input.eng_min_samples()
        v_th = input.vif_th()

        # Calculate features using MetricsEngine
        try:
            # We need standard OHLCV for MetricsEngine
            calc_df = df.copy()
            # Ensure volume column exists for internal indicators
            if 'dollar_vol' in calc_df.columns and 'volume' not in calc_df.columns:
                calc_df['volume'] = calc_df['dollar_vol'] / calc_df['close']
            
            # Check if sufficient data length for lookback
            if len(calc_df) < lookback + 10:
                ui.notification_show(f"Insufficient data ({len(calc_df)}) for lookback {lookback}.", type="error")
                return None

            feat_df = engine.calculate_all_indicators(calc_df, window=lookback)
            
            # Filter to selected features
            valid_feats = [f for f in features if f in feat_df.columns]
            if not valid_feats:
                ui.notification_show("None of the selected features could be calculated.", type="error")
                return None
                
            feat_df = feat_df[valid_feats].dropna()
            
            if len(feat_df) < min_s:
                ui.notification_show(f"Only {len(feat_df)} samples remaining after lookback/NaN removal. Need {min_s}.", type="warning")
                return None
            
            # Feature Stats (Stats + VIF)
            stats_list = []            
            vif_dict = {}
            if len(valid_feats) > 1:
                # Multicollinearity check
                try:
                    # Drop constant columns if any (std=0)
                    X_vif = feat_df.loc[:, feat_df.std() > 0]
                    if not X_vif.empty and X_vif.shape[1] > 1:
                        X = sm.add_constant(X_vif)
                        for i, col in enumerate(X_vif.columns):
                            v = variance_inflation_factor(X.values, i + 1)
                            vif_dict[col] = min(v, 99.0) # Cap for display
                except:
                    pass
            
            for col in valid_feats:
                s = feat_df[col].describe()
                vif = vif_dict.get(col, 1.0)
                
                # Highlight or filter? User said "VIF threshold filter"
                # I'll keep them in stats but maybe flag them? 
                # Actually, let's filter the final feat_df for the plot if VIF > threshold
                
                stats_list.append({
                    "Feature": col,
                    "Mean": s['mean'],
                    "Std": s['std'],
                    "Skew": skew(feat_df[col]),
                    "Kurtosis": kurtosis(feat_df[col], fisher=True),
                    "VIF": vif,
                    "Keep": "YES" if vif <= v_th else "NO"
                })
            
            stats_df = pd.DataFrame(stats_list)
            
            # Filter feat_df for plot and further use
            clean_feats = [s['Feature'] for s in stats_list if s['VIF'] <= v_th]
            feat_df_clean = feat_df[clean_feats] if clean_feats else pd.DataFrame()

            return {
                "feat_df": feat_df,
                "feat_df_clean": feat_df_clean,
                "stats_df": stats_df,
                "bar_type": b_type,
                "ticker": eng['ticker'],
                "vif_threshold": v_th
            }
            
        except Exception as e:
            logger.log("Predictive", "ERROR", f"Feature analysis error: {e}")
            ui.notification_show(f"Analysis Error: {e}", type="error")
            return None

    @reactive.effect
    def _cache_filtered_features():
        res = feature_analysis_result()
        if res is not None:
            # Get list of features where VIF <= threshold (the YES ones)
            clean = [s['Feature'] for s in res['stats_df'].to_dict('records') if s['Keep'] == "YES"]
            if clean:
                filtered_features.set(clean)
                logger.log("Predictive", "INFO", f"Cached {len(clean)} filtered features from Data Engineering tab.")

    @render.table
    def feature_stats_table():
        res = feature_analysis_result()
        if res is None: return None
        
        df = res['stats_df'].copy()
        
        # Round for display but keep as numeric for formatter if possible, 
        # or just use the styled formatter
        styled = (
            df.style
            .hide_index()
            .format({
                "Mean": "{:.4f}",
                "Std": "{:.4f}",
                "Skew": "{:.4f}",
                "Kurtosis": "{:.4f}",
                "VIF": "{:.2f}"
            })
            .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("color", "#000000"), ("background-color", "#FCC780"), ("border", "1px solid #1a4da3"), ("text-transform", "uppercase"), ("font-weight", "700")]},
                {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #1a4da3")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "2px solid #1a4da3"), ("width", "100%")]},
                {"selector": "th", "props": [("font-size", "11px"), ("text-align", "center")]}
            ])
        )
        return styled

    @render_widget
    def feature_dist_plot():
        res = feature_analysis_result()
        if res is None: return None
        
        # Use the CLEANED dataframe (after VIF filtering)
        df = res['feat_df_clean']
        if df.empty: return None
        
        # Limit to first 8 for visual clarity if many survive
        cols = df.columns
        
        fig = go.Figure()
        for col in cols:
            vals = df[col].dropna()
            if vals.std() > 0:
                vals = (vals - vals.mean()) / vals.std()
                
            fig.add_trace(go.Histogram(
                x=vals,
                name=col,
                opacity=0.6,
                histnorm='probability density'
            ))
            
        fig.update_layout(
            template="plotly_dark",
            height=400,
            width=965,
            barmode='overlay',
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
        )
        return fig

    @reactive.effect
    def _():
        inv = manager.get_inventory()
        inventory.set(inv)

    @render.ui
    def symbol_selection_ui():
        inv = inventory.get()
        if not inv: 
            return ui.div(ui.p("No data found. Sync symbols in Data Loader first.", class_="text-danger small mt-2"))
        
        # Filter tickers by selected interval
        interval = input.interval()
        if interval:
            tickers = sorted([sym for sym, intervals in inv.items() if interval in intervals])
        else:
            tickers = sorted(list(inv.keys()))
        
        if input.analysis_goal() == "Directional":
            return ui.input_select("selected_ticker", "Symbol", choices=tickers, selected='BTCUSDT')
        else:
            return ui.input_selectize("selected_tickers", "Symbols", choices=tickers, selected='BTCUSDT', multiple=True)

    @reactive.effect
    def _():
        inv = inventory.get()
        if not inv: return
        intervals = sorted(list(set(i for ivs in inv.values() for i in ivs)))
        ui.update_select("interval", choices=intervals, selected='1h' if '1h' in intervals else intervals[0])

    @render.ui
    def model_select_ui():
        if input.analysis_goal() == "Directional":
            return ui.input_select("reg_type", "Model", choices=["Random Forest Classifier", "XGB Classifier"])
        else:
            return ui.input_select("reg_type", "Model", choices=["Linear (OLS)", "Random Forest Regressor", "XGB Regressor"])


    @reactive.effect
    @reactive.event(input.btn_run_analysis)
    async def run_analysis():
        # 0. Wait for essentials safely
        if input.reg_type() is None or input.interval() is None:
            logger.log("Predictive", "INFO", "Waiting for UI inputs to initialize...")
            ui.notification_show("Please wait for the interface to load", type="warning")
            return

        logger.log("Predictive", "INFO", "--- Analysis Triggered ---")
        
        try:
            logger.log("Predictive", "INFO", "Step 0: Reading UI inputs")
            goal = input.analysis_goal()
            is_classification = (goal == "Directional")
            
            # Read dynamic ticker selection
            if is_classification:
                tickers = [input.selected_ticker()]
            else:
                tickers = list(input.selected_tickers())
                
            if not tickers:
                ui.notification_show("Select at least one ticker", type="error")
                return

            # Read features
            x_features = list(input.eng_features())
            if not x_features:
                ui.notification_show("Select at least one predictor feature", type="error")
                return

            # Read remaining params
            reg_type = input.reg_type()
            test_ratio = input.test_size() / 100.0
            
            rf_max_depth = input.rf_max_depth() if ("Random Forest" in reg_type or "XGB" in reg_type) else None
            interval = input.interval()
            ind_window = input.eng_lookback()
            fwd_window = 1 if is_classification else input.fwd_window()
            min_samples = input.eng_min_samples()
            standardize_flag = input.standardize()

            # 1. Gather Data
            logger.log("Predictive", "INFO", f"Step 1: Gathering data for {len(tickers)} symbols")
            total_steps = len(tickers) + 7  # tickers + preprocess + VIF + primary + meta + backtest + SHAP + store
            with ui.Progress(min=0, max=total_steps) as p:
                p.set(0, message="Stacking data...")
                all_rows = []
                
                # Check Bar Type and Cache
                b_type = input.bar_type()
                eng_cache = engineered_data.get()
                
                # Labeler params from UI
                l_params = None
                if is_classification:
                    l_type = input.labeler_type()
                    if l_type == "Trend": 
                        l_params = {'type': "Trend", 'amp_th': input.amp_th_bps(), 'max_inactive': input.max_inactive()}
                    elif l_type == "BoxRange": 
                        l_params = {'type': "BoxRange", 'vol_window': input.vol_window(), 'upper_mult': input.upper_mult(), 'lower_mult': input.lower_mult(), 'max_holding': input.max_holding()}
                    elif l_type == "Combine":
                        l_params = {
                            'type': "Combine", 
                            'amp_th': input.amp_th_bps(), 'max_inactive': input.max_inactive(),
                            'vol_window': input.vol_window(), 'upper_mult': input.upper_mult(), 'lower_mult': input.lower_mult(), 'max_holding': input.max_holding()
                        }
                    elif l_type == "Regime": 
                        l_params = {'type': "Regime", 'window': input.stat_window(), 'vote_th': input.vote_th()}
                    else:
                        l_params = {'type': "TailSet", 'window': input.tail_window(), 'threshold': input.tail_threshold()}

                for i, sym in enumerate(tickers):
                    df = None
                    if b_type == "Time Bars":
                        df = manager.load_data(sym, interval)
                    else:
                        # Use Engineered Cache
                        target_key = "volume" if b_type == "Volume Bars" else "dollar"
                        if eng_cache["ticker"] == sym and eng_cache["interval"] == interval and eng_cache[target_key] is not None:
                            df = eng_cache[target_key]
                        else:
                            ui.notification_show(f"Engineered data for {sym} ({interval}) not found. Please click 'Apply Engineering' in the Data Engineering tab first.", type="error")
                            return

                    if df is not None and len(df) > max(ind_window, fwd_window) + 10:
                        if 'open_time' in df.columns:
                            df = df.set_index(pd.to_datetime(df['open_time']))
                        
                        try:
                            feats_data = {feat: engine.calculate_rolling_metric(df, feat, window=ind_window, interval=interval) for feat in x_features}
                            
                            # Check label cache first
                            cache_key = None
                            if is_classification:
                                cache_key = f"{sym}_{interval}_{b_type}_{l_params['type']}"
                            else:
                                cache_key = f"{sym}_{interval}_{b_type}_regression_{fwd_window}"
                            
                            label_cache = labeled_data.get()
                            if cache_key in label_cache:
                                print(f"✓ Using cached labels for {sym} (key: {cache_key})")
                                y_series = label_cache[cache_key]['y']
                            else:
                                print(f"⚠ No cached labels found for {sym}, calculating new labels...")
                                if is_classification:
                                    prices = df['close']
                                    if l_params['type'] == "Trend":
                                        lobj = Labeler(amplitude_threshold=l_params['amp_th'], max_inactive_period=l_params['max_inactive'])
                                        y_series = lobj.label(prices)['label']
                                    elif l_params['type'] == "BoxRange":
                                        lobj = TripleBarrierLabeler(vol_window=l_params['vol_window'], upper_mult=l_params['upper_mult'], lower_mult=l_params['lower_mult'], max_holding_period=l_params['max_holding'])
                                        y_series = lobj.label(prices)['label']
                                    elif l_params['type'] == "Combine":
                                        lobj = CombinedLabeler(
                                            amplitude_threshold=l_params['amp_th'], 
                                            max_inactive_period=l_params['max_inactive'],
                                            vol_window=l_params['vol_window'], 
                                            upper_mult=l_params['upper_mult'], 
                                            lower_mult=l_params['lower_mult'], 
                                            max_holding=l_params['max_holding']
                                        )
                                        y_series = lobj.label(prices)['label']
                                    elif l_params['type'] == "Regime":
                                        lobj = StationarityLabeler(window=l_params['window'], vote_th=l_params['vote_th'])
                                        y_series, _ = lobj.label(prices)
                                    elif l_params['type'] == "TailSet":
                                        w = l_params['window'] if l_params['window'] > 0 else 1
                                        lobj = TailSetLabeler(ret_window=w, threshold=l_params['threshold'])
                                        y_series = lobj.label(prices)['label']
                                    
                                    y_series.index = prices.index
                                    
                                    # Shift Correction for forward prediction
                                    if l_params['type'] == "TailSet" and l_params['window'] > 1:
                                        y_series = y_series.shift(-l_params['window'])
                                    else:
                                        y_series = y_series.shift(-1)
                                else:
                                    prices = pd.to_numeric(df['close'], errors='coerce').ffill().values
                                    y_series = pd.Series(np.log(prices[1:] / prices[:-1]), index=df.index[1:]).rolling(window=fwd_window).sum().shift(-fwd_window)
                            
                            temp_df = pd.DataFrame(feats_data)
                            temp_df['Target_Y'] = y_series
                            temp_df['raw_return'] = np.log(pd.to_numeric(df['close'], errors='coerce').ffill() / pd.to_numeric(df['close'], errors='coerce').ffill().shift(1)).shift(-1)
                            temp_df = temp_df.dropna()
                            
                            if len(temp_df) >= min_samples:
                                all_rows.append(temp_df)
                        except Exception as e:
                            logger.log("Predictive", "WARNING", f"Error processing {sym}: {str(e)}")
                    
                    p.set(i + 1)
                    await asyncio.sleep(0.01)
                
                if not all_rows:
                    logger.log("Predictive", "ERROR", "No valid data points found after stacking")
                    ui.notification_show("No valid data points found. Try increasing lookback or symbols.", type="error")
                    return
                logger.log("Predictive", "INFO", f"Step 2: Preprocessing stacked data ({sum(len(df) for df in all_rows)} rows)")
                step = len(tickers)
                p.set(step, message="Preprocessing & cleaning data...")
                final_df = pd.concat(all_rows).sort_index()
                
                X_data = final_df[x_features].copy()
                Y_data = final_df['Target_Y'].copy()
                Y_ret = final_df['raw_return']
                
                # Clean Infs/NaNs
                combined = pd.concat([X_data, Y_data, Y_ret], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                X_data, Y_data, Y_ret = combined[x_features], combined['Target_Y'], combined['raw_return']
                
                if standardize_flag:
                    logger.log("Predictive", "INFO", "Standardizing features")
                    p.set(message="Standardizing features...")
                    for f in x_features:
                        if f in X_data.columns:
                            m, s = X_data[f].mean(), X_data[f].std()
                            if s > 1e-9: X_data[f] = (X_data[f] - m) / s
                
                # Check for manually filtered features from Data Engineering tab Cache
                cached_feats = filtered_features.get()
                if cached_feats and all(f in x_features for f in cached_feats):
                    logger.log("Predictive", "INFO", f"Using {len(cached_feats)} manually filtered features from cache.")
                    active_features = cached_feats
                else:
                    logger.log("Predictive", "INFO", "Running VIF filter (no cache or mismatch)")
                    step += 1
                    p.set(step, message="Running VIF filter...")
                    vif_threshold = input.vif_th()
                    active_features = FeatureSelector.apply_vif_filter(X_data, threshold=vif_threshold) if len(x_features) > 1 else x_features
                
                if not active_features:
                    logger.log("Predictive", "ERROR", "No features left after VIF filter")
                    ui.notification_show("No features left after VIF filter. Try a higher VIF threshold or fewer predictors.", type="error")
                    return
                    
                X_raw = X_data[active_features].copy()
                
                # 3-Fold Sequential Split
                n = len(X_raw)
                test_start = int(n * (1 - test_ratio))
                
                # Auto-Split Training set into 2 folds:
                train_n = test_start
                meta_start = int(train_n * 0.8)
                
                logger.log("Predictive", "INFO", f"Splitting data: n={n}, test_start={test_start}, auto-split train at {meta_start}")
                
                # Part 1: Primary Train
                X_train_primary = X_raw.iloc[:meta_start]
                y_train_primary = Y_data.iloc[:meta_start]
                y_ret_primary = Y_ret.iloc[:meta_start]
                
                # Part 2: Meta Train
                X_train_meta = X_raw.iloc[meta_start:test_start]
                y_train_meta = Y_data.iloc[meta_start:test_start]
                y_ret_meta = Y_ret.iloc[meta_start:test_start]
                
                # Part 3: Test
                X_test = X_raw.iloc[test_start:]
                y_test = Y_data.iloc[test_start:]
                y_ret_test = Y_ret.iloc[test_start:]
                
                logger.log("Predictive", "INFO", f"Sequential Split - Primary: {len(X_train_primary)}, Meta: {len(X_train_meta)}, Test: {len(X_test)}")
                # 3. Train Primary
                logger.log("Predictive", "INFO", f"Step 3: Training Primary model ({reg_type})")
                step += 1
                p.set(step, message=f"Training Primary model ({reg_type})...")
                model = ModelFactory.create_model(reg_type, n_estimators=200, max_depth=rf_max_depth)
                model.fit(X_train_primary, y_train_primary)
                logger.log("Predictive", "INFO", "Primary model training complete")

                # 4. Train Meta-Model using Sophisticated Backtest
                meta_model = None
                meta_results = {}
                try:
                    logger.log("Predictive", "INFO", "Step 4: Training Meta-model")
                    step += 1
                    p.set(step, message="Training Meta-model & Backtest...")
                    # Get Primary predictions on the Meta fold
                    p_meta_input = model.predict(X_train_meta)
                    p_meta_signals = pd.Series(p_meta_input, index=X_train_meta.index)
                    if not is_classification:
                        p_meta_signals = np.sign(p_meta_signals)
                                    
                    if len(tickers) == 1:
                        logger.log("Predictive", "INFO", "Single symbol detected, using BacktestEngine for meta-labels")
                        ticker = tickers[0]
                        full_df = manager.load_data(ticker, interval)
                        if 'open_time' in full_df.columns:
                            full_df = full_df.set_index(pd.to_datetime(full_df['open_time']))
                        
                        # Align df to meta-fold
                        meta_df = full_df.reindex(X_train_meta.index)
                        
                        # Naive meta-labels
                        y_meta = (p_meta_signals * y_ret_meta > 0).astype(int)
                    else:
                        logger.log("Predictive", "INFO", "Multiple symbols: using naive meta-labels (sign match)")
                        if is_classification:
                            y_meta = (p_meta_input == y_train_meta).astype(int)
                        else:
                            y_meta = (np.sign(p_meta_input) * np.sign(y_ret_meta) > 0.0000).astype(int)

                    # Meta features = original features + primary prediction
                    meta_features = pd.concat([X_train_meta, pd.Series(p_meta_input, index=X_train_meta.index, name='primary_pred')], axis=1)
                    meta_model = LogisticRegression(max_iter=1000)
                    meta_model.fit(meta_features, y_meta)
                    logger.log("Predictive", "INFO", "Meta-model training complete")
                    
                    # 4.1 Run Sophisticated Backtest on TEST SET
                    logger.log("Predictive", "INFO", "Step 4.1: Running Backtest on Test Set")
                    step += 1
                    p.set(step, message="Running backtest on test set...")
                    if len(tickers) == 1:
                        ticker = tickers[0]
                        full_df = manager.load_data(ticker, interval)
                        if 'open_time' in full_df.columns:
                            full_df = full_df.set_index(pd.to_datetime(full_df['open_time']))
                        
                        test_df_ohlcv = full_df.reindex(X_test.index)
                        returns_all = np.log(full_df['close'] / full_df['close'].shift(1))
                        vol_test = returns_all.rolling(window=input.vol_window()).std().reindex(X_test.index).fillna(0.005)
                        
                        # RAW signals backtest
                        p_test_input = model.predict(X_test)
                        p_test_signals = pd.Series(p_test_input, index=X_test.index)
                        if not is_classification:
                            p_test_signals = np.sign(p_test_signals)
                        
                        bt_engine = BacktestEngine()
                        _, raw_metrics, raw_equity, _, raw_sig, raw_size = bt_engine.run(test_df_ohlcv, p_test_signals, vol_test, pd.Series(1.0, index=X_test.index))
                        
                        # META signals backtest
                        meta_test_features = pd.concat([X_test, pd.Series(p_test_input, index=X_test.index, name='primary_pred')], axis=1)
                        meta_probs = meta_model.predict_proba(meta_test_features)
                        idx_1 = np.where(meta_model.classes_ == 1)[0]
                        if len(idx_1) > 0:
                            meta_sizing = meta_probs[:, idx_1[0]]
                            meta_sizing = np.clip((meta_sizing * 2 - 1) * 2, 0, 1) 
                        else:
                            meta_sizing = np.zeros(len(X_test))

                        trade_results, metrics, equity_curve, buy_hold_equity, sig_hist, size_hist = bt_engine.run(test_df_ohlcv, p_test_signals, vol_test, pd.Series(meta_sizing, index=X_test.index))
                        
                        meta_results = {
                            'raw_metrics': raw_metrics,
                            'raw_equity': raw_equity,
                            'raw_sig': raw_sig,
                            'raw_size': raw_size,
                            'meta_metrics': metrics,
                            'meta_equity': equity_curve,
                            'meta_sig': sig_hist,
                            'meta_size': size_hist,
                            'buy_hold_equity': buy_hold_equity,
                        }

                except Exception as e:
                    logger.log("Predictive", "WARNING", f"Meta-sizing/Backtest failed: {str(e)}")
                    import traceback
                    logger.log("Predictive", "DEBUG", traceback.format_exc())
                
                # 4.2 Calculate SHAP Values
                shap_results = {}
                try:
                    logger.log("Predictive", "INFO", "Step 4.2: Calculating SHAP values")
                    step += 1
                    p.set(step, message="Calculating SHAP values...")
                    # Primary Model SHAP (Tree-based if RF/XGB)
                    if model is not None:
                        # Use a sample of test data for SHAP to keep it fast
                        X_shap_primary = X_test.iloc[-200:] if len(X_test) > 200 else X_test
                        explainer_primary = shap.Explainer(model)
                        shap_values_primary = explainer_primary(X_shap_primary)
                        shap_results['primary'] = {
                            'values': shap_values_primary,
                            'features': X_test.columns.tolist()
                        }
                    
                    # Meta Model SHAP (Linear)
                    if meta_model is not None:
                        X_shap_meta = meta_test_features.iloc[-200:] if len(meta_test_features) > 200 else meta_test_features
                        explainer_meta = shap.Explainer(meta_model, meta_features)
                        shap_values_meta = explainer_meta(X_shap_meta)
                        shap_results['meta'] = {
                            'values': shap_values_meta,
                            'features': meta_test_features.columns.tolist()
                        }
                    logger.log("Predictive", "INFO", "SHAP calculation complete")
                except Exception as se:
                    logger.log("Predictive", "WARNING", f"SHAP calculation failed: {str(se)}")

                # 5. Store Results
                logger.log("Predictive", "INFO", "Step 5: Storing results")
                step += 1
                p.set(step, message="Storing results...")
                results.set({
                    'model': model,
                    'meta_model': meta_model,
                    'meta_results': meta_results,
                    'shap_results': shap_results,
                    'X_train_primary': X_train_primary,
                    'y_train_primary': y_train_primary,
                    'y_ret_primary': y_ret_primary,
                    'X_train_meta': X_train_meta,
                    'y_train_meta': y_train_meta,
                    'y_ret_meta': y_ret_meta,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_ret_test': y_ret_test,
                    'X_raw': X_raw,
                    'active_features': active_features,
                    'is_classification': is_classification,
                    'reg_type': reg_type,
                    'l_params': l_params,
                    'indicator_window': ind_window,
                    'fwd_window': fwd_window,
                    'tickers': tickers,
                    'interval': interval,
                    'standardize': standardize_flag
                })
                ui.notification_show("Analysis complete!", type="success")
            
        except Exception as e:
            logger.log("Predictive", "ERROR", f"FATAL ERROR in run_analysis: {str(e)}")
            import traceback
            logger.log("Predictive", "ERROR", traceback.format_exc())
            ui.notification_show(f"Analysis failed: {str(e)}", type="error")

    @reactive.calc
    def labeling_result():
        # Trigger ONLY on button click or programmatic trigger
        input.btn_run_labeling()
        label_trigger.get()
        
        with reactive.isolate():
            # Get basics from UI inputs
            goal = input.analysis_goal()
            is_classification = (goal == "Directional")
            interval = input.interval()
            
            # Determine ticker
            if is_classification:
                ticker = input.selected_ticker()
            else:
                ts = list(input.selected_tickers())
                if not ts: return None
                ticker = ts[0]
                
            if not ticker or not interval: return None
        
            # Load data based on Bar Type
            b_type = input.bar_type()
            df = None
            if b_type == "Time Bars":
                df = manager.load_data(ticker, interval)
            else:
                eng_cache = engineered_data.get()
                target_key = "volume" if b_type == "Volume Bars" else "dollar"
                if eng_cache["ticker"] == ticker and eng_cache["interval"] == interval and eng_cache[target_key] is not None:
                    df = eng_cache[target_key]
                else:
                    ui.notification_show(f"Engineered data for {ticker} ({interval}) not found. Apply it first in the Data Engineering tab.", type="error")
                    return None

            if df is None or df.empty: return None
            
            # Consistently set index to datetime for plotting and labeling
            if 'open_time' in df.columns:
                df = df.set_index(pd.to_datetime(df['open_time']))
            
            # Calculate Labels
            try:
                if is_classification:
                    l_type = input.labeler_type()
                    prices = df['close']
                    if l_type == "Trend":
                        lobj = Labeler(amplitude_threshold=input.amp_th_bps(), max_inactive_period=input.max_inactive())
                        y_series = lobj.label(prices)['label']
                    elif l_type == "BoxRange":
                        lobj = TripleBarrierLabeler(vol_window=input.vol_window(), upper_mult=input.upper_mult(), lower_mult=input.lower_mult(), max_holding_period=input.max_holding())
                        y_series = lobj.label(prices)['label']
                    elif l_type == "Combine":
                        lobj = CombinedLabeler(
                            amplitude_threshold=input.amp_th_bps(), 
                            max_inactive_period=input.max_inactive(),
                            vol_window=input.vol_window(), 
                            upper_mult=input.upper_mult(), 
                            lower_mult=input.lower_mult(), 
                            max_holding=input.max_holding()
                        )
                        y_series = lobj.label(prices)['label']
                    elif l_type == "TailSet":
                        lobj = TailSetLabeler(ret_window=input.tail_window(), threshold=input.tail_threshold())
                        y_series = lobj.label(prices)['label']
                    else: # Regime
                        lobj = StationarityLabeler(window=input.stat_window(), vote_th=input.vote_th())
                        y_series, _ = lobj.label(prices)
                    
                    y_series.index = prices.index
                    y_series = y_series.shift(-1)
                else:
                    # Return Value (Forward Return)
                    prices_arr = pd.to_numeric(df['close'], errors='coerce').ffill().values
                    fwd_window = input.fwd_window()
                    y_series = pd.Series(np.log(prices_arr[1:] / prices_arr[:-1]), index=df.index[1:]).rolling(window=fwd_window).sum().shift(-fwd_window)
                
                y_series.index = df.index
                
                # Store in cache
                if is_classification:
                    cache_key = f"{ticker}_{interval}_{b_type}_{l_type}"
                    l_params_for_cache = {
                        'type': l_type,
                        'amp_th': input.amp_th_bps() if l_type in ["Trend", "Combine"] else None,
                        'max_inactive': input.max_inactive() if l_type in ["Trend", "Combine"] else None,
                        'vol_window': input.vol_window() if l_type in ["BoxRange", "Combine"] else None,
                        'upper_mult': input.upper_mult() if l_type in ["BoxRange", "Combine"] else None,
                        'lower_mult': input.lower_mult() if l_type in ["BoxRange", "Combine"] else None,
                        'max_holding': input.max_holding() if l_type in ["BoxRange", "Combine"] else None,
                        'stat_window': input.stat_window() if l_type == "Regime" else None,
                        'vote_th': input.vote_th() if l_type == "Regime" else None,
                        'tail_window': input.tail_window() if l_type == "TailSet" else None,
                        'tail_threshold': input.tail_threshold() if l_type == "TailSet" else None
                    }
                else:
                    cache_key = f"{ticker}_{interval}_{b_type}_regression_{fwd_window}"
                    l_params_for_cache = {"fwd_window": fwd_window}
                
                cache = labeled_data.get()
                cache[cache_key] = {
                    "df": df,
                    "y": y_series,
                    "ticker": ticker,
                    "interval": interval,
                    "bar_type": b_type,
                    "is_classification": is_classification,
                    "params": l_params_for_cache
                }
                labeled_data.set(cache)
                print(f"✓ Stored labels in cache with key: {cache_key}")
                
                return {"df": df, "y": y_series, "ticker": ticker, "is_classification": is_classification}
            except Exception as e:
                logger.log("Predictive", "ERROR", f"Labeling calculation error: {e}")
                return None

    @render_widget
    def label_chart():
        res = labeling_result()
        if res is None:
            return None

        df, y, ticker = res['df'], res['y'], res['ticker']

        res_ = results.get()
        if res_ is not None and ticker in res_['tickers']:
            test_df = res_['X_test']
            df = df[:-len(test_df)]  # show only training data if model exists

        df = df.copy()
        df.index = df.index.strftime('%Y-%m-%d %H:%M')

        if res['is_classification']:
            y_clean = y.dropna()
            df_plot = df
            # Convert labels to numeric for continuous color scale
            df_plot['label'] = y_clean.map({1: 1, 0: 0, -1: -1})

            fig = px.scatter(
                df_plot,
                x=df_plot.index,
                y="close",
                color="label",
                color_continuous_scale="Spectral_r",
                range_color=[-1, 1],
                size_max=7
            )

            # Customize the colorbar to show label names
            fig.update_traces(
                marker=dict(colorbar=dict(
                    title="Label",
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=["DOWN", "SIDEWAYS", "UP"]
                ))
            )

        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'],
                name='Price',
                line=dict(color='#9AA0A6', width=1.5)
            ))

        # Layout styling
        fig.update_layout(
            template="plotly_dark",
            height=450,
            width=965,
            margin=dict(t=40, b=60, l=30, r=30),
            xaxis_title=None,
            yaxis_title=None,
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)")
        )

        return fig

    @render.table
    def label_summary_table():
        res = labeling_result()
        if res is None: return None
        
        y = res['y'].dropna()
        if res['is_classification']:
            counts = y.value_counts().sort_index()
            mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}
            summary = pd.DataFrame({
                "Label": [mapping.get(k, k) for k in counts.index],
                "Count": counts.values,
                "Percentage": (counts.values / len(y) * 100).round(1)
            })
        if res['is_classification']:
            styled = (
                summary.style
                .hide_index()
                .format({"Count": "{:.0f}", "Percentage": "{:.1f}%"})
                .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
                .set_table_styles([
                    {"selector": "th", "props": [("color", "#000000"), ("background-color", "#FCC780"), ("border", "1px solid #1a4da3"), ("text-transform", "uppercase"), ("font-weight", "700")]},
                    {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #1a4da3")]},
                    {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "2px solid #1a4da3"), ("width", "100%")]}
                ])
            )
        else:
            styled = (
                summary.style
                .hide_index()
                .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
                .set_table_styles([
                    {"selector": "th", "props": [("color", "#000000"), ("background-color", "#FCC780"), ("border", "1px solid #1a4da3"), ("text-transform", "uppercase"), ("font-weight", "700")]},
                    {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #1a4da3")]},
                    {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "2px solid #1a4da3"), ("width", "100%")]}
                ])
            )
            
        return styled

    @render.ui
    def kpi_row():
        res = results.get()
        if res is None: return ui.div()
        
        try:
            X_test, y_test, model = res['X_test'], res['y_test'], res['model']
            # We compare Test against Primary Train
            X_train, y_train = res['X_train_primary'], res['y_train_primary']
            
            if len(X_test) == 0:
                return ui.div(ui.p("Test set is empty.", class_="text-warning"))
            
            preds_test = model.predict(X_test)
            preds_train = model.predict(X_train)
            is_classification = res['is_classification']

            if is_classification:
                acc_test = accuracy_score(y_test, preds_test)
                acc_train = accuracy_score(y_train, preds_train)
                return ui.layout_columns(
                    ui.value_box("Test Accuracy", f"{acc_test:.1%}", showcase=fa.icon_svg("bullseye")),
                    ui.value_box("Train Accuracy", f"{acc_train:.1%}", showcase=fa.icon_svg("graduation-cap")),
                    ui.value_box("Total Samples", len(y_train) + len(res['y_train_meta']) + len(y_test), showcase=fa.icon_svg("database"))
                )
            else:
                r2_test = r2_score(y_test, preds_test)
                r2_train = r2_score(y_train, preds_train)
                mse = mean_squared_error(y_test, preds_test)
                return ui.layout_columns(
                    ui.value_box("Test R²", f"{r2_test:.3f}", showcase=fa.icon_svg("chart-line")),
                    ui.value_box("Train R²", f"{r2_train:.3f}", showcase=fa.icon_svg("graduation-cap")),
                    ui.value_box("Test MSE", f"{mse:.5f}", showcase=fa.icon_svg("calculator"))
                )
        except Exception as e:
            return ui.div(f"Error in KPIs: {e}", class_="text-danger")

    @render.table
    def classification_report_table():
        res = results.get()
        if not res or not res['is_classification']: return None
        preds_test = res['model'].predict(res['X_test'])
        report = classification_report(res['y_test'], preds_test, labels=[-1, 0, 1], target_names=["DOWN", "SIDEWAYS", "UP"], output_dict=True)
        df = pd.DataFrame(report).transpose().round(3).reset_index()
        df.columns = ["Metric", "Precision", "Recall", "F1-Score", "Support"]
        styled = (
            df.style.hide_index()
            .format({
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1-Score": "{:.1%}",
                "Support": "{:.0f}"
            })
            .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("color", "#000000"), ("background-color", "#FCC780"), ("border", "1px solid #1a4da3"), ("text-transform", "uppercase"), ("font-weight", "700")]},
                {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #1a4da3")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "2px solid #1a4da3"), ("width", "100%")]}
            ])
        )
        return styled

    @render.ui
    async def live_predictions_table():
        res = results.get()
        if res is None: return None
        
        tickers = res['tickers']
        interval = res['interval']
        active_features = res['active_features']
        model = res['model']
        meta_model = res['meta_model']
        is_classification = res['is_classification']
        standardize = res['standardize']
        X_raw = res['X_raw']
        ind_window = res['indicator_window']
        
        prediction_rows = []
        
        with ui.Progress(min=0, max=len(tickers)) as p:
            p.set(message="Calculating live predictions...")
            for i, sym in enumerate(tickers):
                try:
                    df = manager.load_data(sym, interval)
                    if df is not None and len(df) >= max(ind_window, 100):
                        latest_price = float(df['close'].iloc[-1])
                        
                        # Calculate ALL indicators in one vectorized pass per ticker
                        all_inds = engine.calculate_all_indicators(df, window=ind_window, interval=interval)
                        latest_row = all_inds.iloc[-1]
                        
                        latest_vals = {}
                        for feat in active_features:
                            val = latest_row[feat] if feat in latest_row else np.nan
                            
                            if standardize and feat in X_raw.columns:
                                m, s = X_raw[feat].mean(), X_raw[feat].std()
                                val = (val - m) / s if s > 1e-9 else val
                            latest_vals[feat] = val
                        
                        if any(np.isnan(v) for v in latest_vals.values()):
                            continue
                            
                        X_pred = pd.DataFrame([latest_vals])
                        
                        signal = model.predict(X_pred)[0]
                        
                        # Meta-Sizing
                        meta_size = 1.0
                        if meta_model is not None:
                            X_m = pd.concat([X_pred, pd.Series([signal], index=X_pred.index, name='primary_pred')], axis=1)
                            probs = meta_model.predict_proba(X_m)[0]
                            idx_1 = np.where(meta_model.classes_ == 1)[0]
                            meta_sizing = probs[idx_1[0]] if len(idx_1) > 0 else (1.0 if meta_model.classes_[0] == 1 else 0.0)
                            meta_sizing = np.clip((meta_sizing * 2 - 1) * 2, 0, 1)

                        row = {"Ticker": sym, "Price": f"{latest_price:,.2f}", "Sizing": f"{meta_sizing:.1%}"}
                        if is_classification:
                            mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}
                            pred_val = mapping.get(signal, "N/A")
                            badge_class = "prediction-up" if signal == 1 else "prediction-down" if signal == -1 else ""
                            row.update({"Predicted": f'<span class="prediction-badge {badge_class}">{pred_val}</span>'})
                            
                            if hasattr(model, "predict_proba"):
                                confidence = np.max(model.predict_proba(X_pred)[0])
                                row["Confidence"] = f"{confidence:.1%}"
                        else:
                            row.update({"Predicted Ret": f"{signal:.4f}"})
                            row["Signal"] = "LONG" if signal > 0 else "SHORT"
                        
                        # Terminal styling for sizing
                        row["Sizing"] = f'<span style="color: #FCC780; font-weight: 700;">{meta_size:.1%}' + (' [!]' if meta_size > 0.8 else '') + '</span>'
                        
                        prediction_rows.append(row)
                except Exception as e:
                    pass
                
                p.set(i + 1)
                await asyncio.sleep(0.01)

        if not prediction_rows: return None
        df = pd.DataFrame(prediction_rows)
        # Use pandas styler to render HTML safely with terminal styling
        styled = (
            df.style
            .hide_index()
            .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("color", "#000000"), ("background-color", "#FCC780"), ("border", "1px solid #1a4da3"), ("text-transform", "uppercase"), ("font-weight", "700")]},
                {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #1a4da3")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "2px solid #1a4da3"), ("width", "100%")]}
            ])
        )
        return ui.HTML(styled.to_html(escape=False))


    from plotly.subplots import make_subplots

    @render_widget
    def pnl_plot():
        res = results.get()
        if res is None: return None
        
        meta_results = res.get('meta_results', {})
        
        # If we have sophisticated backtest results, use them
        if meta_results:
            raw_equity = meta_results['raw_equity']
            meta_equity = meta_results['meta_equity']
            raw_size = meta_results['raw_size']
            meta_size = meta_results['meta_size']
            buy_hold_equity = meta_results['buy_hold_equity']
            
            # Align indices 
            plot_df = pd.DataFrame({
                'Raw Equity': raw_equity,
                'Meta Equity': meta_equity,
                'BnH Equity': buy_hold_equity,
                'Raw Size': raw_size,
                'Meta Size': meta_size,
            }).fillna(method='ffill')
            
            # Ensure index is datetime for proper Plotly handling
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                plot_df.index = pd.to_datetime(plot_df.index)

            plot_df.index = plot_df.index.strftime('%Y-%m-%d %H:%M')

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4]
            )

            # Row 1: Equity Curves
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['Raw Equity'] / plot_df['Raw Equity'].iloc[0] - 1,
                name="Raw Signal PnL",
                line=dict(color="#9AA0A6", width=3)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['Meta Equity'] / plot_df['Meta Equity'].iloc[0] - 1,
                name="Meta Optimized PnL",
                line=dict(color="#00E5A8", width=3)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['BnH Equity'] / plot_df['BnH Equity'].iloc[0] - 1,
                name="Buy & Hold PnL",
                line=dict(color="#FFD700", width=2, dash='dot')
            ), row=1, col=1)

            # Row 2: Signals and Sizes from Backtest
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['Raw Size'],
                name="Raw Size",
                line=dict(color="#9AA0A6", width=1.5),
                opacity=0.5
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['Meta Size'],
                name="Meta Size",
                line=dict(color="#4cc9f0", width=1.5)
            ), row=2, col=1)

            # Layout update
            fig.update_layout(
                height=550,
                width=1500,
                template="plotly_dark",
                margin=dict(t=40, b=60, l=40, r=30),
                paper_bgcolor="#0b3d91",
                plot_bgcolor="#0b3d91",
                font=dict(family="Space Mono", color="white"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )

            # Update Y axes
            fig.update_yaxes(title_text="Cumulative PnL", row=1, col=1,
                            gridcolor="rgba(255,255,255,0.6)",
                            linecolor="white",
                            tickcolor="white",
                            zerolinecolor="white")

            fig.update_yaxes(title_text="Signal", row=2, col=1,
                            gridcolor="rgba(255,255,255,0.6)",
                            linecolor="white",
                            tickcolor="white",
                            zerolinecolor="white")

            # Update X axes (shared)
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.6)",
                            linecolor="white",
                            tickcolor="white",
                            zerolinecolor="white")

            return fig

        # Fallback to naive backtest
        model = res['model']
        X_test = res['X_test']
        y_ret_test = res['y_ret_test']
        meta_model = res['meta_model']
        is_classification = res['is_classification']
        
        p = model.predict(X_test)
        bt = pd.DataFrame({'Pred': p, 'Ret': y_ret_test}).sort_index()
        bt.index = bt.index.strftime('%Y-%m-%d %H:%M')
        
        meta_sizing = np.ones(len(bt))
        if meta_model is not None:
            X_meta = pd.concat([X_test, pd.Series(p, index=X_test.index, name='primary_pred')], axis=1)
            probs = meta_model.predict_proba(X_meta)
            idx_1 = np.where(meta_model.classes_ == 1)[0]
            if len(idx_1) > 0:
                meta_sizing = probs[:, idx_1[0]]
                meta_sizing = np.clip((meta_sizing * 2 - 1) * 2, 0, 1)

        
        bt['Size'] = meta_sizing
        bt['Size'] = np.clip(bt['Size'], -1, 1)
        bt['Signal'] = np.sign(bt['Pred']) if not is_classification else bt['Pred']

        if is_classification:
            bt['L_raw'] = np.where(bt['Pred'] == 1, bt['Ret'], 0)
            bt['S_raw'] = np.where(bt['Pred'] == -1, -bt['Ret'], 0)
        else:
            bt['L_raw'] = np.where(bt['Pred'] > 0, bt['Ret'], 0)
            bt['S_raw'] = np.where(bt['Pred'] < 0, -bt['Ret'], 0)
            
        bt['Both_raw'] = bt['L_raw'] + bt['S_raw']
        bt['Both_meta'] = bt['Both_raw'] * bt['Size']

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )

        perf_series = [
            ("Both_raw", "Raw Signal", "#1a1a1a", 3, None),       # dark gray
            ("Both_meta", "Meta Optimized", "#ff007f", 3, None),   # neon pink
            ("Ret", "Benchmark", "#00bfae", 2, "dot"),            # cyan/teal
        ]

        for col, name, color, width, dash in perf_series:
            fig.add_trace(
                go.Scatter(
                    x=bt.index,
                    y=np.exp(bt[col].cumsum()) - 1,
                    name=name,
                    line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width)
                ),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=bt.index,
                y=bt['Signal'],
                name="Signal",
                line=dict(color="#4cc9f0", width=1),
                opacity=0.8
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=bt.index,
                y=bt['Size'] * bt['Signal'],
                name="Size",
                line=dict(color="#72efdd", width=2)
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            width=1400,
            template="plotly_dark",
            margin=dict(t=40, b=60, l=40, r=30),
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5)
        )

        fig.update_yaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)

        return fig

    @render_widget
    def predicted_label_chart():
        res = results.get()
        if res is None:
            return None
            
        test_df = res['X_test']
        model = res['model']
        meta_model = res['meta_model']
        is_classification = res['is_classification']
        
        if not is_classification or len(test_df) == 0:
            return None
            
        primary_preds = model.predict(test_df)
        
        X_meta = pd.concat(
            [test_df, pd.Series(primary_preds, index=test_df.index, name='primary_pred')],
            axis=1
        )
        
        meta_probs = meta_model.predict_proba(X_meta)
        idx_1 = np.where(meta_model.classes_ == 1)[0]
        
        if len(idx_1) > 0:
            meta_sizing = meta_probs[:, idx_1[0]]
            meta_sizing = np.clip((2 * meta_sizing - 1) * 2, 0, 1)
            true_pred = primary_preds * meta_sizing
        else:
            true_pred = primary_preds.astype(float)
            
        ticker = res['tickers'][0]
        interval = res['interval']
        full_df = manager.load_data(ticker, interval)
        if full_df is None:
            return None
            
        if 'open_time' in full_df.columns:
            full_df = full_df.set_index(pd.to_datetime(full_df['open_time']))
            
        plot_df = full_df.loc[test_df.index].copy()
        plot_df['pred'] = true_pred
        plot_df.index = plot_df.index.strftime('%Y-%m-%d %H:%M')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['close'],
            name='Price',
            line=dict(color='#9AA0A6', width=1.5)
        ))
        
        fig = px.scatter(
            plot_df,
            x=plot_df.index,
            y='close',
            color='pred',
            color_continuous_scale='Spectral_r',
            range_color=[-1, 1],
            size_max=7
        )

        fig.update_traces(
            mode='markers',
            marker=dict(size=7),
            hovertemplate='Time: %{x}<br>Price: %{y}<br>Signal: %{marker.color:.2f}<extra></extra>'
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=550,
            width=1500,
            margin=dict(t=50, b=80, l=40, r=40),
            xaxis_title="Time",
            yaxis_title="Price",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
        
    @render_widget
    def shap_summary_primary():
        res = results.get()
        if res is None or 'shap_results' not in res or 'primary' not in res['shap_results']:
            return None
        
        shap_data = res['shap_results']['primary']
        sv = shap_data['values']
        
        if not hasattr(sv, 'values'):
            return None
            
        # Extract feature names and mean absolute SHAP values
        feature_names = sv.feature_names
        # For multi-class, sv might be (samples, features, classes)
        if len(sv.shape) == 3:
            # Aggregate across all classes or pick one? Usually mean abs is better for global
            vals = np.abs(sv.values).mean(axis=(0, 2))
        else:
            vals = np.abs(sv.values).mean(axis=0)
            
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({'Feature': feature_names, 'Importance': vals}).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            df_plot, x='Importance', y='Feature', orientation='h',
            template="plotly_dark",
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=400,
            width=710,
        )
        fig.update_traces(marker_color="#FCC780")
        
        return fig

    @render_widget
    def shap_summary_meta():
        res = results.get()
        if res is None or 'shap_results' not in res or 'meta' not in res['shap_results']:
            return None
        
        shap_data = res['shap_results']['meta']
        sv = shap_data['values']
        
        if not hasattr(sv, 'values'):
            return None
            
        feature_names = sv.feature_names
        if len(sv.shape) == 3:
            vals = np.abs(sv.values).mean(axis=(0, 2))
        else:
            vals = np.abs(sv.values).mean(axis=0)
            
        df_plot = pd.DataFrame({'Feature': feature_names, 'Importance': vals}).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            df_plot, x='Importance', y='Feature', orientation='h',
            template="plotly_dark"
        )
        
        fig.update_layout(
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=400,
            width=710,
        )
        fig.update_traces(marker_color="#FCC780")
        
        return fig

    @render_widget
    def shap_beeswarm_plot():
        res = results.get()
        if res is None or 'shap_results' not in res or 'primary' not in res['shap_results']:
            return None
            
        shap_data = res['shap_results']['primary']
        sv = shap_data['values']
        
        if not hasattr(sv, 'values'):
            return None
            
        # Get mean abs importance for feature ordering
        feature_names = sv.feature_names
        if len(sv.shape) == 3:
            vals = np.abs(sv.values).mean(axis=(0, 2))
            sv_vals = sv.values[:, :, 1 if sv.shape[2] > 1 else 0]
            sv_data = sv.data
        else:
            vals = np.abs(sv.values).mean(axis=0)
            sv_vals = sv.values
            sv_data = sv.data

        # Top 15 features for performance
        order = np.argsort(vals)[-15:]
        
        plot_rows = []
        for i in order:
            feat_name = feature_names[i]
            feat_vals = sv_vals[:, i]
            raw_vals = sv_data[:, i] if sv_data is not None else np.zeros_like(feat_vals)
            
        # Subsample if too many points for Plotly
        if len(sv_vals) > 500:
            idx = np.random.choice(len(sv_vals), 500, replace=False)
            sv_vals = sv_vals[idx]
            sv_data = sv_data[idx] if sv_data is not None else None

        plot_rows = []
        for i, feat_idx in enumerate(order):
            feat_name = feature_names[feat_idx]
            f_vals = sv_vals[:, feat_idx]
            r_vals = sv_data[:, feat_idx] if sv_data is not None else np.zeros_like(f_vals)
            
            # Simple manual jitter for swarm effect
            # We map each feature to its index and add noise
            for f_val, r_val in zip(f_vals, r_vals):
                plot_rows.append({
                    'Feature': feat_name,
                    'FeatureIdx': i,
                    'JitteredIdx': i + np.random.normal(0, 0.08),
                    'SHAP Value': f_val,
                    'Feature Value': r_val
                })
        
        df_p = pd.DataFrame(plot_rows)
        
        fig = px.scatter(
            df_p, x='SHAP Value', y='JitteredIdx', color='Feature Value',
            template="plotly_dark",
            color_continuous_scale='Spectral_r',
            labels={'JitteredIdx': 'Feature'},
            size_max=15
        )
        
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        
        fig.update_layout(
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.3)", 
                zerolinecolor="rgba(255, 255, 255, 0.5)",
                tickmode='array',
                tickvals=list(range(len(order))),
                ticktext=[feature_names[i] for i in order]
            ),
            margin=dict(l=10, r=10, t=30, b=50),
            height=550,
            width=710,
            coloraxis_colorbar=dict(title="Feature Value", len=0.5, yanchor="middle", y=0.5)
        )
        return fig

    @render_widget
    def shap_local_plot():
        res = results.get()
        if res is None or 'shap_results' not in res or 'primary' not in res['shap_results']:
            return None
            
        shap_data = res['shap_results']['primary']
        sv = shap_data['values']
        
        if not hasattr(sv, 'values'):
            return None
            
        # Waterfall plot for the LATEST bar in the test set
        idx = -1
        
        sv_row = sv[idx]
        feature_names = sv.feature_names
        
        if len(sv_row.shape) == 2: # (features, classes)
            class_idx = 1 if sv_row.shape[1] > 1 else 0
            row_vals = sv_row.values[:, class_idx]
            base_val = sv_row.base_values[class_idx]
        else:
            row_vals = sv_row.values
            base_val = sv_row.base_values

        # Sort by absolute value for a better waterfall
        order = np.argsort(np.abs(row_vals))[-10:] # Top 10 contributors
        
        final_names = [feature_names[i] for i in order]
        final_vals = [row_vals[i] for i in order]
        
        # Add 'Other' if needed? For now just top 10
        
        fig = go.Figure(go.Waterfall(
            name="SHAP", orientation="h",
            measure=["absolute"] + ["relative"] * (len(final_names) - 1) + ["total"],
            y=["[EXPECTED VALUE]"] + final_names[1:] + ["[PREDICTION]"],
            x=[base_val] + final_vals[1:] + [0], # Plotly calculates the total
            connector={"line":{"color":"#FCC780"}},
            increasing={"marker":{"color":"#00E5A8"}},
            decreasing={"marker":{"color":"#FF4D4D"}},
            totals={"marker":{"color":"#FCC780"}}
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=550,
            width=710,
            title=dict(text=f"Local Explanation for {res.get('ticker', 'Latest Symbol')}", font=dict(size=14))
        )
        
        return fig

    @render.table
    def strategy_metrics_table():
        res = results.get()
        if res is None:
            return None
        
        meta_results = res.get('meta_results', {})
        if meta_results:
            raw_m = meta_results['raw_metrics']
            meta_m = meta_results['meta_metrics']
            
            # For benchmark stats, we'll use naive calc on y_ret_test
            y_ret_test = res['y_ret_test']
            equity_bench = np.exp(y_ret_test.cumsum())
            bench_m = {
                "Strategy": "Benchmark",
                "Total Profits": equity_bench.iloc[-1] - 1,
                "Win Rate": (y_ret_test > 0).sum() / len(y_ret_test) if len(y_ret_test) > 0 else 0,
                "Trades": 0,
                "Sharpe": (y_ret_test.mean() / y_ret_test.std() * np.sqrt(MetricsEngine.get_annual_scaling(res['interval']))) if y_ret_test.std() > 1e-9 else 0,
                "Max Drawdown": ((equity_bench / equity_bench.cummax()) - 1).min()
            }

            df = pd.DataFrame([
                {
                    "Strategy": "Raw Signal",
                    "Total Profits": raw_m['total_return'],
                    "Win Rate": raw_m['win_rate'],
                    "Trades": raw_m['total_trades'],
                    "Sharpe": raw_m['sharpe_ratio'],
                    "Max Drawdown": raw_m['max_drawdown']
                },
                {
                    "Strategy": "Meta Optimized",
                    "Total Profits": meta_m['total_return'],
                    "Win Rate": meta_m['win_rate'],
                    "Trades": meta_m['total_trades'],
                    "Sharpe": meta_m['sharpe_ratio'],
                    "Max Drawdown": meta_m['max_drawdown']
                },
                bench_m
            ])
        else:
            # Fallback to naive metrics
            model = res['model']
            X_test = res['X_test']
            y_ret_test = res['y_ret_test']
            meta_model = res['meta_model']
            is_classification = res['is_classification']
            
            p = model.predict(X_test)
            bt = pd.DataFrame({'Pred': p, 'Ret': y_ret_test}, index=X_test.index).sort_index()
            
            meta_sizing = np.ones(len(bt))
            if meta_model is not None:
                X_meta = pd.concat([X_test, pd.Series(p, index=X_test.index, name='primary_pred')], axis=1)
                probs = meta_model.predict_proba(X_meta)
                idx_1 = np.where(meta_model.classes_ == 1)[0]
                if len(idx_1) > 0:
                    meta_sizing = probs[:, idx_1[0]]
                    meta_sizing = np.clip((2 * meta_sizing - 1) * 2, 0, 1)
            
            if is_classification:
                bt['Raw'] = np.where(bt['Pred'] == 1, bt['Ret'], 0) + np.where(bt['Pred'] == -1, -bt['Ret'], 0)
            else:
                bt['Raw'] = np.where(bt['Pred'] > 0, bt['Ret'], 0) + np.where(bt['Pred'] < 0, -bt['Ret'], 0)
            
            bt['Meta'] = bt['Raw'] * meta_sizing
            bt['Meta'] = np.clip(bt['Meta'], -1, 1)
            
            def get_stats(series, label):
                ret = series
                equity = np.exp(ret.cumsum())
                total_ret = equity.iloc[-1] - 1
                active = ret[ret != 0]
                win_rate = (active > 0).sum() / len(active) if len(active) > 0 else 0
                ann_factor = MetricsEngine.get_annual_scaling(res['interval'])
                sharpe = (ret.mean() * ann_factor / (ret.std() * np.sqrt(ann_factor))) if ret.std() > 1e-9 else 0
                max_dd = ((equity / equity.cummax()) - 1).min() * -1 * 100
                return {
                    "Strategy": label,
                    "Total Profits": total_ret,
                    "Win Rate": win_rate,
                    "Trades": len(active) if label != "Benchmark" else 0,
                    "Sharpe": sharpe,
                    "Max Drawdown": max_dd
                }
            
            df = pd.DataFrame([
                get_stats(bt['Raw'], "Raw Signal"),
                get_stats(bt['Meta'], "Meta Optimized"),
                get_stats(bt['Ret'], "Benchmark")
            ])
        
        df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
        
        styled = (
            df.style
            .hide_index()
            .format({
                "Total Profits": "{:+.2%}",
                "Win Rate": "{:.1%}",
                "Trades": "{:.0f}", 
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2f}%"
            })
            .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("color", "#000000"), ("background-color", "#FCC780"), ("border", "1px solid #1a4da3"), ("text-transform", "uppercase"), ("font-weight", "700")]},
                {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #1a4da3")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "2px solid #1a4da3"), ("width", "100%")]}
            ])
        )
        return styled

    @render_widget
    def cm_or_fit_plot():
        res = results.get()
        if res is None:
            return None

        X_test, y_test, model = res['X_test'], res['y_test'], res['model']
        preds = model.predict(X_test)

        if res['is_classification']:

            labels = ["DOWN", "SIDEWAYS", "UP"]
            cm = confusion_matrix(y_test, preds, labels=[-1, 0, 1])[::-1]

            fig = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels[::-1],
                    colorscale = [
                        [0.0, "#8B0000"],   # dark red
                        [0.25, "#FF073A"],  # neon red
                        [0.5, "#FFAA00"],   # neon orange/yellow
                        [0.75, "#00FF7F"],  # neon green
                        [1.0, "#00B200"]    # deep green
                    ],
                    showscale=True,
                    hoverinfo="skip"
                )
            )

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    fig.add_annotation(
                        x=labels[j],
                        y=labels[::-1][i],
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(size=8),
                        xref="x",
                        yref="y"
                    )

            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(t=18, b=2, l=2, r=2),
                paper_bgcolor="#0b3d91",
                plot_bgcolor="#0b3d91",
                font=dict(family="Space Mono", color="white"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
                dragmode=False
            )

            fig.update_xaxes(
                fixedrange=True,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=8)
            )

            fig.update_yaxes(
                fixedrange=True,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=8)
            )

            return fig

        pdf = pd.DataFrame({"Predicted": preds, "Actual": y_test}).sample(min(len(y_test), 500))

        fig = go.Figure(
            go.Scatter(
                x=pdf["Predicted"],
                y=pdf["Actual"],
                mode="markers",
                marker=dict(size=4, opacity=0.4),
                hoverinfo="skip"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(t=18, b=2, l=2, r=2),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            dragmode=False
        )

        fig.update_xaxes(fixedrange=True, tickfont=dict(size=8))
        fig.update_yaxes(fixedrange=True, tickfont=dict(size=8))

        return fig


    @render_widget
    def importance_plot():
        res = results.get()
        if res is None:
            return None

        model = res['model']
        feats = res['active_features']

        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            labels = [METRIC_LABELS.get(f, f) for f in feats]
            order = np.argsort(vals)[-8:]
            x_vals = vals[order]
            y_vals = [labels[i] for i in order]

        elif hasattr(model, "params"):
            coefs = model.params.drop("const", errors="ignore")
            if coefs.empty:
                return None
            order = np.argsort(np.abs(coefs))[-8:]
            x_vals = coefs.iloc[order]
            y_vals = [METRIC_LABELS.get(f, f) for f in coefs.index[order]]

        else:
            return None

        fig = px.bar(
            x=x_vals,
            y=y_vals,
            orientation="h",
            color=y_vals,  # numeric values mapped
            template="plotly_dark",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(t=18, b=2, l=2, r=2),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            dragmode=False
        )

        fig.update_xaxes(fixedrange=True, tickfont=dict(size=8))
        fig.update_yaxes(fixedrange=True, tickfont=dict(size=8))

        return fig

    @render.table
    def dist_train_table():
        res_label = labeling_result()
        res_ = results.get()
        
        # Prioritize labeling preview if it exists to ensure current chart matches table
        if res_label is not None and res_label['is_classification']:
            y_series = res_label['y']
            title = "Preview Distribution"
        elif res_ is not None and res_['is_classification']:
            y_series = res_['y_train_primary']
            title = "Train Distribution"
        else:
            return None

        mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}
        df = pd.Series(y_series).map(mapping)

        summary = (
            df.value_counts()
            .reindex(["DOWN", "SIDEWAYS", "UP"])
            .fillna(0)
            .astype(int)
            .rename_axis("Regime")
            .reset_index(name="Count")
        )

        total = summary["Count"].sum()
        summary["Pct"] = ((summary["Count"] / total) * 100).round(1)
        
        styled = (
            summary.style
            .hide_index()
            .format({"Count": "{:.0f}", "Pct": "{:.1f}%"})
            .set_properties(**{"font-size": "11px", "text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("font-size", "11px"), ("text-align", "center")]}])
        )
        return styled


    @render.table
    def dist_test_table():
        res_ = results.get()
        if res_ is None or not res_['is_classification']:
            return None

        mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}

        df = pd.Series(res_['y_test']).map(mapping)

        summary = (
            df.value_counts()
            .reindex(["DOWN", "SIDEWAYS", "UP"])
            .fillna(0)
            .astype(int)
            .rename_axis("Regime")
            .reset_index(name="Count")
        )

        total = summary["Count"].sum()
        summary["Pct"] = ((summary["Count"] / total) * 100).round(1)
        
        styled = (
            summary.style
            .hide_index()
            .format({"Count": "{:.0f}", "Pct": "{:.1f}%"})
            .set_properties(**{"font-size": "11px", "text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("font-size", "11px"), ("text-align", "center")]}])
        )
        return styled

    @render_widget
    def engineering_dist_plot():
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

        # Correct palette source
        base_colors = pc.diverging.Spectral_r
        colors = base_colors[:len(datasets)]

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
            height=390,
            width=965,
            margin=dict(t=40, b=60, l=30, r=30),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            xaxis_title="Log Returns",
            yaxis_title="Density",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
        )
        fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
        fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
        return fig

    @render_widget
    def engineering_cum_plot():
        res = engineering_result()
        if res is None: return None
        
        fig = go.Figure()
        colors = {"time": "#9AA0A6", "volume": "#00E5A8", "dollar": "#FFA726"}
        
        for k in ["time", "volume", "dollar"]:
            df = res.get(k)
            if df is not None and not df.empty and 'ret' in df.columns:
                cum_ret = df['ret'].dropna().cumsum()
                fig.add_trace(go.Scatter(
                    x=cum_ret.index,
                    y=cum_ret.values,
                    name=k.capitalize() + " Bars",
                    line=dict(color=colors[k], width=2)
                ))
                
        fig.update_layout(
            template="plotly_dark",
            height=400,
            width=965,
            margin=dict(t=40, b=60, l=30, r=30),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            xaxis_title=None,
            yaxis_title="Cumulative Log Return",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)"),
        )
        return fig

    @render.table
    def engineering_stats_table():
        res = engineering_result()
        if res is None: return None
        
        stats_list = []
        from scipy.stats import skew, kurtosis
        
        for k in ["time", "volume", "dollar"]:
            df = res.get(k)
            if df is not None and not df.empty and 'ret' in df.columns:
                rets = df['ret'].dropna()
                if len(rets) > 0:
                    s = rets.describe()
                    stats_list.append({
                        "Type": k.capitalize(),
                        "Count": int(s['count']),
                        "Mean": s['mean'],
                        "Std": s['std'],
                        "Min": s['min'],
                        "Max": s['max'],
                        "Skew": skew(rets),
                        "Kurtosis": kurtosis(rets)
                    })
                    
        if not stats_list: return None
        
        df_stats = pd.DataFrame(stats_list)

        # Round columns
        for col in ["Mean", "Std", "Min", "Max", "Skew", "Kurtosis"]:
            df_stats[col] = df_stats[col].map(lambda x: f"{x:.5f}")
            
        styled = (
            df_stats.style
            .hide_index()
            .set_properties(**{"font-size": "11px", "text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("font-size", "11px"), ("text-align", "center")]}])
        )
        return styled
