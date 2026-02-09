import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from ml_engine.modeling.feature_selection import FeatureSelector
from ml_engine.modeling.factory import ModelFactory
from ml_engine.predictive.predictor import Predictor
from ml_engine.labeling.labeler import Labeler, TripleBarrierLabeler, StationarityLabeler
from sklearn.linear_model import LogisticRegression
import os
import sys

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import APP_TITLE, APP_ICON, METRIC_LABELS, ALL_METRICS, DEFAULT_FETCH_INTERVALS, BENCHMARK_SYMBOL, BINANCE_URL
import streamlit.components.v1 as components

# Helper for Auto-Opening URLs
def open_url_js(url):
    js = f"""
    <script>
        window.open("{url}", "_blank").focus();
    </script>
    """
    components.html(js, height=0)

def format_metric(m_key):
    return METRIC_LABELS.get(m_key, m_key)

# --- Reusable Live Predictions Function ---
def show_live_predictions_section(selected_tickers, selected_interval, active_features, model, reg_type, is_classification, standardize, X_data, indicator_window, fwd_window, l_params, manager, BENCHMARK_SYMBOL, metrics_engine, Predictor, meta_model=None):
    st.write("**Live Predictions (Current Data)**")
    st.caption("Predictions based on the latest observation for each selected ticker.")
    
    prediction_rows = []
    skipped_tickers = []
    benchmark_returns = None
    if 'rel_strength_z' in active_features:
        b_df = manager.load_data(BENCHMARK_SYMBOL, selected_interval)
        if b_df is not None and not b_df.empty:
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()
    
    for ticker in selected_tickers:
        try:
            df_latest = manager.load_data(ticker, selected_interval)
            if df_latest is None or len(df_latest) < max(indicator_window, 100):
                skipped_tickers.append((ticker, "Insufficient data"))
                continue
            
            latest_price = float(df_latest['close'].iloc[-1])
            curr_regime = "N/A"
            if is_classification:
                current_labels = None
                if l_params['type'] == "Trend":
                    lobj = Labeler(amplitude_threshold=l_params['amp_th'], max_inactive_period=l_params['max_inactive'])
                    current_labels = lobj.label(df_latest['close'])['label']
                elif l_params['type'] == "BoxRange":
                    lobj = TripleBarrierLabeler(vol_window=l_params['vol_window'], upper_mult=l_params['upper_mult'], 
                                                lower_mult=l_params['lower_mult'], max_holding_period=l_params['max_holding'])
                    current_labels = lobj.label(df_latest['close'])['label']
                else:
                    lobj = StationarityLabeler(window=l_params['window'], vote_th=l_params['vote_th'])
                    current_labels, _ = lobj.label(df_latest['close'])
                mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}
                curr_regime = mapping.get(current_labels.iloc[-1], "N/A") if current_labels is not None else "N/A"

            latest_vals = {}
            for feat in active_features:
                try:
                    feat_series = metrics_engine.calculate_rolling_metric(df_latest, feat, window=indicator_window, benchmark_returns=benchmark_returns, interval=selected_interval)
                    val = feat_series.iloc[-1] if len(feat_series) > 0 else np.nan
                    if standardize and feat in X_data.columns:
                        m, s = X_data[feat].mean(), X_data[feat].std()
                        val = (val - m) / s if s > 1e-9 else val
                    latest_vals[feat] = val
                except: latest_vals[feat] = np.nan
            
            if any(np.isnan(v) for v in latest_vals.values()):
                skipped_tickers.append((ticker, "Data gap"))
                continue
                
            X_pred = pd.DataFrame([latest_vals])
            m_type = 'OLS'
            if "Random Forest" in reg_type: m_type = 'Random Forest Classifier' if is_classification else 'Random Forest Regressor'
            elif "XGB" in reg_type: m_type = 'XGB Classifier' if is_classification else 'XGB Regressor'

            predictor = Predictor(model, m_type, scaler=None) 
            result = predictor.predict(X_pred)
            
            # Meta-Sizing
            meta_size = 1.0
            if meta_model is not None:
                X_meta = X_pred.copy()
                # Ensure primary_pred type matches training (float for regression, int for classification)
                X_meta['primary_pred'] = result.get('signal', 0) if is_classification else result.get('predicted_value', 0.0)
                
                # Robust probability extraction for class 1 (profitability=True)
                probs = meta_model.predict_proba(X_meta)[0]
                idx_1 = np.where(meta_model.classes_ == 1)[0]
                meta_prob = probs[idx_1[0]] if len(idx_1) > 0 else (1.0 if meta_model.classes_[0] == 1 else 0.0)
                meta_size = meta_prob

            row = {"Ticker": ticker, "Price": latest_price, "Sizing": meta_size}
            if is_classification:
                row.update({"Current": curr_regime, "Predicted": result.get('signal_name', 'N/A'), "Confidence": result.get('confidence', 0)})
            else:
                row.update({"Predicted Ret": result.get('predicted_value', 0), "Signal": result.get('signal', 'N/A')})
            prediction_rows.append(row)
        except Exception as e:
            import traceback
            skipped_tickers.append((ticker, f"Error: {str(e)}"))
            # Optional: st.error(traceback.format_exc()) # Too noisy if multiple tickers fail
    
    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        sort_col = "Confidence" if is_classification else "Predicted Ret"
        pred_df = pred_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        fmt = {"Price": "{:,.2f}", "Sizing": "{:.2%}"}
        if is_classification: fmt["Confidence"] = "{:.1%}"
        else: fmt["Predicted Ret"] = "{:.4f}"
        st.dataframe(pred_df.style.format(fmt).background_gradient(subset=["Sizing"], cmap="Greens"), hide_index=True, use_container_width=True)

    if skipped_tickers:
        with st.expander(f"Skipped {len(skipped_tickers)} tickers", expanded=False):
            for t, r in skipped_tickers: st.caption(f"{t}: {r}")

st.set_page_config(page_title=f"{APP_TITLE} - Predictive", page_icon=APP_ICON, layout="wide")
st.markdown(f"## Predictive Analysis")

manager = DataManager()
metrics_engine = MetricsEngine()

# Sidebar settings
st.sidebar.markdown("### Data Selection")
inventory = manager.get_inventory()
if not inventory:
    st.warning("No data found. Please go to Data Loader first.")
    st.stop()

all_tickers = sorted(list(inventory.keys()))
default_idx = 0
for i, t in enumerate(all_tickers):
    if "BTC" in t.upper():
        default_idx = i
        break

is_directional = (st.sidebar.radio("Analysis Goal", ["Directional", "Return Value"], key="analysis_goal_radio") == "Directional")
if is_directional:
    selected_tickers = [st.sidebar.selectbox("Active Symbol", all_tickers, index=default_idx)]
    analysis_goal = "Directional"
else:
    with st.sidebar.expander("Select Symbols to Stack", expanded=False):
        select_all = st.checkbox("Select All Tickers", value=False)
        init_tickers = [all_tickers[default_idx]] if "BTC" in all_tickers[default_idx].upper() else (all_tickers[:2] if len(all_tickers) >= 2 else all_tickers)
        default_tickers = all_tickers if select_all else init_tickers
        selected_tickers = st.multiselect("Active Symbols", all_tickers, default=default_tickers)
    analysis_goal = "Return Value"

all_intervals = sorted(list(set(i for ivs in inventory.values() for i in ivs)))
selected_interval = st.sidebar.selectbox("Interval", all_intervals, index=all_intervals.index('1h') if '1h' in all_intervals else 0)

with st.sidebar.expander("Model Parameters", expanded=True):
    indicator_window = st.number_input("Indicator Lookback", min_value=2, max_value=500, value=60)
    fwd_window = st.number_input("Prediction Window", min_value=1, max_value=200, value=10) if not is_directional else 1
    min_samples = st.number_input("Min Samples per Symbol", min_value=10, max_value=1000, value=100)

    if is_directional:
        reg_type = st.selectbox("Model", ["Random Forest Classifier", "XGB Classifier"])
        st.markdown("---")
        labeler_type = st.selectbox("Labeler", ["Trend", "BoxRange", "Regime"])
        if labeler_type == "Trend":
            amp_th_bps = st.number_input("Amplitude Threshold (bps)", min_value=1, max_value=1000, value=200)
            max_inactive = st.number_input("Max Inactive Period", min_value=1, max_value=100, value=20)
        elif labeler_type == "BoxRange":
            vol_window = st.number_input("Volatility Window", min_value=10, max_value=300, value=20)
            upper_mult = st.number_input("Upper Barrier Mult", min_value=0.1, max_value=5.0, value=2.0)
            lower_mult = st.number_input("Lower Barrier Mult", min_value=0.1, max_value=5.0, value=2.0)
            max_holding = st.number_input("Max Holding Period", min_value=1, max_value=100, value=20)
        else:
            stat_window = st.number_input("Stationarity Window", min_value=10, max_value=500, value=20)
            vote_th = st.slider("Vote Threshold", 0.1, 1.0, 0.4, 0.05)
    else:
        reg_type = st.selectbox("Model", ["Linear (OLS)", "Random Forest Regressor", "XGB Regressor"])

    rf_max_depth = st.slider("Max Depth", 2, 50, 20) if "Random Forest" in reg_type or "XGB" in reg_type else None

with st.sidebar.expander("Validation & Preprocessing", expanded=False):
    test_size = st.slider("Test Set Ratio (%)", 0, 50, 30, 5) / 100.0
    standardize = st.toggle("Standardize Features", value=True)
    enable_meta = True # Mandatory meta-labeling
    remove_outliers = st.toggle("Remove Outliers", value=False, disabled = is_directional)
    outlier_threshold = st.slider("Outlier Threshold", 1.0, 10.0, 3.0, 0.5, disabled=not remove_outliers)

with st.sidebar.expander("Advanced & Visuals", expanded=False):
    log_x = st.toggle("Log Scale X", value=False, disabled = is_directional)
    log_y = st.toggle("Log Scale Y", value=False, disabled = is_directional)
    vif_threshold = st.slider("VIF Threshold", 2.0, 20.0, 5.0)

select_all_x = st.checkbox("Select All Predictors", value=False)
default_x = ALL_METRICS if select_all_x else (['price_zscore', 'rsi_norm', 'volatility', 'ewma', 'return_lag1', 'return_lag2', 'return_lag3', 'autocorr_5'] if 'price_zscore' in ALL_METRICS else [ALL_METRICS[0]])
x_features = st.multiselect("Predictors (X)", ALL_METRICS, default=default_x, format_func=format_metric)

@st.cache_data(show_spinner=False)
def get_stacked_data_cached(selected_tickers, selected_interval, x_features, indicator_window, fwd_window, min_samples, analysis_goal, l_params):
    all_rows = []
    stats = []
    benchmark_returns = None
    if 'rel_strength_z' in x_features:
        b_df = manager.load_data(BENCHMARK_SYMBOL, selected_interval)
        if b_df is not None and not b_df.empty:
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()

    for sym in selected_tickers:
        row_stat = {"Ticker": sym, "Status": "Processing"}
        df = manager.load_data(sym, selected_interval)
        if df is None or df.empty:
            row_stat["Status"] = "Load Failed"
            stats.append(row_stat)
            continue
            
        row_stat["Raw Rows"] = len(df)
        if len(df) <= max(indicator_window, fwd_window) + 10:
            row_stat["Status"] = "Too Short"
            stats.append(row_stat)
            continue
            
        if 'open_time' in df.columns:
            df = df.set_index(pd.to_datetime(df['open_time']))
        try:
            feats_data = {feat: metrics_engine.calculate_rolling_metric(df, feat, window=indicator_window, benchmark_returns=benchmark_returns, interval=selected_interval) for feat in x_features}
            if analysis_goal == "Directional":
                prices = df['close']
                if l_params['type'] == "Trend":
                    lobj = Labeler(amplitude_threshold=l_params['amp_th'], max_inactive_period=l_params['max_inactive'])
                    y_series = lobj.label(prices)['label']
                    y_series.index = prices.index # Restore index
                elif l_params['type'] == "BoxRange":
                    lobj = TripleBarrierLabeler(vol_window=l_params['vol_window'], upper_mult=l_params['upper_mult'], lower_mult=l_params['lower_mult'], max_holding_period=l_params['max_holding'])
                    y_series = lobj.label(prices)['label']
                    y_series.index = prices.index # Restore index
                else:
                    lobj = StationarityLabeler(window=l_params['window'], vote_th=l_params['vote_th'])
                    y_series, _ = lobj.label(prices)
            else:
                prices = pd.to_numeric(df['close'], errors='coerce').ffill().values
                y_series = pd.Series(np.log(prices[1:] / prices[:-1]), index=df.index[1:]).rolling(window=fwd_window).sum().shift(-fwd_window)
            
            temp_df = pd.DataFrame(feats_data)
            temp_df['Target_Y'] = y_series
            temp_df['raw_return'] = np.log(pd.to_numeric(df['close'], errors='coerce').ffill() / pd.to_numeric(df['close'], errors='coerce').ffill().shift(1)).shift(-1)
            
            row_stat["Before DropNA"] = len(temp_df)
            temp_df = temp_df.dropna()
            row_stat["After DropNA"] = len(temp_df)
            
            if len(temp_df) >= min_samples:
                all_rows.append(temp_df)
                row_stat["Status"] = "Success"
            else:
                row_stat["Status"] = f"Too few samples ({len(temp_df)} < {min_samples})"
        except Exception as e:
            row_stat["Status"] = f"Error: {str(e)[:50]}"
        stats.append(row_stat)
        
    return (pd.concat(all_rows) if all_rows else pd.DataFrame()), stats

l_params = None
if is_directional:
    if labeler_type == "Trend": l_params = {'type': "Trend", 'amp_th': amp_th_bps, 'max_inactive': max_inactive}
    elif labeler_type == "BoxRange": l_params = {'type': "BoxRange", 'vol_window': vol_window, 'upper_mult': upper_mult, 'lower_mult': lower_mult, 'max_holding': max_holding}
    else: l_params = {'type': "Stationarity", 'window': stat_window, 'vote_th': vote_th}

compiled_df, diag_stats = get_stacked_data_cached(selected_tickers, selected_interval, x_features, indicator_window, fwd_window, min_samples, analysis_goal, l_params)

if compiled_df.empty:
    st.error("No valid data points found.")
    with st.expander("Diagnostics: Why is data missing?", expanded=True):
        st.dataframe(pd.DataFrame(diag_stats), use_container_width=True, hide_index=True)
    st.stop()

with st.expander("Observations", expanded=False):
    edited_df = st.data_editor(compiled_df, num_rows="dynamic", use_container_width=True, hide_index=True)

if st.button("Run Analysis", type="primary"):
    final_df = edited_df.sort_index()
    if remove_outliers:
        z = (final_df['Target_Y'] - final_df['Target_Y'].mean()).abs() / final_df['Target_Y'].std()
        final_df = final_df[z <= outlier_threshold]
    
    if len(final_df) < 20:
        st.error("Too few points.")
        st.stop()
        
    X_data = final_df[x_features].copy()
    X_ref = X_data.copy() # Keep reference for live standardization
    Y_data = final_df['Target_Y'].copy()
    is_classification = (analysis_goal == "Directional")

    # Clean
    combined = pd.concat([X_data, Y_data, final_df['raw_return']], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    X_data, Y_data, Y_ret = combined[x_features], combined['Target_Y'], combined['raw_return']
    X_ref = combined[x_features].copy()

    if standardize:
        for f in x_features:
            m, s = X_data[f].mean(), X_data[f].std()
            if s > 1e-9: X_data[f] = (X_data[f] - m) / s
                
    active_features = FeatureSelector.apply_vif_filter(X_data, threshold=vif_threshold) if len(x_features) > 1 else x_features
    # Include Y_ret in the split to keep it aligned for meta-labeling
    X_train, X_test, y_train, y_test, y_ret_train, y_ret_test = train_test_split(X_data[active_features], Y_data, Y_ret, test_size=test_size, shuffle=False)
    
    try:
        if "Random Forest" in reg_type: model = ModelFactory.create_model(reg_type, n_estimators=200, max_depth=rf_max_depth)
        elif "XGB" in reg_type: model = ModelFactory.create_model(reg_type, n_estimators=200, max_depth=rf_max_depth)
        else:
            model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        _ = model.fit(X_train, y_train) if not hasattr(model, 'rsquared') else None
    except Exception as e:
        st.error(f"Fit failed: {e}")
        st.stop()

    meta_model = None
    if enable_meta:
        p_train = model.predict(X_train if not hasattr(model, 'rsquared') else sm.add_constant(X_train))
        
        if is_classification:
            y_meta = (p_train * y_ret_train > 0.00001).astype(int)
        else:
            y_meta = (np.sign(p_train) * np.sign(y_ret_train) > 0.00001).astype(int)
        
        meta_features = pd.concat([X_train, pd.Series(p_train, index=X_train.index, name='primary_pred')], axis=1)
        
        meta_model = LogisticRegression(max_iter=1000)
        _ = meta_model.fit(meta_features, y_meta)
        
        m_preds = meta_model.predict(meta_features)
        meta_acc = accuracy_score(y_meta, m_preds)

    # --- RESULTS ---
    st.markdown(f"#### Results (N={len(final_df)} | {100-test_size*100:.0f}% Train / {test_size*100:.0f}% Test)")
    t_perf, t_vis, t_back = st.tabs(["Performance", "Visuals", "Backtest"])
    
    with t_perf:
        m1, m2, m3, m4 = st.columns(4)
        if is_classification:
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            tr_acc = accuracy_score(y_train, train_preds)
            te_acc = accuracy_score(y_test, test_preds)
            m1.metric("Acc (Train)", f"{tr_acc:.1%}")
            m2.metric("Acc (Test)", f"{te_acc:.1%}")
            if hasattr(model, 'feature_importances_'):
                m3.metric("Top Feature", format_metric(active_features[np.argmax(model.feature_importances_)]))
        else:
            train_preds = model.predict(X_train if not hasattr(model, 'rsquared') else sm.add_constant(X_train))
            test_preds = model.predict(X_test if not hasattr(model, 'rsquared') else sm.add_constant(X_test))
            tr_r2 = r2_score(y_train, train_preds)
            te_r2 = r2_score(y_test, test_preds)
            m1.metric("R2 (Train)", f"{tr_r2:.3f}")
            m2.metric("R2 (Test)", f"{te_r2:.3f}")
            m3.metric("MSE", f"{mean_squared_error(y_test, test_preds):.5f}")
        m4.metric("Total N", len(final_df))
        
        st.write("---")
        show_live_predictions_section(selected_tickers, selected_interval, active_features, model, reg_type, is_classification, standardize, X_ref, indicator_window, fwd_window, l_params, manager, BENCHMARK_SYMBOL, metrics_engine, Predictor, meta_model)
        st.write("---")
        
        if is_classification:
            from sklearn.metrics import classification_report
            st.write("**Classification Report (Test Set)**")
            report = classification_report(y_test, test_preds, labels=[-1, 0, 1], target_names=["DOWN", "SIDEWAYS", "UP"], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"), use_container_width=True)

    with t_vis:
        v1, v2 = st.columns(2)
        with v1:
            if is_classification:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, test_preds, labels=[-1, 0, 1])
                fig_cm = ff.create_annotated_heatmap(z=cm[::-1], x=["DOWN", "SIDEWAYS", "UP"], y=["UP", "SIDEWAYS", "DOWN"], colorscale='Viridis', showscale=True)
                fig_cm.update_layout(title="Confusion Matrix", height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                for name, p, a in [('Train', train_preds, y_train), ('Test', test_preds, y_test)]:
                    pdf = pd.DataFrame({'P': p, 'A': a})
                    fig = px.scatter(pdf.sample(min(len(pdf), 2000)), x='P', y='A', opacity=0.3, title=f"{name} Fit")
                    fig.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=0))
                    st.plotly_chart(fig, use_container_width=True)
        with v2:
            if hasattr(model, 'feature_importances_'):
                imps = model.feature_importances_
                idx = np.argsort(imps)
                fig_imp = px.bar(x=imps[idx], y=[format_metric(active_features[i]) for i in idx], orientation='h', title="Feature Importance")
            else:
                coefs = model.params.drop('const', errors='ignore')
                idx = np.argsort(np.abs(coefs))
                fig_imp = px.bar(x=coefs.iloc[idx], y=[format_metric(f) for f in coefs.index[idx]], orientation='h', title="Feature Coefficients")
            fig_imp.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        # Distributions Row
        mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}
        d_col1, d_col2 = st.columns(2)
        if is_classification:
            with d_col1:
                train_dist = pd.DataFrame({'Label': y_train})
                train_dist['Regime'] = train_dist['Label'].map(mapping)
                fig_tr = px.histogram(train_dist, x='Regime', title="Label Dist (Train)", category_orders={"Regime": ["DOWN", "SIDEWAYS", "UP"]}, height=250)
                fig_tr.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_tr, use_container_width=True)
            with d_col2:
                test_dist = pd.DataFrame({'Label': y_test})
                test_dist['Regime'] = test_dist['Label'].map(mapping)
                fig_te = px.histogram(test_dist, x='Regime', title="Label Dist (Test)", category_orders={"Regime": ["DOWN", "SIDEWAYS", "UP"]}, height=250)
                fig_te.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_te, use_container_width=True)

    with t_back:
        p = model.predict(X_test if not hasattr(model, 'rsquared') else sm.add_constant(X_test))
        bt = pd.DataFrame({'Pred': p, 'Ret': final_df.loc[X_test.index, 'raw_return']}, index=X_test.index).sort_index()
        
        # Apply Meta-Sizing to Backtest if enabled
        meta_sizing_series = None
        if enable_meta and meta_model is not None:
            # Prepare meta-features for the entire test set
            X_meta_test = X_test.copy()
            X_meta_test['primary_pred'] = p
            
            # Vectorized meta-probability extraction
            meta_probs = meta_model.predict_proba(X_meta_test)
            idx_1 = np.where(meta_model.classes_ == 1)[0]
            if len(idx_1) > 0:
                meta_probs_1 = meta_probs[:, idx_1[0]]
            else:
                meta_probs_1 = np.ones(len(X_meta_test)) if meta_model.classes_[0] == 1 else np.zeros(len(X_meta_test))
            
            meta_sizing_series = meta_probs_1

        if is_classification:
            bt['L_raw'] = np.where(bt['Pred'] == 1, bt['Ret'], 0)
            bt['S_raw'] = np.where(bt['Pred'] == -1, -bt['Ret'], 0)
        else:
            bt['L_raw'] = np.where(bt['Pred'] > 0, bt['Ret'], 0)
            bt['S_raw'] = np.where(bt['Pred'] < 0, -bt['Ret'], 0)
        
        bt['Both_raw'] = bt['L_raw'] + bt['S_raw']
        
        # Meta-Adjusted
        bt['L_meta'] = bt['L_raw'] * meta_sizing_series
        bt['S_meta'] = bt['S_raw'] * meta_sizing_series
        bt['Both_meta'] = bt['L_meta'] + bt['S_meta']
            
        fig_pnl = go.Figure()

        fig_pnl.add_trace(go.Scatter(
            x=bt.index,
            y=np.exp(bt['Both_raw'].cumsum())-1,
            name='Raw Signal (L+S)',
            line=dict(color='#9AA0A6', width=3)
        ))

        fig_pnl.add_trace(go.Scatter(
            x=bt.index,
            y=np.exp(bt['L_meta'].cumsum())-1,
            name='Meta Long Only',
            line=dict(color='#7C4DFF', width=2, dash='dash'),
            visible='legendonly'
        ))

        fig_pnl.add_trace(go.Scatter(
            x=bt.index,
            y=np.exp(bt['S_meta'].cumsum())-1,
            name='Meta Short Only',
            line=dict(color='#FF4D8D', width=2, dash='dash'),
            visible='legendonly'
        ))

        fig_pnl.add_trace(go.Scatter(
            x=bt.index,
            y=np.exp(bt['Both_meta'].cumsum())-1,
            name='Meta Long & Short',
            line=dict(color='#00E5A8', width=3)
        ))

        fig_pnl.add_trace(go.Scatter(
            x=bt.index,
            y=np.exp(bt['Ret'].cumsum())-1,
            name='Benchmark',
            line=dict(color='#FFA726', width=2, dash='dot')
        ))

        fig_pnl.update_layout(
            height=400,
            margin=dict(t=30, b=0, l=0, r=0),
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", y=1.1),
            template="plotly_dark"
        )

        st.plotly_chart(fig_pnl, use_container_width=True)

        p_stats = []

        def perf_metrics(series):
            if series.empty:
                return 0,0,0,0
            ret = series
            equity = np.exp(ret.cumsum())
            total_ret = equity.iloc[-1] - 1
            trades = ret[ret != 0]
            wins = trades[trades > 0]
            losses = trades[trades < 0]
            profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
            downside = ret[ret < 0]
            sortino = (ret.mean() / downside.std()) * np.sqrt(252) if downside.std() != 0 else 0
            roll_max = equity.cummax()
            dd = (equity / roll_max) - 1
            max_dd = dd.min()
            return total_ret, profit_factor, sharpe, sortino, max_dd

        for label, col in [("Raw Signal", "Both_raw"), ("Meta Optimized", "Both_meta")]:
            active = bt[bt[col] != 0]
            win_rate = (active[col] > 0).sum() / len(active) if not active.empty else 0
            
            total_ret, pf, sharpe, sortino, max_dd = perf_metrics(bt[col])
            
            p_stats.append({
                "Strategy": label,
                "Total Ret": f"{total_ret:+.2%}",
                "Win Rate": f"{win_rate:.1%}",
                "Trades": len(active),
                "Profit Factor": f"{pf:.2f}" if not np.isnan(pf) else "NA",
                "Sharpe": f"{sharpe:.2f}",
                "Sortino": f"{sortino:.2f}",
                "Max DD": f"{max_dd:.2%}"
            })

        st.dataframe(pd.DataFrame(p_stats), hide_index=True, use_container_width=True)
