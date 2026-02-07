import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
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

st.set_page_config(page_title=f"{APP_TITLE} - Regression", page_icon=APP_ICON, layout="wide")

st.markdown(f"## Predictive Regression Analysis")

# Initialize
manager = DataManager()
metrics_engine = MetricsEngine()

def format_metric(m_key):
    return METRIC_LABELS.get(m_key, m_key)

def apply_vif_filter(X_input, threshold=10.0):
    """Iteratively remove features with VIF > threshold"""
    if X_input.shape[1] <= 1:
        return X_input.columns.tolist()
        
    cols = X_input.columns.tolist()
    while True:
        if len(cols) <= 1:
            break
            
        # Add constant for VIF calculation
        X_vif = sm.add_constant(X_input[cols])
        vifs = []
        for i in range(X_vif.shape[1]):
            col_name = X_vif.columns[i]
            if col_name == 'const':
                vifs.append(0) # We don't drop the constant
                continue
            try:
                # variance_inflation_factor returns inf if perfect collinearity
                v = variance_inflation_factor(X_vif.values, i)
            except Exception as e:
                # If calculation fails (e.g. singular matrix), treat as extremely high VIF
                v = 999.0
            vifs.append(v)
            
        vif_series = pd.Series(vifs, index=X_vif.columns)
        vif_no_const = vif_series.drop('const', errors='ignore')
        
        max_vif = vif_no_const.max()
        if max_vif > threshold or np.isnan(max_vif) or np.isinf(max_vif):
            exclude_col = vif_no_const.idxmax()
            cols.remove(exclude_col)
        else:
            break
    return cols

# Sidebar settings
st.sidebar.markdown("### Data Selection")
inventory = manager.get_inventory()
if not inventory:
    st.warning("No data found. Please go to Data Loader first.")
    st.stop()

# 1. Ticker Selection (Minimized Area)
all_tickers = sorted(list(inventory.keys()))
with st.sidebar.expander("Select Symbols to Stack", expanded=False):
    select_all = st.checkbox("Select All Tickers", value=False)
    default_tickers = all_tickers if select_all else (all_tickers[:2] if len(all_tickers) >= 2 else all_tickers)
    selected_tickers = st.multiselect("Active Symbols", all_tickers, default=default_tickers)

# 2. Timeframe
all_intervals = set()
for intervals in inventory.values():
    for i in intervals:
        all_intervals.add(i)
all_intervals = sorted(list(all_intervals))
selected_interval = st.sidebar.selectbox("Interval", all_intervals, index=all_intervals.index('1h') if '1h' in all_intervals else 0)

# 3. Targets and Windows
st.sidebar.markdown("### Analysis Parameters")
indicator_window = st.sidebar.number_input("Indicator Lookback", min_value=2, max_value=500, value=60)
fwd_window = st.sidebar.number_input("Prediction Window", min_value=1, max_value=200, value=10)
min_samples = st.sidebar.number_input("Min Samples per Symbol", min_value=10, max_value=1000, value=100)

st.sidebar.markdown("### Preprocessing")
standardize = st.sidebar.toggle("Standardize Features (Z-Score)", value=True, help="Normalize predictors to mean=0, std=1. Recommended for multivariate regression.")
remove_outliers = st.sidebar.toggle("Remove Outliers (Y)", value=False, help="Prune extreme 'fat-tail' returns from the target vector.")
outlier_threshold = st.sidebar.slider("Outlier Threshold (Std Dev)", 1.0, 10.0, 3.0, 0.5, disabled=not remove_outliers)

st.sidebar.markdown("### Model Validation")
test_size = st.sidebar.slider("Test Set Ratio (%)", 0, 50, 20, 5) / 100.0
st.sidebar.caption("Percentage of data kept separate for testing.")

st.sidebar.markdown("### Visualization")
log_x = st.sidebar.toggle("Log Scale X", value=False)
log_y = st.sidebar.toggle("Log Scale Y", value=False)

st.sidebar.markdown("### Model Selection")
analysis_goal = st.sidebar.radio("Analysis Goal", ["Return Value", "Directional (Win/Loss)"], help="Directional models the probability of a positive return.")

if analysis_goal == "Directional (Win/Loss)":
    reg_type = st.sidebar.selectbox("Model", ["Logistic (Logit)", "Random Forest Classifier"], help="Handles categorical target (0/1).")
else:
    reg_type = st.sidebar.selectbox("Model", ["Linear (OLS)", "Random Forest Regressor"], help="Handles continuous target (returns).")

rf_max_depth = None
if "Random Forest" in reg_type:
    rf_max_depth = st.sidebar.slider("RF: Max Depth", 2, 20, 5, help="Limit tree depth to prevent overfitting. Lower = simpler model.")

# --- Main Logic ---
select_all_x = st.checkbox("Select All Predictors", value=False)
default_x = ALL_METRICS if select_all_x else (['price_zscore', 'rsi_norm'] if 'price_zscore' in ALL_METRICS else [ALL_METRICS[0]])

x_features = st.multiselect(
    "Predictors (X)", 
    ALL_METRICS, 
    default=default_x, 
    format_func=format_metric,
    key="x_features_multiselect"
)

if len(x_features) > 1:
    st.sidebar.markdown("### Feature Selection")
    vif_threshold = st.sidebar.slider("VIF Multicollinearity Threshold", min_value=2.0, max_value=20.0, value=10.0, step=1.0, help="Features with Variance Inflation Factor above this will be removed. 5-10 is standard.")
else:
    vif_threshold = 999.0

if not selected_tickers:
    st.warning("Select at least one ticker.")
    st.stop()

# --- Data Collection ---
@st.cache_data(show_spinner="Fetching data from cache...")
def get_stacked_data(selected_tickers, selected_interval, x_features, indicator_window, fwd_window, min_samples):
    all_rows = []
    
    # Pre-calculate benchmark returns if needed
    benchmark_returns = None
    if 'rel_strength_z' in x_features:
        b_df = manager.load_data(BENCHMARK_SYMBOL, selected_interval)
        if b_df is not None and not b_df.empty:
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()

    for sym in selected_tickers:
        df = manager.load_data(sym, selected_interval)
        if df is not None and len(df) > max(indicator_window, fwd_window) + 10:
            try:
                # 1. Calculate all requested features
                feats_data = {}
                for feat in x_features:
                    s = metrics_engine.calculate_rolling_metric(
                        df, 
                        feat, 
                        window=indicator_window, 
                        benchmark_returns=benchmark_returns,
                        interval=selected_interval
                    )
                    feats_data[feat] = s
                
                # 2. Calculate Future Returns (Y)
                prices = pd.to_numeric(df['close'], errors='coerce').ffill().values
                log_rets = np.log(prices[1:] / prices[:-1])
                log_rets_s = pd.Series(log_rets, index=df.index[1:])
                y_series = log_rets_s.rolling(window=fwd_window).sum().shift(-fwd_window)
                
                # 3. Merge and Align
                temp_df = pd.DataFrame(feats_data)
                temp_df['Target_Y'] = y_series
                temp_df['Symbol'] = sym
                temp_df = temp_df.dropna()
                
                if len(temp_df) >= min_samples:
                    all_rows.append(temp_df)
            except Exception as e:
                continue
                
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

# 1. Compile Data
compiled_df = get_stacked_data(selected_tickers, selected_interval, x_features, indicator_window, fwd_window, min_samples)

if compiled_df.empty:
    st.error("No valid data points found with current settings.")
    st.stop()

# 2. Observation Table (Data Editor)
st.markdown("### Observations & Manual Filtering")

# Pre-sort observation table by Absolute Return to help identify outliers
if 'Target_Y' in compiled_df.columns:
    compiled_df['Abs_Return'] = compiled_df['Target_Y'].abs()
    compiled_df = compiled_df.sort_values('Abs_Return', ascending=False).drop(columns=['Abs_Return'])

# Add Z-Score column for outlier highlighting if requested
if remove_outliers:
    m_y = compiled_df['Target_Y'].mean()
    s_y = compiled_df['Target_Y'].std()
    compiled_df['Y_ZScore'] = (compiled_df['Target_Y'] - m_y) / (s_y if s_y > 1e-9 else 1.0)
    
st.info("View compiled data below. Use checkboxes to select/delete rows to prune specific points.")
edited_df = st.data_editor(
    compiled_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Target_Y": st.column_config.NumberColumn("Target (Y)", format="%.4f"),
        "Y_ZScore": st.column_config.NumberColumn("Y Z-Score", format="%.2f") if remove_outliers else None
    }
)

# 3. Run Analysis Button
if st.button("Run Regression Analysis", type="primary"):
    # Pre-processing from sidebar
    final_df = edited_df.copy()
    
    # Automatic Outlier Removal
    if remove_outliers:
        m_y = final_df['Target_Y'].mean()
        s_y = final_df['Target_Y'].std()
        if s_y > 1e-9:
            z_scores = (final_df['Target_Y'] - m_y).abs() / s_y
            final_df = final_df[z_scores <= outlier_threshold]
            st.caption(f"Auto-pruned {len(edited_df) - len(final_df)} further outliers.")

    if len(final_df) < 20:
        st.error("Insufficient data points after filtering.")
        st.stop()
        
    X_data = final_df[x_features].copy()
    Y_data = final_df['Target_Y'].copy()
    
    # --- DATA CLEANING BEFORE VIF ---
    # We must remove NaNs/Infs from X and Y before running VIF or it fails
    is_classification = (analysis_goal == "Directional (Win/Loss)")
    if is_classification:
        Y_data = (Y_data > 0).astype(int)
        
    combined_clean = pd.concat([X_data, Y_data], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(combined_clean) < 20:
        st.error("Insufficient valid data points (NaNs/Infs pruned).")
        st.stop()
        
    X_data = combined_clean[x_features]
    Y_data = combined_clean['Target_Y']

    # Standardization (if requested)
    if standardize:
        for feat in x_features:
            m = X_data[feat].mean()
            s = X_data[feat].std()
            if s > 1e-9:
                X_data[feat] = (X_data[feat] - m) / s
                
    # --- VIF FILTERING ---
    if len(x_features) > 1 and vif_threshold < 20.0:
        with st.status("Checking Multicollinearity (VIF)..."):
            active_features = apply_vif_filter(X_data, threshold=vif_threshold)
            dropped = set(x_features) - set(active_features)
            if dropped:
                st.warning(f"Dropped due to high VIF: {', '.join([format_metric(f) for f in dropped])}")
            X_data = X_data[active_features]
    else:
        active_features = x_features

    # Final feature set after VIF
    is_rf = "Random Forest" in reg_type
    is_logit = (reg_type == "Logistic (Logit)")
    
    # --- TRAIN/TEST SPLIT ---
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X_data, X_data, Y_data, Y_data
        
    try:
        if is_logit:
            X_train_const = sm.add_constant(X_train)
            model = sm.Logit(y_train, X_train_const).fit(disp=0)
        elif reg_type == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=100, max_depth=rf_max_depth, random_state=42)
            model.fit(X_train, y_train)
        elif reg_type == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, max_depth=rf_max_depth, random_state=42)
            model.fit(X_train, y_train)
        else:
            X_train_const = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_train_const).fit()
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.stop()

    # --- VISUALIZATION ---
    if len(active_features) == 1:
        # UNIVARIATE
        feat = active_features[0]
        # Show both in one plot but distinguish markers
        train_plot = pd.DataFrame({'X': X_train[feat], 'Y': y_train, 'Set': 'Train'})
        test_plot = pd.DataFrame({'X': X_test[feat], 'Y': y_test, 'Set': 'Test'})
        plot_df = pd.concat([train_plot, test_plot])
        
        if len(plot_df) > 5000:
            plot_df = plot_df.sample(5000)
            
        fig = px.scatter(
            plot_df, x='X', y='Y', color='Set',
            opacity=0.4,
            log_x=log_x, log_y=log_y,
            labels={'X': format_metric(feat) + (" (Z-Score)" if standardize else ""), 'Y': "Outcome" if is_classification else "Future Return"},
            title=f"{reg_type}: {format_metric(feat)} Validation",
            color_discrete_map={'Train': '#636EFA', 'Test': '#EF553B'}
        )
        
        # Add Fit Line (based on training model)
        grid_x = np.linspace(plot_df['X'].min(), plot_df['X'].max(), 100)
        if is_rf:
            grid_y = model.predict(grid_x.reshape(-1, 1))
        else:
            grid_df = pd.DataFrame({'const': 1.0, feat: grid_x})
            grid_y = model.predict(grid_df)
        
        fig.add_trace(go.Scatter(
            x=grid_x, y=grid_y, 
            mode='lines', 
            name='Model Fit (Train)', 
            line=dict(color='yellow', width=3)
        ))
        
        fig.update_xaxes(autorange=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # MULTIVARIATE
        if is_classification:
            # Predictions for both sets
            if is_rf:
                train_probs = model.predict_proba(X_train)[:, 1]
                test_probs = model.predict_proba(X_test)[:, 1]
            else:
                X_train_const = sm.add_constant(X_train, has_constant='add')
                X_test_const = sm.add_constant(X_test, has_constant='add')
                train_probs = model.predict(X_train_const)
                test_probs = model.predict(X_test_const)
                
            c1, c2 = st.columns(2)
            with c1:
                # Distribution (Test Set only or colored by set)
                dist_df = pd.concat([
                    pd.DataFrame({'Prob': train_probs, 'Actual': y_train, 'Set': 'Train'}),
                    pd.DataFrame({'Prob': test_probs, 'Actual': y_test, 'Set': 'Test'})
                ])
                fig = px.histogram(
                    dist_df, x='Prob', color='Actual', facet_col='Set',
                    barmode='overlay', nbins=50,
                    color_discrete_map={0: 'red', 1: 'green'},
                    title=f"Probability Distributions (Train vs Test)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                # Combined ROC Curve
                fig_roc = go.Figure()
                # Train
                fpr_tr, tpr_tr, _ = roc_curve(y_train, train_probs)
                auc_tr = auc(fpr_tr, tpr_tr)
                fig_roc.add_trace(go.Scatter(x=fpr_tr, y=tpr_tr, name=f'Train (AUC: {auc_tr:.3f})', line=dict(color='blue', width=2)))
                # Test
                fpr_ts, tpr_ts, _ = roc_curve(y_test, test_probs)
                auc_ts = auc(fpr_ts, tpr_ts)
                fig_roc.add_trace(go.Scatter(x=fpr_ts, y=tpr_ts, name=f'Test (AUC: {auc_ts:.3f})', line=dict(color='red', width=3)))
                
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(title="ROC Comparison", xaxis_title="FPR", yaxis_title="TPR", legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
                st.plotly_chart(fig_roc, use_container_width=True)
        else:
            # REGRESSION MULTIVARIATE
            if is_rf:
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
            else:
                X_train_const = sm.add_constant(X_train, has_constant='add')
                X_test_const = sm.add_constant(X_test, has_constant='add')
                train_preds = model.predict(X_train_const)
                test_preds = model.predict(X_test_const)
            
            c1, c2 = st.columns(2)
            for set_name, preds, actual, col in [('Train', train_preds, y_train, c1), ('Test', test_preds, y_test, c2)]:
                plot_df = pd.DataFrame({'Predicted': preds, 'Actual': actual})
                if len(plot_df) > 3000: plot_df = plot_df.sample(3000)
                fig = px.scatter(
                    plot_df, x='Predicted', y='Actual', opacity=0.3,
                    title=f"Performance: {set_name} Set",
                    labels={'Predicted': 'Model Prediction', 'Actual': 'Actual Return'}
                )
                mv, Mx = plot_df['Predicted'].min(), plot_df['Predicted'].max()
                fig.add_trace(go.Scatter(x=[mv, Mx], y=[mv, Mx], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
                col.plotly_chart(fig, use_container_width=True)

    # --- STATS ---
    st.markdown(f"### Analysis Results (N={len(final_df)} | {100-test_size*100:.0f}% Train / {test_size*100:.0f}% Test)")
    m1, m2, m3, m4 = st.columns(4)
    
    if is_classification:
        if is_rf:
            train_probs = model.predict_proba(X_train)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            train_auc = auc(*roc_curve(y_train, train_probs)[:2])
            test_auc = auc(*roc_curve(y_test, test_probs)[:2])
            train_acc = (model.predict(X_train) == y_train).mean()
            test_acc = (model.predict(X_test) == y_test).mean()
            
            m1.metric("ROC AUC (Train/Test)", f"{train_auc:.3f} / {test_auc:.3f}")
            m2.metric("Accuracy (Train/Test)", f"{train_acc:.1%} / {test_acc:.1%}")
            m3.metric("OOB/Score", f"{model.score(X_train, y_train):.3f}")
            m4.metric("Tree Count", f"{len(model.estimators_)}")
        else:
            # Logit stats (mostly on train)
            m1.metric("Pseudo R-Squared", f"{model.prsquared:.3f}")
            m2.metric("Accuracy (Train/Test)", f"{((model.predict(X_train_const)>0.5).astype(int)==y_train).mean():.1%} / {((model.predict(X_test_const)>0.5).astype(int)==y_test).mean():.1%}")
            m3.metric("LLR P-Value", f"{model.llr_pvalue:.2e}")
            m4.metric("N (Train/Test)", f"{len(y_train)} / {len(y_test)}")
    else:
        if is_rf:
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
            m1.metric("R-Squared (Train/Test)", f"{train_r2:.3f} / {test_r2:.3f}")
            m2.metric("MSE (Test)", f"{mean_squared_error(y_test, model.predict(X_test)):.6f}")
            m3.metric("Max Depth", f"{model.max_depth if model.max_depth else 'None'}")
            m4.metric("N (Train/Test)", f"{len(y_train)} / {len(y_test)}")
        else:
            # OLS
            train_r2 = model.rsquared
            # Manual test R2
            test_preds = model.predict(X_test_const)
            test_r2 = r2_score(y_test, test_preds)
            m1.metric("R-Squared (Train/Test)", f"{train_r2:.3f} / {test_r2:.3f}")
            m2.metric("Adj. R-Squared", f"{model.rsquared_adj:.3f}")
            m3.metric("F-Statistic", f"{model.fvalue:.1f}")
            m4.metric("Prob (F-stat)", f"{model.f_pvalue:.2e}")

    st.markdown("#### Feature Importance & Significance")
    if is_rf:
        # RF Importances
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        
        feat_imp_df = pd.DataFrame({
            "Feature": [format_metric(active_features[i]) for i in idx],
            "Importance": importances[idx],
            "Cumulative": np.cumsum(importances[idx])
        })
        st.dataframe(feat_imp_df.style.format({"Importance": "{:.4f}", "Cumulative": "{:.4f}"}), hide_index=True, use_container_width=True)
    else:
        var_names = ["Intercept"] + [format_metric(f) for f in active_features]
        
        res_data = {
            "Feature": var_names,
            "Coefficient": model.params,
            "Std Error": model.bse,
            "z-Stat" if is_logit else "t-Stat": model.tvalues,
            "P-Value": model.pvalues,
            "Conf. [0.025]": model.conf_int().iloc[:, 0],
            "Conf. [0.975]": model.conf_int().iloc[:, 1]
        }
        
        if is_logit:
            res_data["Odds Ratio"] = np.exp(model.params)
            
        coef_df = pd.DataFrame(res_data)
        coef_df = coef_df.sort_values("P-Value")
        
        fmt = {
            "Coefficient": "{:.6f}",
            "Std Error": "{:.6f}",
            "P-Value": "{:.4f}",
            "Conf. [0.025]": "{:.6f}",
            "Conf. [0.975]": "{:.6f}"
        }
        if is_logit:
            fmt["z-Stat"] = "{:.2f}"
            fmt["Odds Ratio"] = "{:.4f}"
        else:
            fmt["t-Stat"] = "{:.2f}"
            
        st.dataframe(coef_df.style.format(fmt), hide_index=True, use_container_width=True)

    # --- VIF DIAGNOSTICS ---
    if len(active_features) > 1:
        with st.expander("View Multicollinearity Diagnostics (VIF)", expanded=False):
            st.markdown("Features with VIF > threshold were automatically removed. Below are the final scores for retained features:")
            X_vif_final = sm.add_constant(X_data)
            vif_data = []
            for i, col in enumerate(X_vif_final.columns):
                if col == 'const': continue
                try:
                    v = variance_inflation_factor(X_vif_final.values, i)
                except:
                    v = np.inf
                vif_data.append({"Feature": format_metric(col), "VIF": v})
            
            vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
            st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), hide_index=True, use_container_width=True)

    if not is_rf:
        with st.expander("View Full Regression Table"):
            st.text(model.summary())

    # --- LIVE PREDICTIONS ---
    st.markdown("---")
    st.markdown("### Live Predictions (Current Data)")
    st.caption("Predictions based on the latest observation for each selected ticker.")
    
    prediction_rows = []
    skipped_tickers = []
    
    # Pre-calculate benchmark returns for rel_strength_z
    benchmark_returns = None
    if 'rel_strength_z' in active_features:
        b_df = manager.load_data(BENCHMARK_SYMBOL, selected_interval)
        if b_df is not None and not b_df.empty:
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()
    
    for ticker in selected_tickers:
        try:
            df_latest = manager.load_data(ticker, selected_interval)
            if df_latest is None or len(df_latest) < indicator_window:
                skipped_tickers.append((ticker, "Insufficient data"))
                continue
            
            # Calculate metrics for the latest observation
            # Get the latest values
            latest_vals = {}
            missing_feats = []
            for feat in active_features:
                try:
                    feat_series = metrics_engine.calculate_rolling_metric(
                        df_latest, feat, window=indicator_window, 
                        benchmark_returns=benchmark_returns,
                        interval=selected_interval
                    )
                    val = feat_series.iloc[-1] if len(feat_series) > 0 else np.nan
                    if standardize and feat in X_data.columns:
                        # Must use training mean/std for standardization
                        m = X_data[feat].mean()
                        s = X_data[feat].std()
                        if s > 1e-9:
                            val = (val - m) / s
                    latest_vals[feat] = val
                except Exception as ex:
                    latest_vals[feat] = np.nan
                    missing_feats.append(f"{feat}: {str(ex)[:30]}")
            
            if any(np.isnan(v) for v in latest_vals.values()):
                nan_feats = [k for k, v in latest_vals.items() if np.isnan(v)]
                skipped_tickers.append((ticker, f"NaN in: {', '.join(nan_feats[:3])}"))
                continue
                
            X_pred = pd.DataFrame([latest_vals])
            
            if is_rf:
                if is_classification:
                    prob = model.predict_proba(X_pred)[0, 1]
                    pred_label = "UP" if prob > 0.5 else "DOWN"
                    prediction_rows.append({"Ticker": ticker, "Prob(Up)": prob, "Signal": pred_label})
                else:
                    pred_ret = model.predict(X_pred)[0]
                    prediction_rows.append({"Ticker": ticker, "Predicted Return": pred_ret, "Signal": "UP" if pred_ret > 0 else "DOWN"})
            else:
                X_pred_const = sm.add_constant(X_pred, has_constant='add')
                if is_classification:
                    prob = model.predict(X_pred_const)[0]
                    pred_label = "UP" if prob > 0.5 else "DOWN"
                    prediction_rows.append({"Ticker": ticker, "Prob(Up)": prob, "Signal": pred_label})
                else:
                    pred_ret = model.predict(X_pred_const)[0]
                    prediction_rows.append({"Ticker": ticker, "Predicted Return": pred_ret, "Signal": "UP" if pred_ret > 0 else "DOWN"})
        except Exception as e:
            skipped_tickers.append((ticker, str(e)[:50]))
            continue
    
    if skipped_tickers:
        with st.expander(f"⚠️ {len(skipped_tickers)} ticker(s) skipped", expanded=False):
            for t, reason in skipped_tickers:
                st.caption(f"**{t}**: {reason}")
    
    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        if is_classification:
            pred_df = pred_df.sort_values("Prob(Up)", ascending=False).reset_index(drop=True)
        else:
            pred_df = pred_df.sort_values("Predicted Return", ascending=False).reset_index(drop=True)
                
        # Use dataframe with selection
        event = st.dataframe(
            pred_df.style.format({"Prob(Up)": "{:.2%}"} if is_classification else {"Predicted Return": "{:.4f}"}).background_gradient(
                subset=["Prob(Up)"] if is_classification else ["Predicted Return"], cmap="RdYlGn"
            ),
            hide_index=True, 
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="live_predictions_table"
        )
        
    else:
        st.info("No valid predictions could be generated.")

    # --- DETAILED SINGLE TICKER PREDICTION ---
    st.markdown("---")
    st.markdown("### 🎯 Predict Next Move for Selected Ticker")
    st.caption(f"Get detailed prediction for a single ticker based on the fitted {reg_type} model.")
    
    # Store the model and settings in session state for use
    st.session_state['fitted_model'] = model
    st.session_state['active_features'] = active_features
    st.session_state['is_rf'] = is_rf
    st.session_state['is_classification'] = is_classification
    st.session_state['standardize'] = standardize
    st.session_state['X_data'] = X_data
    st.session_state['indicator_window'] = indicator_window
    st.session_state['fwd_window'] = fwd_window
    st.session_state['selected_interval'] = selected_interval

# --- Prediction Section (outside the button block so it persists) ---
if 'fitted_model' in st.session_state:
    model = st.session_state['fitted_model']
    active_features = st.session_state['active_features']
    is_rf = st.session_state['is_rf']
    is_classification = st.session_state['is_classification']
    standardize = st.session_state['standardize']
    X_data = st.session_state['X_data']
    indicator_window = st.session_state['indicator_window']
    fwd_window = st.session_state['fwd_window']
    selected_interval_stored = st.session_state['selected_interval']
    
    st.markdown("---")
    st.markdown("### Predict Next Move for Selected Ticker")
    
    # Get default ticker from Live Predictions selection, or first ticker
    default_ticker = st.session_state.get('selected_focus_ticker', selected_tickers[0] if selected_tickers else None)
    default_index = 0
    if default_ticker and default_ticker in selected_tickers:
        default_index = selected_tickers.index(default_ticker)
    
    focus_ticker = st.selectbox(
        "Choose Ticker to Predict", 
        options=selected_tickers, 
        index=default_index,
        key="focus_ticker_select"
    )
    
    if focus_ticker:
        try:
            # Load latest data for the ticker
            df_focus = manager.load_data(focus_ticker, selected_interval_stored)
            if df_focus is None or len(df_focus) < indicator_window:
                st.error(f"Insufficient data for {focus_ticker}.")
            else:
                # Get latest price info
                latest_close = float(df_focus['close'].iloc[-1])
                latest_time = df_focus.index[-1] if hasattr(df_focus, 'index') else "N/A"
                
                # Pre-calculate benchmark returns for rel_strength_z
                benchmark_returns = None
                if 'rel_strength_z' in active_features:
                    b_df = manager.load_data(BENCHMARK_SYMBOL, selected_interval_stored)
                    if b_df is not None and not b_df.empty:
                        b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
                        benchmark_returns = b_close.pct_change().dropna()
                
                # Build feature vector by calculating each metric
                latest_vals = {}
                feature_display = []
                for feat in active_features:
                    try:
                        feat_series = metrics_engine.calculate_rolling_metric(
                            df_focus, feat, window=indicator_window, 
                            benchmark_returns=benchmark_returns,
                            interval=selected_interval_stored
                        )
                        if len(feat_series) > 0:
                            raw_val = feat_series.iloc[-1]
                            
                            if standardize:
                                m = X_data[feat].mean()
                                s = X_data[feat].std()
                                if s > 1e-9:
                                    standardized_val = (raw_val - m) / s
                                    latest_vals[feat] = standardized_val
                                    feature_display.append({
                                        "Feature": format_metric(feat), 
                                        "Raw Value": raw_val,
                                        "Standardized": standardized_val
                                    })
                                else:
                                    latest_vals[feat] = raw_val
                                    feature_display.append({
                                        "Feature": format_metric(feat), 
                                        "Raw Value": raw_val,
                                        "Standardized": raw_val
                                    })
                            else:
                                latest_vals[feat] = raw_val
                                feature_display.append({
                                    "Feature": format_metric(feat), 
                                    "Value": raw_val
                                })
                        else:
                            latest_vals[feat] = np.nan
                    except Exception:
                        latest_vals[feat] = np.nan
                
                if any(np.isnan(v) for v in latest_vals.values()):
                    st.error("Some features have missing values. Cannot generate prediction.")
                else:
                    X_pred = pd.DataFrame([latest_vals])
                    
                    # Make prediction
                    if is_rf:
                        if is_classification:
                            prob = model.predict_proba(X_pred)[0, 1]
                            pred_class = model.predict(X_pred)[0]
                            signal = "📈 UP" if pred_class == 1 else "📉 DOWN"
                            confidence = max(prob, 1 - prob)
                        else:
                            pred_ret = model.predict(X_pred)[0]
                            signal = "📈 UP" if pred_ret > 0 else "📉 DOWN"
                            confidence = None
                    else:
                        X_pred_const = sm.add_constant(X_pred, has_constant='add')
                        if is_classification:
                            prob = model.predict(X_pred_const)[0]
                            signal = "📈 UP" if prob > 0.5 else "📉 DOWN"
                            confidence = max(prob, 1 - prob)
                            pred_class = 1 if prob > 0.5 else 0
                        else:
                            pred_ret = model.predict(X_pred_const)[0]
                            signal = "📈 UP" if pred_ret > 0 else "📉 DOWN"
                            confidence = None
                    
                    # Display Results
                    st.markdown(f"#### Prediction for **{focus_ticker}**")
                    
                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current Price", f"${latest_close:,.4f}" if latest_close < 1 else f"${latest_close:,.2f}")
                    m2.metric("Prediction Window", f"{fwd_window} periods")
                    
                    if is_classification:
                        m3.metric("Predicted Direction", signal)
                        m4.metric("Confidence", f"{confidence:.1%}")
                    else:
                        m3.metric("Predicted Return", f"{pred_ret:+.2%}")
                        m4.metric("Direction", signal)
                    
                    # Prediction Box
                    if is_classification:
                        if pred_class == 1:
                            st.success(f"**{signal}** — Model predicts {focus_ticker} will go UP in the next {fwd_window} periods with {confidence:.1%} confidence.")
                        else:
                            st.error(f"**{signal}** — Model predicts {focus_ticker} will go DOWN in the next {fwd_window} periods with {confidence:.1%} confidence.")
                    else:
                        expected_price = latest_close * np.exp(pred_ret)
                        if pred_ret > 0:
                            st.success(f"**{signal}** — Model predicts {focus_ticker} will return **{pred_ret:+.2%}** over the next {fwd_window} periods.\n\nExpected price: **${expected_price:,.4f}**" if expected_price < 1 else f"**{signal}** — Model predicts {focus_ticker} will return **{pred_ret:+.2%}** over the next {fwd_window} periods.\n\nExpected price: **${expected_price:,.2f}**")
                        else:
                            st.error(f"**{signal}** — Model predicts {focus_ticker} will return **{pred_ret:+.2%}** over the next {fwd_window} periods.\n\nExpected price: **${expected_price:,.4f}**" if expected_price < 1 else f"**{signal}** — Model predicts {focus_ticker} will return **{pred_ret:+.2%}** over the next {fwd_window} periods.\n\nExpected price: **${expected_price:,.2f}**")
                    
                    # Feature Values Table
                    with st.expander("View Current Feature Values", expanded=True):
                        feat_df = pd.DataFrame(feature_display)
                        if standardize:
                            st.dataframe(
                                feat_df.style.format({"Raw Value": "{:.4f}", "Standardized": "{:.4f}"}),
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.dataframe(
                                feat_df.style.format({"Value": "{:.4f}"}),
                                hide_index=True, use_container_width=True
                            )
                            
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
else:
    st.markdown("---")
    st.info("👆 Run **Regression Analysis** first to enable single-ticker predictions.")
