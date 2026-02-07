"""
Regime Filter Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataManager
from src.metrics import MetricsEngine
from labler_algo import StationarityLabeler, Labeler
from src.config import APP_TITLE, APP_ICON, METRIC_LABELS, ALL_METRICS, DEFAULT_FETCH_INTERVALS, BENCHMARK_SYMBOL

st.set_page_config(page_title=f"{APP_TITLE} - Regime Filter", page_icon=APP_ICON, layout="wide")

st.markdown(f"## 📈 Regime Classification & Filtering")

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

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.markdown("### Data Selection")
inventory = manager.get_inventory()
all_symbols = sorted(list(inventory.keys()))

if not all_symbols:
    st.warning("No data available. Please fetch data using the Data Loader.")
    st.stop()

selected_symbol = st.sidebar.selectbox("Select Ticker", all_symbols)
avail_intervals = sorted(inventory.get(selected_symbol, []))

if not avail_intervals:
    st.error(f"No intervals found for {selected_symbol}")
    st.stop()
    
selected_interval = st.sidebar.selectbox("Select Interval", avail_intervals, index=0)

st.sidebar.markdown("### Regime Labeling Parameters")
regime_option = st.sidebar.radio("Regime Type", ["Mean Reverting (Stationarity)", "Trend (Directional)"])

if regime_option == "Mean Reverting (Stationarity)":
    st.sidebar.caption("Identifies periods where price is stationary vs. trending.")
    stat_window = st.sidebar.number_input("Window Size", min_value=10, max_value=200, value=50)
    vote_th = st.sidebar.slider("Vote Threshold", 0.1, 1.0, 0.4, 0.05, help="Higher = stricter stationarity")
else:
    st.sidebar.caption("Identifies Up/Down/Neutral trends based on price action.")
    amp_th = st.sidebar.number_input("Amplitude Threshold (%)", min_value=1, max_value=200, value=50)
    max_inactive = st.sidebar.number_input("Max Inactive Period", min_value=1, max_value=20, value=5)

st.sidebar.markdown("### Feature Engineering")
indicator_window = st.sidebar.number_input("Indicator Lookback", min_value=2, max_value=200, value=14)

st.sidebar.markdown("### Preprocessing")
standardize = st.sidebar.toggle("Standardize Features (Z-Score)", value=True)
test_size = st.sidebar.slider("Test Set Ratio (%)", 10, 50, 20, 5) / 100.0

st.sidebar.markdown("### Model Selection")
model_type = st.sidebar.selectbox("Classifier", ["Result Weighted Voting (Ensemble)", "Logistic Regression", "Random Forest"])

rf_max_depth = 5
if model_type == "Random Forest":
    rf_max_depth = st.sidebar.slider("RF: Max Depth", 2, 20, 5)

# --- Main Logic ---
select_all_x = st.checkbox("Select All Predictors", value=False)
# Safe default selection
safe_defaults = [m for m in ['price_zscore', 'rsi_norm', 'volatility'] if m in ALL_METRICS]
if not safe_defaults and ALL_METRICS:
    safe_defaults = [ALL_METRICS[0]]

default_x = ALL_METRICS if select_all_x else safe_defaults

x_features = st.multiselect(
    "Predictors (X)", 
    ALL_METRICS, 
    default=default_x,
    format_func=format_metric
)

if len(x_features) > 1:
    vif_threshold = st.sidebar.slider("VIF Threshold", 2.0, 20.0, 10.0)
else:
    vif_threshold = 999.0

# -----------------------------
# Data Loading & Labeling
# -----------------------------
status_container = st.empty()

@st.cache_data(show_spinner=False)
def load_and_label(symbol, interval, regime_type, s_window, s_vote, t_amp, t_inactive):
    df = manager.load_data(symbol, interval)
    if df is None or df.empty:
        return None, None
    
    # Ensure Datetime Index for plotting and timedelta calculations
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time')
    
    price_s = df['close']
    
    if "Mean Reverting" in regime_type:
        labeler = StationarityLabeler(window=s_window, vote_th=s_vote)
        labels, _ = labeler.label(price_s)
        # 1 = Stationarity (Mean Reverting), 0 = Random Walk/Trend
    else:
        labeler = Labeler(amplitude_threshold=t_amp, max_inactive_period=t_inactive)
        trend_df = labeler.label(price_s)
        labels = trend_df['label'] 
        
        # FIX: Labeler resets index. Restore DatetimeIndex if lengths match.
        if len(labels) == len(price_s):
            labels.index = price_s.index
        else:
            # If length changed (e.g. dropna inside), we try to align?
            # Assuming price_s is clean from load_data
            # Just warn if mismatch
            pass
    
    return df, labels

with status_container.status("Processing Data..."):
    df_raw, regime_labels = load_and_label(
        selected_symbol, selected_interval, regime_option, 
        stat_window if 'stat_window' in locals() else 50, 
        vote_th if 'vote_th' in locals() else 0.4,
        amp_th if 'amp_th' in locals() else 50,
        max_inactive if 'max_inactive' in locals() else 5
    )

if df_raw is None:
    st.error("Failed to load data.")
    st.stop()

# -----------------------------
# Visualization: Price & Regimes
# -----------------------------
st.write("### Historical Regimes")
price_series = df_raw['close']
fig = go.Figure()
fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values, name='Price', line=dict(color='gray', width=1)))

# Create colored rectangles for regimes
# Optimize by grouping consecutive indices
regime_changes = regime_labels.ne(regime_labels.shift()).cumsum()
grouped_regimes = regime_labels.groupby(regime_changes)

for _, group in grouped_regimes:
    label_val = group.iloc[0]
    if label_val != 0: # Assuming 0 is "Neutral" or "Trending/Random" depending on mode
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        if "Mean Reverting" in regime_option:
            # 1 = Mean Reverting (Green), 0 = Trending (Transparent/None)
            if label_val == 1:
                color = "rgba(0, 255, 0, 0.2)" # Green for Mean Reversion
                name = "Mean Reversion"
            else:
                continue
        else:
            # Trend: 1 = Up (Green), -1 = Down (Red)
            if label_val == 1:
                color = "rgba(0, 255, 0, 0.2)"
                name = "Up Trend"
            elif label_val == -1:
                color = "rgba(255, 0, 0, 0.2)"
                name = "Down Trend"
            else:
                continue
        
        # Calculate duration if possible
        duration_sec = 0
        if isinstance(group.index, pd.DatetimeIndex):
            duration_sec = (group.index[-1] - group.index[0]).total_seconds()
        else:
             # Fallback for integer index (assuming 4h candles)
            duration_sec = (group.index[-1] - group.index[0]) * 4 * 3600
            
        fig.add_vrect(
            x0=start_idx, x1=end_idx,
            fillcolor=color, layer="below", line_width=0,
            annotation_text=name if duration_sec > 3600*24*10 else None,
            annotation_position="top left"
        )

fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Feature Engineering & Alignment
# -----------------------------
@st.cache_data(show_spinner=False)
def calculate_feature_matrix(symbol, interval, features, ind_window):
    # We load data again or pass df. It's fast to reload from memory cache
    d = manager.load_data(symbol, interval)
    
    if 'open_time' in d.columns:
        d['open_time'] = pd.to_datetime(d['open_time'])
        d = d.set_index('open_time')
        
    feats = {}
    for f in features:
        try:
            feats[f] = metrics_engine.calculate_rolling_metric(d, f, window=ind_window)
        except:
            feats[f] = pd.Series(np.nan, index=d.index)
    return pd.DataFrame(feats, index=d.index)

with status_container.status("Calculating Features..."):
    X_all = calculate_feature_matrix(selected_symbol, selected_interval, x_features, indicator_window)
    
    # Align X and y
    # regime_labels might be longer or shorter? They come from same df, so index should match.
    common_index = X_all.index.intersection(regime_labels.index)
    X_aligned = X_all.loc[common_index]
    y_aligned = regime_labels.loc[common_index]
    
    # Drop NaNs
    valid_mask = ~X_aligned.isna().any(axis=1) & ~y_aligned.isna()
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
# --- Feature Inspection Table ---
with st.expander("View Feature Data (X & y)", expanded=False):
    view_df = X_aligned.copy() # Visualize data INCLUDING NaNs
    view_df['Regime_Label'] = y_aligned
    st.write(f"Data Points: {len(view_df)} (including NaNs)")
    st.dataframe(view_df.tail(100), use_container_width=True)

if len(X_clean) < 100:
    st.error("Insufficient data points after cleaning (NaN removal). Need at least 100.")
    st.stop()

# -----------------------------
# Classification Setup
# -----------------------------
# Check classes
classes = np.unique(y_clean)
n_classes = len(classes)

if n_classes < 2:
    st.warning(f"Only 1 class found in data: {classes}. Cannot train classifier.")
    st.stop()

# For "Mean Reverting", classes are likely [0, 1].
# For "Trend", classes are likely [-1, 0, 1].
# If standardizing:
# VIF Filter
# Run VIF on cleaned data (unscaled or scaled doesn't matter much for VIF, but unscaled is safer for interpretation)
# We determine columns to KEEP first
if len(x_features) > 1 and vif_threshold < 20.0:
    with status_container.status("Checking Multicollinearity..."):
        # Use X_clean for VIF check
        kept_cols = apply_vif_filter(X_clean, threshold=vif_threshold)
        
        # Safety check: if VIF removes everything (unlikely), keep at least one
        if not kept_cols:
            st.warning("VIF removed all features! Reverting to original selection.")
            kept_cols = x_features
            
        dropped_cols = set(x_features) - set(kept_cols)
        if dropped_cols:
            st.warning(f"dropped VIF > {vif_threshold}: {dropped_cols}")
else:
    kept_cols = x_features

# Filter X to kept columns
X_final = X_clean[kept_cols].copy()

# Standardize
if standardize:
    scaler = StandardScaler()
    X_vals = scaler.fit_transform(X_final)
    # Re-create DataFrame with correct columns
    X_final = pd.DataFrame(X_vals, columns=kept_cols, index=X_final.index)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_clean, test_size=test_size, shuffle=False) # Time series split (shuffle=False)

# -----------------------------
# Modeling
# -----------------------------
# Note: For Trend (-1, 0, 1), simple LogisticRegression interprets as multinomial naturally.
# RF also handles multiclass.
# However, ROC curves are typically binary or need One-vs-Rest for multiclass.

st.markdown("### Model Training & Performance")

try:
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, multi_class='auto')
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=rf_max_depth, random_state=42)
    else:
        # Fallback or Ensemble placeholder
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    model.fit(X_train, y_train)
    
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

# Evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

if hasattr(model, "predict_proba"):
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
else:
    train_probs = None
    test_probs = None

c1, c2, c3 = st.columns(3)
c1.metric("Train Accuracy", f"{train_acc:.2%}")
c2.metric("Test Accuracy", f"{test_acc:.2%}")
c3.metric("Samples (Train/Test)", f"{len(y_train)} / {len(y_test)}")

# -----------------------------
# Performance Visuals
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Probabilities", "Feature Importance", "Confusion Matrix"])

with tab1:
    # Plot Probability Distributions (If Binary, class 1. If Multiclass, max prob?)
    # Let's assume Binary for Mean Reversion (0/1) or focus on "Signal Class" for Trend.
    if n_classes == 2:
        # Binary
        pos_class_idx = 1 # Assuming class 1 is the "Regime" we want
        # Ensure we have probabilities
        if train_probs is not None:
            # Check if pos_class_idx is valid 
            if pos_class_idx >= train_probs.shape[1]: 
                pos_class_idx = 0
            
            dist_df = pd.concat([
                pd.DataFrame({'Prob': train_probs[:, pos_class_idx], 'Set': 'Train', 'Actual': y_train}),
                pd.DataFrame({'Prob': test_probs[:, pos_class_idx], 'Set': 'Test', 'Actual': y_test})
            ])
            
            fig_hist = px.histogram(
                dist_df, x='Prob', color='Actual', facet_col='Set',
                barmode='overlay', nbins=50,
                color_discrete_map={0: 'red', 1: 'green', -1: 'red'}, # Adjust mapping as needed
                title=f"Probability Distribution (Class {classes[pos_class_idx]})"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # ROC Curve - Only for Binary (0, 1) or mapped binary
            try:
                # Need to handle if y is not 0/1 but -1/1 etc.
                # pos_label needs to be one of the classes
                pos_lbl = classes[pos_class_idx]
                
                fpr_tr, tpr_tr, _ = roc_curve(y_train, train_probs[:, pos_class_idx], pos_label=pos_lbl)
                auc_tr = auc(fpr_tr, tpr_tr)
                
                fpr_ts, tpr_ts, _ = roc_curve(y_test, test_probs[:, pos_class_idx], pos_label=pos_lbl)
                auc_ts = auc(fpr_ts, tpr_ts)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_tr, y=tpr_tr, name=f'Train AUC: {auc_tr:.3f}'))
                fig_roc.add_trace(go.Scatter(x=fpr_ts, y=tpr_ts, name=f'Test AUC: {auc_ts:.3f}'))
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not calculate ROC: {e}")
            
    else:
        st.info(f"Multiclass Classification ({n_classes} classes). ROC Curve skipped. See Confusion Matrix.")

with tab2:
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": kept_cols,
            "Importance": imps
        }).sort_values("Importance", ascending=False)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.bar_chart(feat_df.set_index("Feature"))
    elif hasattr(model, "coef_"):
        # For Logit, coefs are (n_classes, n_features) or (1, n_features)
        coefs = model.coef_[0] # simplified
        feat_df = pd.DataFrame({
            "Feature": kept_cols,
            "Coefficient": coefs 
        }).sort_values("Coefficient", ascending=False)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.write("Train Confusion Matrix")
        cm_tr = confusion_matrix(y_train, train_pred)
        st.write(pd.DataFrame(cm_tr, index=classes, columns=classes))
    with col2:
        st.write("Test Confusion Matrix")
        cm_ts = confusion_matrix(y_test, test_pred)
        st.write(pd.DataFrame(cm_ts, index=classes, columns=classes))

# -----------------------------
# Live Prediction
# -----------------------------
st.markdown("---")
st.markdown(f"### 🎯 Live Regime Prediction: {selected_symbol}")

# Get latest features
# We need to compute features on the FULL latest dataset to get the last point correctly
# Reuse load_data to get latest
# Reuse load_data to get latest
df_latest = manager.load_data(selected_symbol, selected_interval)
if df_latest is not None:
    if 'open_time' in df_latest.columns:
        df_latest['open_time'] = pd.to_datetime(df_latest['open_time'])
        df_latest = df_latest.set_index('open_time')
        
    # Compute vector
    latest_feat = {}
    for f in kept_cols:
        s = metrics_engine.calculate_rolling_metric(df_latest, f, window=indicator_window)
        latest_feat[f] = s.iloc[-1]
    
    X_latest_df = pd.DataFrame([latest_feat])
    # Ensure correct column order matching scaler fit
    X_latest_df = X_latest_df[kept_cols]
    
    # Standardize if needed (using scaler fitted on TRAIN+TEST or just TRAIN? Ideally properties of Training data)
    # Here we used scaler on X_clean (all valid data). Let's assume that's "History"
    if standardize:
        X_latest_vals = scaler.transform(X_latest_df) # Transform using the scaler fitted on historical data
    else:
        X_latest_vals = X_latest_df.values
        
    # Predict
    curr_pred = model.predict(X_latest_vals)[0]
    curr_probs = model.predict_proba(X_latest_vals)[0] if hasattr(model, "predict_proba") else None
    
    # Display
    c_p1, c_p2 = st.columns([1, 2])
    
    with c_p1:
        st.metric("Current Price", f"{df_latest['close'].iloc[-1]:.4f}")
        
    with c_p2:
        regime_name = str(curr_pred)
        if "Mean Reverting" in regime_option:
            regime_name = "🔄 Mean Reverting" if curr_pred == 1 else "➡️ Random/Trend"
            color = "green" if curr_pred == 1 else "gray"
        else:
            if curr_pred == 1: regime_name = "📈 Up Trend"; color="green"
            elif curr_pred == -1: regime_name = "📉 Down Trend"; color="red"
            else: regime_name = "➡️ Neutral"; color="gray"
            
        st.markdown(f"#### Predicted State: :{color}[{regime_name}]")
        
        if curr_probs is not None:
            # Show confidence
            # Find prob of the predicted class
            # classes are the model.classes_
            class_idx = np.where(model.classes_ == curr_pred)[0][0]
            confidence = curr_probs[class_idx]
            st.write(f"Confidence: **{confidence:.1%}**")
            
            # Show full prob breakdown
            prob_df = pd.DataFrame([curr_probs], columns=[str(c) for c in model.classes_], index=["Probability"])
            st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)
