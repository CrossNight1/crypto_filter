from shiny import ui, render, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from src.data import DataManager
from ml_engine.analysis.correlation import CorrelationEngine, DecompositionEngine
from scipy.cluster.hierarchy import linkage
from src.config import AVAILABLE_INTERVALS, MANDATORY_CRYPTO
from concurrent.futures import ThreadPoolExecutor, as_completed

def _sanitize(data):
    """Replace inf/nan with 0 to prevent Plotly JSON serialization errors."""
    if isinstance(data, pd.DataFrame):
        return data.replace([np.inf, -np.inf], np.nan).fillna(0)
    elif isinstance(data, np.ndarray):
        d = np.where(np.isfinite(data), data, 0)
        return d
    return data


def multivariate_analysis_ui():
    return ui.navset_card_underline(
        ui.nav_panel("Matrix Radar",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Matrix Filter"),
                    ui.input_action_button(
                        "btn_gen_corr",
                        "Generate Matrix",
                        class_="btn-primary w-100 mt-3"
                    ),

                    ui.input_text(
                        "focus_corr_symbol",
                        "Focus Symbol",
                        value="",
                        placeholder="e.g. BTCUSDT"
                    ),

                    ui.input_select(
                        "dependence_structure",
                        "Dependence Structure",
                        choices=["Correlation", "Covariance", "Cointegration"]
                    ),

                    ui.panel_conditional(
                        "input.dependence_structure === 'Correlation'",
                        ui.input_select(
                            "corr_method",
                            "Method",
                            choices=["pearson", "kendall", "spearman"]
                        ),
                    ),

                    ui.input_select("corr_interval", "Timeframe", choices=AVAILABLE_INTERVALS, selected="1h"),

                    ui.input_numeric(
                        "window_size",
                        "Window Size",
                        value=100,
                        min=10,
                        max=2000
                    ),

                    ui.input_text(
                        "n_assets",
                        "Top Volume",
                        value="20",
                        placeholder="e.g. 20",
                        update_on="blur"
                    ),

                    ui.input_selectize(
                        "corr_symbols",
                        "Select Symbols",
                        choices=[],
                        multiple=True
                    )
                ),
                ui.output_ui("matrix_view")
            )
        ),
        ui.nav_panel("Decomposition",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Decomposition"),

                    ui.input_action_button(
                        "btn_run_decomp",
                        "Run Decomposition",
                        class_="btn-primary w-100 mt-3"
                    ),

                    ui.input_select(
                        "decomp_method",
                        "Method",
                        choices={
                            "eigen": "PCA (Eigenvalue)",
                            "rmt": "RMT Spectral Filter",
                            # "kfactor": "K-Factor",
                            # "ica": "ICA",
                            # "distance": "Distance Matrix",
                            "cluster": "Hierarchical Clustering",
                            "mst": "Spillover Network"
                        }
                    ),

                    ui.panel_conditional(
                        "input.decomp_method === 'ica' || input.decomp_method === 'eigen'",
                        ui.input_slider(
                            "n_components",
                            "Category Count (K)",
                            value=5,
                            min=1,
                            max=5,
                            step=1
                        )
                    ),

                    ui.panel_conditional(
                        "input.decomp_method === 'kfactor'",
                        ui.input_select(
                            "k_factor_mode",
                            "Filter Mode",
                            choices={
                                "top": "Systematic",
                                "bottom": "Idiosyncratic"
                            }
                        ),
                    ),

                    ui.panel_conditional(
                        "input.decomp_method === 'cluster'",
                        ui.input_select(
                            "linkage_method",
                            "Linkage",
                            choices=["ward", "single", "complete", "average"]
                        ),
                    ),

                    ui.panel_conditional(
                        "input.decomp_method === 'mst'",
                        ui.input_select(
                            "spillover_type",
                            "Spillover Type",
                            choices={
                                "volatility": "Volatility",
                                "return": "Return"
                            }
                        ),
                    ),

                    ui.input_select("decomp_interval", "Timeframe", choices=[]),

                    ui.input_numeric(
                        "decomp_window",
                        "Window Size",
                        value=100,
                        min=10,
                        max=2000
                    ),

                    ui.input_text(
                        "decomp_n_assets",
                        "Top Volume",
                        value="20",
                        placeholder="e.g. 20",
                        update_on="blur"
                    ),

                    ui.input_selectize(
                        "decomp_symbols",
                        "Select Symbols",
                        choices=[],
                        multiple=True
                    )
                ),
                ui.output_ui("decomp_view")
            )
        )
    )


def multivariate_analysis_server(input, output, session):

    manager = DataManager()
    correlation_matrix = reactive.Value(pd.DataFrame())
    decomp_result = reactive.Value(None)
    
    # Track selected symbols for each tab
    selected_symbols_corr = reactive.Value(set())
    selected_symbols_decomp = reactive.Value(set())

    # ── Shared inventory updates ──────────────────────────────

    @reactive.effect
    def _():
        inventory = manager.get_inventory()
        if not inventory:
            return
        # intervals = sorted(list(set(i for ivs in inventory.values() for i in ivs)))
        ui.update_select("corr_interval", choices=AVAILABLE_INTERVALS, selected="1h")
        ui.update_select("decomp_interval", choices=AVAILABLE_INTERVALS, selected="1h")


    @reactive.effect
    def _():
        # Initialize selections if empty
        if not selected_symbols_corr.get():
            syms = manager.fetcher.get_top_volume_symbols(top_n=input.n_assets())
            selected_symbols_corr.set(set(MANDATORY_CRYPTO).union(syms))
        
        if not selected_symbols_decomp.get():
            syms = manager.fetcher.get_top_volume_symbols(top_n=input.decomp_n_assets())
            selected_symbols_decomp.set(set(MANDATORY_CRYPTO).union(syms))

        # React to interval changes but preserve or update selections based on inventory
        inventory = manager.get_inventory()
        if not inventory:
            return
            
        corr_int = input.corr_interval()
        all_syms_corr = sorted([s for s, ints in inventory.items() if corr_int in ints])
        curr_sel_corr = sorted(list(selected_symbols_corr.get()))
        # Filter selected symbols to ensure they exist in current interval if needed 
        # (or just show all choices and let data loader handle missing data)
        ui.update_selectize("corr_symbols", choices=all_syms_corr, selected=curr_sel_corr)
        
        decomp_int = input.decomp_interval()
        all_syms_decomp = sorted([s for s, ints in inventory.items() if decomp_int in ints])
        curr_sel_decomp = sorted(list(selected_symbols_decomp.get()))
        ui.update_selectize("decomp_symbols", choices=all_syms_decomp, selected=curr_sel_decomp)
        
    # ── Helper: load return data ──────────────────────────────

    def _load_return_data(symbols, interval, progress=None):
        """Shared helper to load log-return series for symbols concurrently."""
        data_map = {}
        
        def fetch_symbol_returns(sym):
            try:
                df = manager.load_data(sym, interval)
                if df is not None and not df.empty:
                    close_vals = df.set_index("open_time")["close"]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ret_series = np.log(close_vals / close_vals.shift(1))
                    return sym, ret_series.replace([np.inf, -np.inf], np.nan).dropna()
            except Exception as e:
                print(f"Error loading {sym}: {e}")
            return sym, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(fetch_symbol_returns, sym): sym for sym in symbols}
            
            for i, future in enumerate(as_completed(future_to_sym)):
                sym, result = future.result()
                if result is not None:
                    data_map[sym] = result
                if progress:
                    progress.set(i + 1)
        return data_map

    def _load_price_data(symbols, interval, progress=None):
        """Shared helper to load price series for symbols concurrently."""
        data_map = {}
        
        def fetch_symbol_price(sym):
            try:
                df = manager.load_data(sym, interval)
                if df is not None and not df.empty:
                    return sym, df.set_index("open_time")["close"]
            except Exception as e:
                print(f"Error loading {sym}: {e}")
            return sym, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(fetch_symbol_price, sym): sym for sym in symbols}
            
            for i, future in enumerate(as_completed(future_to_sym)):
                sym, result = future.result()
                if result is not None:
                    data_map[sym] = result
                    progress.set(i + 1)
        return data_map
    @reactive.effect
    def _():
        try:
            val = input.n_assets()
            if not val: return
            n_assets = int(val)
        except ValueError:
            return
            
        interval = input.corr_interval()
        
        with ui.Progress(min=0, max=100) as p:
            p.set(5, message="Refreshing symbols...", detail=f"Fetching top {n_assets} high-volume assets")
            new_syms = manager.fetcher.get_top_volume_symbols(top_n=n_assets)
            syms = sorted(list(set(MANDATORY_CRYPTO).union(new_syms)))
            selected_symbols_corr.set(set(syms))
            
            inventory = manager.get_inventory()
            sym_choices = sorted([s for s, ints in inventory.items() if interval in ints])
            ui.update_selectize("corr_symbols", choices=sym_choices, selected=syms)

            if n_assets > 10:
                p.set(10, message="Syncing ticker data...", detail=f"Updating {len(syms)} assets")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(manager.load_data, s, interval, auto_sync=True) for s in syms]
                    for i, _ in enumerate(as_completed(futures)):
                        p.set(10 + int(90 * (i+1)/len(syms)), detail=f"Syncing {i+1}/{len(syms)}: {syms[i]}")
            p.set(100, message="Sync complete")

    # ══════════════════════════════════════════════════════════
    # TAB 1: MATRIX RADAR
    # ══════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.btn_gen_corr)
    def _():
        interval = input.corr_interval()
        structure = input.dependence_structure()
        syms = list(input.corr_symbols())

        if not syms:
            ui.notification_show("Please select at least one symbol", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            # 1. Check data availability
            p.set(10, message="Checking data...", detail=f"Preparing {len(syms)} assets")
            symbols = syms
            if structure == "Correlation":
                data_map = _load_return_data(symbols, interval, p)
                if not data_map: return
                raw_matrix = CorrelationEngine.calculate_matrix(
                    data_map,
                    method=input.corr_method(),
                    window_size=input.window_size()
                )
            elif structure == "Covariance":
                data_map = _load_return_data(symbols, interval, p)
                if not data_map: return
                raw_matrix = CorrelationEngine.calculate_covariance(
                    data_map,
                    window_size=input.window_size()
                )
            elif structure == "Cointegration":
                price = _load_price_data(symbols, interval, p)
                if not price: return
                price_df = pd.DataFrame(price)
                raw_matrix = CorrelationEngine.calculate_coint_matrix(
                    price_df,
                    window_size=input.window_size()
                )
                raw_matrix = raw_matrix * -1
                
            elif structure == "Partial Correlation":
                data_map = _load_return_data(symbols, interval, p)
                if not data_map: return
                raw_matrix = CorrelationEngine.calculate_partial_correlation(
                    data_map,
                    window_size=input.window_size()
                )
            elif structure == "Precision Matrix":
                data_map = _load_return_data(symbols, interval, p)
                if not data_map: return
                raw_matrix = CorrelationEngine.calculate_precision_matrix(
                    data_map,
                    window_size=input.window_size()
                )
            else:
                return

            filtered_matrix, _ = CorrelationEngine.filter_blanks(raw_matrix)
            filtered_matrix = filtered_matrix.sort_values(by=filtered_matrix.columns[0], ascending=False)
            correlation_matrix.set(filtered_matrix)

    @render.ui
    def matrix_view():
        df = correlation_matrix.get()
        if df.empty:
            # print("Matrix is empty")
            return None

        df = df[-input.window_size():]
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        print("Matrix generated")
        return ui.div(
            ui.card(
                ui.card_header(f"{input.dependence_structure()} Matrix"),
                output_widget("matrix_chart"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Raw Data"),
                ui.output_data_frame("matrix_table")
            )
        )

    @render.data_frame
    def matrix_table():
        df = correlation_matrix.get()
        if df.empty:
            return None
            
        # Match cleaning logic applied in the view
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        
        return df

    @render_widget
    def matrix_chart():
        df = correlation_matrix.get()
        if df.empty:
            return go.Figure()

        structure = input.dependence_structure()
        if structure == "Correlation":
            zmin, zmax = -1, 1
        else:
            zmin, zmax = None, None

        # Sanitize for Plotly
        clean_df = _sanitize(df)

        fig = px.imshow(
            clean_df, text_auto=".2f", aspect="auto",
            color_continuous_scale="Spectral_r",
            zmin=zmin, zmax=zmax,
            template="plotly_dark"
        )

        focus_sym = input.focus_corr_symbol().strip().upper()
        if focus_sym:
            # Check if symbol exists in columns or index
            cols = list(clean_df.columns)
            indices = list(clean_df.index)
            
            shapes = []
            
            if focus_sym in cols:
                col_idx = cols.index(focus_sym)
                # Highlight Column
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=col_idx - 0.5, y0=0,
                    x1=col_idx + 0.5, y1=1,
                    line=dict(color="#FF00FF", width=2),
                    fillcolor="rgba(0,0,0,0)"
                ))
            
            if focus_sym in indices:
                row_idx = indices.index(focus_sym)
                # Highlight Row
                shapes.append(dict(
                    type="rect",
                    xref="paper", yref="y",
                    x0=0, y0=row_idx - 0.5, 
                    x1=1, y1=row_idx + 0.5,
                    line=dict(color="#FF00FF", width=2),
                    fillcolor="rgba(0,0,0,0)"
                ))
                
            if shapes:
                fig.update_layout(shapes=shapes)
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.3)"),
            height=500, width=1470
        )
        return fig

    @reactive.effect
    def _():
        try:
            val = input.decomp_n_assets()
            if not val: return
            n_assets = int(val)
        except ValueError:
            return
            
        interval = input.decomp_interval()
        
        with ui.Progress(min=0, max=100) as p:
            p.set(5, message="Refreshing symbols...", detail=f"Fetching top {n_assets} high-volume assets")
            new_syms = manager.fetcher.get_top_volume_symbols(top_n=n_assets)
            syms = sorted(list(set(MANDATORY_CRYPTO).union(new_syms)))
            selected_symbols_decomp.set(set(syms))
            
            inventory = manager.get_inventory()
            sym_choices = sorted([s for s, ints in inventory.items() if interval in ints])
            ui.update_selectize("decomp_symbols", choices=sym_choices, selected=syms)

            if n_assets > 10:
                p.set(10, message="Syncing ticker data...", detail=f"Updating {len(syms)} assets")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(manager.load_data, s, interval, auto_sync=True) for s in syms]
                    for i, _ in enumerate(as_completed(futures)):
                        p.set(10 + int(90 * (i+1)/len(syms)), detail=f"Syncing {i+1}/{len(syms)}: {syms[i]}")
            p.set(100, message="Sync complete")

    # ══════════════════════════════════════════════════════════
    # TAB 2: DECOMPOSITION
    # ══════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.btn_run_decomp)
    def _():
        interval = input.decomp_interval()
        method = input.decomp_method()
        window = input.decomp_window()
        syms = list(input.decomp_symbols())

        if not syms:
            ui.notification_show("Please select at least one symbol", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            # 1. Check data availability
            p.set(10, message="Checking data...", detail=f"Preparing {len(syms)} assets")
            symbols = syms
            data_map = _load_return_data(symbols, interval, p)
            # Build correlation matrix as base for most methods

            df_aligned = pd.DataFrame(data_map).dropna()
            
            # Use aligned data for calculations
            corr = CorrelationEngine.calculate_matrix(data_map, method="pearson", window_size=window)
            corr_clean, _ = CorrelationEngine.filter_blanks(corr)

            step = 20
            step += 1
            p.set(step, message=f"Running {method}...")

            # Store aligned returns for context chart
            wide_df = pd.DataFrame(data_map)[-100:]
            wide_df = wide_df.dropna(axis=1, how="all")
            wide_df = wide_df.dropna(axis=0, how="all")
            wide_df = wide_df[corr_clean.columns]
            
            T = len(wide_df)
            N = len(corr_clean.columns)

            result = {
                'method': method,
                'returns': wide_df
            }

            if method == "eigen":
                k = input.n_components()
                result['data'] = DecompositionEngine.pca_decompose(wide_df, n_components=k)
                result['data']['mode'] = "Standard" # For card title compatibility

            elif method == "rmt":
                result['data'] = DecompositionEngine.spectral_filter_rmt(corr_clean, T, N)

            elif method == "kfactor":
                # k is now ignored by k_factor_decompose in favor of automated PC1 logic
                result['data'] = DecompositionEngine.k_factor_decompose(
                    wide_df, 0, mode=input.k_factor_mode()
                )

            elif method == "ica":
                n_comp = min(input.n_components(), N, T)
                result['data'] = DecompositionEngine.ica_decompose(wide_df, n_components=n_comp)

            elif method == "distance":
                result['data'] = DecompositionEngine.distance_matrix(corr_clean)

            elif method == "cluster":
                dist = DecompositionEngine.distance_matrix(corr_clean)
                result['data'] = DecompositionEngine.hierarchical_cluster(dist, method=input.linkage_method())
                result['dist'] = dist

            elif method == "mst":
                # dist = DecompositionEngine.distance_matrix(corr_clean)
                # result['data'] = DecompositionEngine.mst_spillover(dist)
                s_type = input.spillover_type()
                if s_type == "volatility":
                    vol = df_aligned.ewm(span=window, adjust=False).std().dropna()
                    result['data'] = DecompositionEngine.vol_spillover(vol)
                else:
                    result['data'] = DecompositionEngine.return_spillover(df_aligned)
                result['spillover_type'] = s_type

            result['corr'] = corr_clean
            p.set(100, message="Done")
            decomp_result.set(result)

    @render.ui
    def decomp_view():
        res = decomp_result.get()
        if res is None:
            return None

        method = res['method']
        cards = []

        if method == "eigen":
            cards.append(ui.card(
                ui.card_header("Market-Neutral PCA: Spectrum (PC1 = Market Factor)"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header(f"Market-Neutral Correlation ({res['data']['k']} factors after Market)"),
                output_widget("decomp_chart_2"),
                full_screen=True
            ))

        elif method == "rmt":
            d = res['data']
            cards.append(ui.card(
                ui.card_header(f"RMT Spectral Filter — {d['n_signal']} signal / {d['n_noise']} noise eigenvalues"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header("Denoised Correlation Matrix"),
                output_widget("decomp_chart_2"),
                full_screen=True
            ))

        elif method == "kfactor":
            mode_lbl = "Systematic (PC1)" if res['data']['mode'] == 'top' else "Idiosyncratic (PC2+)"
            cards.append(ui.card(
                ui.card_header(f"K-Factor Spectrum: {mode_lbl}"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header(f"Reconstructed Correlation Matrix ({mode_lbl})"),
                output_widget("decomp_chart_2"),
                full_screen=True
            ))

        elif method == "ica":
            cards.append(ui.card(
                ui.card_header(f"ICA Mixing Matrix — {res['data']['n_components']} components"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header("Independent Components (Sources)"),
                output_widget("decomp_chart_2"),
                full_screen=True
            ))

        elif method == "distance":
            cards.append(ui.card(
                ui.card_header("Distance Matrix D = √(2(1−ρ))"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))

        elif method == "cluster":
            d = res['data']
            cards.append(ui.card(
                ui.card_header(f"Hierarchical Clustering — {d['n_clusters']} clusters"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header("Clustered Correlation Matrix"),
                output_widget("decomp_chart_2"),
                full_screen=True
            ))

        elif method == "mst":
            s_type = res.get('spillover_type', 'volatility').capitalize()
            cards.append(ui.card(
                ui.card_header(f"{s_type} Spillover Network — {res['data'].get('n_edges', 0)} edges (Total Spillover Index: {res['data'].get('total_spillover_index', 0):.1f}%)"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header(f"{s_type} Spillover Centrality (Out vs In)"),
                output_widget("decomp_chart_2"),
                full_screen=True
            ))

        # ── Global Context Chart (Chart 3) ────────────────
        if method in ["eigen"]:
            cards.append(ui.card(
                ui.card_header("Cumulative Returns"),
                output_widget("decomp_chart_3"),
                full_screen=True
            ))

        return ui.div(*cards)

    # ── Decomposition Chart Renderers ─────────────────────────

    def _dark_layout(fig, height=500,  width=1470):
        # Sanitize all trace data to prevent JSON inf errors
        for trace in fig.data:
            for attr in ['x', 'y', 'z']:
                vals = getattr(trace, attr, None)
                if vals is not None:
                    try:
                        arr = np.array(vals, dtype=float)
                        arr = np.where(np.isfinite(arr), arr, 0)
                        setattr(trace, attr, arr.tolist())
                    except (ValueError, TypeError):
                        pass  # non-numeric data (e.g. string labels)

        layout = dict(
            template="plotly_dark",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            margin=dict(l=40, r=30, t=40, b=60),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            height=height
        )
        if width:
            layout['width'] = width
        fig.update_layout(**layout)
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.3)", linecolor="white", tickcolor="white", zerolinecolor="white")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.3)", linecolor="white", tickcolor="white", zerolinecolor="white")
        return fig

    @render_widget
    def decomp_chart_1():
        res = decomp_result.get()
        if res is None:
            return None

        method = res['method']
        d = res['data']

        # ── PCA/K-Factor Eigenvalue Spectrum ──────────────
        if method == "eigen" or method == "kfactor":
            n = len(d['eigenvalues'])
            k = d.get('k', n)
            mode = d.get('mode', 'top')
            
            # Determine bar colors based on selection
            colors = ["rgba(100,100,100,0.4)"] * n
            if method == 'eigen':
                # PC1 is Market Factor
                if n > 0:
                    colors[0] = "rgba(100,200,255,0.8)" # Light blue for market
                # PC2..PC(K+1) are selected systematic components
                for i in range(1, min(k+1, n)):
                    colors[i] = "rgba(255,165,0,0.9)"  # Orange
            elif mode == 'top':
                for i in range(min(k, n)):
                    colors[i] = "rgba(255,165,0,0.9)"
            else:
                for i in range(max(0, n-k), n):
                    colors[i] = "rgba(0,255,150,0.9)"

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=list(range(1, n + 1)),
                y=_sanitize(d['explained_ratio']),
                name="Explained Ratio",
                marker_color=colors
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=list(range(1, n + 1)),
                y=_sanitize(d['cumulative']),
                name="Cumulative",
                mode="lines+markers",
                line=dict(color="cyan", width=2),
                marker=dict(size=5)
            ), secondary_y=True)
            fig.update_xaxes(title_text="Eigenvalue Index")
            fig.update_yaxes(title_text="Explained Ratio", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 1.05])
            return _dark_layout(fig, height=450, width=1470)

        # ── RMT Eigenvalue Comparison ─────────────────────
        elif method == "rmt":
            raw = d['eigenvalues_raw']
            filtered = d['eigenvalues_filtered']
            n = len(raw)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, n + 1)), y=raw,
                name="Raw", marker_color="rgba(255,100,100,0.6)"
            ))
            fig.add_trace(go.Bar(
                x=list(range(1, n + 1)), y=filtered,
                name="Filtered", marker_color="rgba(0,200,255,0.8)"
            ))
            fig.add_hline(y=d['lambda_plus'], line_dash="dash", line_color="yellow",
                          annotation_text=f"λ+ = {d['lambda_plus']:.2f}")
            fig.add_hline(y=d['lambda_minus'], line_dash="dash", line_color="orange",
                          annotation_text=f"λ− = {d['lambda_minus']:.2f}")
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title_text="Eigenvalue Index")
            fig.update_yaxes(title_text="Eigenvalue")
            return _dark_layout(fig, height=450)

        # ── K-Factor Loadings Heatmap (Removed, using spectrum) ──
        elif method == "none": # Placeholder to skip
            pass

        # ── ICA Mixing Matrix ─────────────────────────────
        elif method == "ica":
            mixing = _sanitize(d['mixing'])
            fig = px.imshow(
                mixing, text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)

        # ── Distance Matrix Heatmap ───────────────────────
        elif method == "distance":
            fig = px.imshow(
                _sanitize(d), text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)

        elif method == "cluster":
            import plotly.figure_factory as ff
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import linkage

            corr = _sanitize(res['corr'])
            labels = d['labels']
            n_clusters = d.get('n_clusters', 1)

            dist_matrix = np.sqrt(2 * (1 - corr.values))
            dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1.0, neginf=0.0)

            fig = ff.create_dendrogram(
                dist_matrix,
                labels=labels,
                distfun=lambda x: x,
                linkagefun=lambda x: linkage(squareform(x, checks=False), method=input.linkage_method()),
                orientation='bottom',
                color_threshold=dist_matrix.max()
            )

            # Beautify lines
            for i, trace in enumerate(fig.data):
                trace.line.width = 2
                trace.line.color = 'orange'

            fig.update_layout(
                height=500,
                width=1470,
                template="plotly_dark",
                paper_bgcolor="#0b3d91",
                plot_bgcolor="#0b3d91",
                xaxis=dict(showticklabels=True, showgrid=False, mirror=False, tickangle=45, tickfont=dict(size=10, color='white')),
                yaxis=dict(
                    title_text="Distance",
                    showticklabels=False,
                    showgrid=False,
                    gridcolor="rgba(255,255,255,0.2)",
                    zeroline=False,
                    mirror=False
                ),
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=100)
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.2)", linecolor="white", tickcolor="white", zerolinecolor="white")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.2)", linecolor="white", tickcolor="white", zerolinecolor="white")
            fig.update_xaxes(constrain='domain')
            
            return fig

        # ── MST Network Graph ─────────────────────────────
        elif method == "mst":
            edges = d['edges']
            labels = d['labels']

            # Build adjacency for layout using spring-like positioning
            n = len(labels)
            label_idx = {l: i for i, l in enumerate(labels)}

            import networkx as nx

            # Build nx Graph for layout calculation
            G_layout = nx.Graph()
            G_layout.add_nodes_from(labels)
            for edge in edges:
                # Use inverse weight for distance if possible, or just connectivity
                G_layout.add_edge(edge['source'], edge['target'], weight=edge['weight'])

            # pos = nx.spring_layout(G_layout, k=1/np.sqrt(len(labels)), iterations=100, seed=42)
            pos = nx.spring_layout(G_layout, k=len(labels), iterations=100, seed=42)

            fig = go.Figure()

            # Draw edges
            if edges:
                max_w = max(e['weight'] for e in edges)
                min_w = min(e['weight'] for e in edges)
                range_w = max_w - min_w if max_w > min_w else 1.0
            else:
                max_w = min_w = 0

            annotations = []
            for edge in edges:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                
                # Scale width between 1.0 and 2.0
                norm_w = (edge['weight'] - min_w) / range_w
                width = 1.0 + 1.0 * norm_w
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=width, color="rgba(255,165,0,0.4)"),
                    hoverinfo="skip",
                    showlegend=False
                ))

                # Add directional arrow via annotation
                # We offset slightly so the arrowhead isn't buried in the node
                # The arrow points from source to target (directed spillover)
                annotations.append(dict(
                    ax=x0, ay=y0, axref='x', ayref='y',
                    x=x1, y=y1, xref='x', yref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=width,
                    arrowcolor="rgba(255,165,0,0.8)",
                    standoff=15, # offset from the target node center
                    startstandoff=10 # offset from source node
                ))

            # Draw nodes
            node_x = [pos[l][0] for l in labels]
            node_y = [pos[l][1] for l in labels]

            # Color and size by spillover influence
            spillover = d.get('spillover', {})
            in_spill = d.get('in_spillover', {})
            out_spill = d.get('out_spillover', {})
            colors = [spillover.get(l, 0) for l in labels]
            
            # Scale node size based on net influence magnitude
            spill_vals = np.array(list(spillover.values()))
            if len(spill_vals) > 0:
                abs_spill = np.abs(spill_vals)
                s_min, s_max = abs_spill.min(), abs_spill.max()
                s_range = s_max - s_min if s_max > s_min else 1.0
                # node_sizes = [10 + 20 * (abs(spillover.get(l, 0)) - s_min) / s_range for l in labels]
                raw_sizes = np.array([10 + 10 ** (1 + spillover.get(l, 0)) for l in labels])

                # Min-max normalize to [0,1]
                s_min, s_max = raw_sizes.min(), raw_sizes.max()
                s_range = s_max - s_min if s_max > s_min else 1.0
                node_sizes = (raw_sizes - s_min) / s_range
                node_sizes = node_sizes * 20 + 10

            else:
                node_sizes = [10] * len(labels)

            h_texts = [
                f"{l}<br>"
                f"Net: {spillover.get(l, 0):.2f}<br>"
                f"Out (Transmitted): {out_spill.get(l, 0):.2f}<br>"
                f"In (Received): {in_spill.get(l, 0):.2f}"
                for l in labels
            ]

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_sizes, color=colors,
                    colorscale="Spectral_r",
                    line=dict(width=1, color="white"),
                    showscale=True,
                    colorbar=dict(title="Net Spillover")
                    ),
                text=labels,
                textposition="top center",
                textfont=dict(size=10, color="white"),
                hovertext=h_texts,
                hoverinfo="text",
                showlegend=False
                ))

            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=annotations
            )
            return _dark_layout(fig, height=600)

        return go.Figure()

    @render_widget
    def decomp_chart_2():
        res = decomp_result.get()
        if res is None:
            return go.Figure()

        method = res['method']
        d = res['data']

        # ── RMT Denoised Matrix ───────────────────────────
        if method == "rmt":
            fig = px.imshow(
                _sanitize(d['denoised_matrix']), text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                zmin=-1, zmax=1,
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)

        # ── Eigen/K-Factor Reconstructed Correlation Matrix ──────────
        elif method == "kfactor" or method == "eigen":
            fig = px.imshow(
                _sanitize(d['reconstructed']), text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                zmin=-1, zmax=1,
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)

        # ── ICA Source Signals Time Series ────────────────
        elif method == "ica":
            sources = _sanitize(d['sources'])
            fig = go.Figure()
            for col in sources.columns:
                fig.add_trace(go.Scatter(
                    x=sources.index, y=sources[col],
                    mode="lines", name=col,
                    line=dict(width=1)
                ))
            fig.update_yaxes(title_text="Source Signal")
            return _dark_layout(fig, height=400)

        # ── Clustered Correlation Heatmap ─────────────────
        elif method == "cluster":
            from scipy.cluster.hierarchy import leaves_list
            corr = res['corr']
            d = res['data']
            Z = d['linkage_matrix']
            labels = d['labels']            

            # Use leaves_list to match dendrogram order
            order = leaves_list(Z)
            sorted_labels = [labels[i] for i in order]
            sorted_corr = corr.loc[sorted_labels, sorted_labels]

            fig = px.imshow(
                _sanitize(sorted_corr), text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                zmin=-1, zmax=1,
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)


        # ── MST Spillover Scatter (Out vs In) ──────────────
        if method == "mst":
            in_spill = d.get('in_spillover', {})
            out_spill = d.get('out_spillover', {})
            labels = d.get('labels', [])
            
            x_vals = [out_spill.get(l, 0) for l in labels]
            y_vals = [in_spill.get(l, 0) for l in labels]
            nets = [out_spill.get(l, 0) - in_spill.get(l, 0) for l in labels]
            
            # Replicate node size logic from chart 1 for consistency
            spill_vals = np.array(nets)
            if len(spill_vals) > 0:
                raw_sizes = np.array([10 + 10 ** (1 + (out_spill.get(l, 0) - in_spill.get(l, 0))) for l in labels])
                s_min, s_max = raw_sizes.min(), raw_sizes.max()
                s_range = s_max - s_min if s_max > s_min else 1.0
                node_sizes = (raw_sizes - s_min) / s_range
                node_sizes = node_sizes * 20 + 10
            else:
                node_sizes = [15] * len(labels)

            fig = go.Figure()
            
            # Diagonal line where Net = 0
            all_max = max(max(x_vals), max(y_vals)) if labels else 1.0
            fig.add_trace(go.Scatter(
                x=[0, all_max * 1.1], y=[0, all_max * 1.1],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.2)', dash='dash'),
                name='Net Zero Drivers Line',
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=nets,
                    colorscale="Spectral_r",
                    line=dict(width=1, color="white"),
                    showscale=True,
                    colorbar=dict(title="Net Spillover")
                ),
                text=labels,
                textposition="top center",
                hovertext=[
                    f"<b>{l}</b><br>Out (Transmitted): {out_spill.get(l, 0):.4f}<br>In (Received): {in_spill.get(l, 0):.4f}<br>Net Influence: {out_spill.get(l, 0) - in_spill.get(l, 0):.4f}"
                    for l in labels
                ],
                hoverinfo="text"
            ))
            
            fig.update_layout(
                xaxis_title="Out (Transmitted)",
                yaxis_title="In (Received)",
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50),
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)

        return go.Figure()

    @render_widget
    def decomp_chart_3():
        """Context chart: Cumulative reconstructed returns of the raw data used for analysis."""
        res = decomp_result.get()
        d = res['data']
        if d is None or 'reconstructed_returns' not in d:
            return go.Figure()

        returns = _sanitize(d['reconstructed_returns'])
        if returns.empty:
            return go.Figure()
        
        cumulative = (1 + returns).cumprod()
        cumulative.index = cumulative.index.strftime('%Y-%m-%d %H:%M')
        
        fig = go.Figure()
        for col in cumulative.columns:
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative[col],
                mode='lines',
                name=col,
                line=dict(width=1)
            ))
        fig.update_yaxes(title_text="Cumulative Log Return")
        return _dark_layout(fig, height=400, width=1470)
