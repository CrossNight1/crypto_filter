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

                    ui.input_select("corr_interval", "Timeframe", choices=[]),

                    ui.input_numeric(
                        "window_size",
                        "Window Size",
                        value=50,
                        min=10,
                        max=2000
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
                        "input.decomp_method === 'kfactor' || input.decomp_method === 'ica' || input.decomp_method === 'eigen'",
                        ui.input_slider(
                            "n_components",
                            "Components (K)",
                            value=5,
                            min=1,
                            max=50,
                            step=1
                        ),
                    ),

                    ui.panel_conditional(
                        "input.decomp_method === 'kfactor' || input.decomp_method === 'eigen'",
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

                    ui.input_select("decomp_interval", "Timeframe", choices=[]),

                    ui.input_numeric(
                        "decomp_window",
                        "Window Size",
                        value=50,
                        min=10,
                        max=2000
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

    # ── Shared inventory updates ──────────────────────────────

    @reactive.effect
    def _():
        inventory = manager.get_inventory()
        if not inventory:
            return
        intervals = sorted(list(set(i for ivs in inventory.values() for i in ivs)))
        ui.update_select("corr_interval", choices=intervals)
        ui.update_select("decomp_interval", choices=intervals)

    @reactive.effect
    def _():
        interval = input.corr_interval()
        inventory = manager.get_inventory()
        if not interval or not inventory:
            return
        symbols = sorted([s for s, ints in inventory.items() if interval in ints])
        ui.update_selectize("corr_symbols", choices=symbols, selected=symbols[:50])

    @reactive.effect
    def _():
        interval = input.decomp_interval()
        inventory = manager.get_inventory()
        if not interval or not inventory:
            return
        symbols = sorted([s for s, ints in inventory.items() if interval in ints])
        ui.update_selectize("decomp_symbols", choices=symbols, selected=symbols[:50])

    @reactive.Effect
    def _():
        # Update slider max based on selected symbols
        syms = input.decomp_symbols()
        if syms:
            curr_max = len(syms)
            ui.update_slider("n_components", max=curr_max)
        
    # ── Helper: load return data ──────────────────────────────

    def _load_return_data(symbols, interval, progress=None):
        """Shared helper to load log-return series for symbols."""
        data_map = {}
        for i, sym in enumerate(symbols):
            df = manager.load_data(sym, interval)
            if df is not None and not df.empty:
                df = df.set_index("open_time")["close"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ret_series = np.log(df / df.shift(1))
                ret_series = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
                data_map[sym] = ret_series
            if progress:
                progress.set(i + 1)
        return data_map

    def _load_price_data(symbols, interval, progress=None):
        """Shared helper to load log-return series for symbols."""
        data_map = {}
        for i, sym in enumerate(symbols):
            df = manager.load_data(sym, interval)
            if df is not None and not df.empty:
                df = df.set_index("open_time")["close"]
                data_map[sym] = df
            if progress:
                progress.set(i + 1)
        return data_map

    # ══════════════════════════════════════════════════════════
    # TAB 1: MATRIX RADAR
    # ══════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.btn_gen_corr)
    def _():
        symbols = input.corr_symbols()
        interval = input.corr_interval()
        structure = input.dependence_structure()

        if not symbols or not interval:
            return

        with ui.Progress(min=0, max=len(symbols)) as p:
            p.set(message="Computing matrix...")
            data_map = _load_return_data(symbols, interval, p)

            if not data_map:
                return

            if structure == "Correlation":
                raw_matrix = CorrelationEngine.calculate_matrix(
                    data_map,
                    method=input.corr_method(),
                    window_size=input.window_size()
                )
            elif structure == "Covariance":
                raw_matrix = CorrelationEngine.calculate_covariance(
                    data_map,
                    window_size=input.window_size()
                )
            elif structure == "Cointegration":
                price = _load_price_data(symbols, interval, p)
                price = pd.DataFrame(price)
                raw_matrix = CorrelationEngine.calculate_coint_matrix(
                    price,
                    window_size=input.window_size()
                )
                raw_matrix = raw_matrix * -1
                
            elif structure == "Partial Correlation":
                raw_matrix = CorrelationEngine.calculate_partial_correlation(
                    data_map,
                    window_size=input.window_size()
                )
            elif structure == "Precision Matrix":
                raw_matrix = CorrelationEngine.calculate_precision_matrix(
                    data_map,
                    window_size=input.window_size()
                )

            filtered_matrix, _ = CorrelationEngine.filter_blanks(raw_matrix)
            filtered_matrix = filtered_matrix.sort_values(by=filtered_matrix.columns[0], ascending=False)
            correlation_matrix.set(filtered_matrix)

    @render.ui
    def matrix_view():
        df = correlation_matrix.get()
        if df.empty:
            print("Matrix is empty")
            return None
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
        return correlation_matrix.get()

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

        fig = px.imshow(
            _sanitize(df), text_auto=".2f", aspect="auto",
            color_continuous_scale="Spectral_r",
            zmin=zmin, zmax=zmax,
            template="plotly_dark"
        )
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

    # ══════════════════════════════════════════════════════════
    # TAB 2: DECOMPOSITION
    # ══════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.btn_run_decomp)
    def _():
        symbols = input.decomp_symbols()
        interval = input.decomp_interval()
        method = input.decomp_method()
        window = input.decomp_window()

        if not symbols or not interval:
            return

        total = len(symbols) + 3
        with ui.Progress(min=0, max=total) as p:
            p.set(0, message="Loading data...")
            data_map = _load_return_data(symbols, interval, p)

            if not data_map:
                ui.notification_show("No data loaded", type="error")
                return

            step = len(symbols)
            p.set(step, message="Computing correlation matrix...")
            # Build correlation matrix as base for most methods
            df_aligned = pd.DataFrame(data_map).dropna()
            
            # Use aligned data for calculations
            corr = df_aligned.corr(method="pearson") # calculate_matrix does this internally but we need the aligned df
            # Or assume CorrelationEngine handles it.
            # Let's stick to CorrelationEngine for the matrix 
            corr = CorrelationEngine.calculate_matrix(data_map, method="pearson", window_size=window)
            corr_clean, _ = CorrelationEngine.filter_blanks(corr)

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
                k = min(input.n_components(), N)
                result['data'] = DecompositionEngine.k_factor_decompose(
                    wide_df, k, mode=input.k_factor_mode()
                )
                evals = result['data']['eigenvalues']
                total = evals.sum()
                result['data']['explained_ratio'] = evals / total if total > 0 else evals * 0
                result['data']['cumulative'] = np.cumsum(result['data']['explained_ratio'])

            elif method == "rmt":
                result['data'] = DecompositionEngine.spectral_filter_rmt(corr_clean, T, N)

            elif method == "kfactor":
                k = min(input.n_components(), N)
                result['data'] = DecompositionEngine.k_factor_decompose(
                    wide_df, k, mode=input.k_factor_mode()
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
                dist = DecompositionEngine.distance_matrix(corr_clean)
                result['data'] = DecompositionEngine.mst_spillover(dist)

            result['corr'] = corr_clean
            step += 1
            p.set(step, message="Done")
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
                ui.card_header("Eigenvalue Spectrum & Explained Variance"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header(f"Reconstructed Correlation Matrix ({res['data']['mode']} {res['data']['k']} factors)"),
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
            d = res['data']
            cards.append(ui.card(
                ui.card_header(f"K={d['k']} Factor Loadings — {d['variance_captured']:.1%} variance captured"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header("Reconstructed Matrix"),
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
            cards.append(ui.card(
                ui.card_header(f"Spill Over Network — {res['data']['n_edges']} edges"),
                output_widget("decomp_chart_1"),
                full_screen=True
            ))
            cards.append(ui.card(
                ui.card_header("Spillover Centrality"),
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
            return go.Figure()

        method = res['method']
        d = res['data']

        # ── PCA Eigenvalue Spectrum ───────────────────────
        if method == "eigen":
            n = len(d['eigenvalues'])
            k = d.get('k', n)
            mode = d.get('mode', 'top')
            
            # Determine bar colors based on selection
            colors = ["rgba(100,100,100,0.4)"] * n
            if mode == 'top':
                for i in range(min(k, n)):
                    colors[i] = "rgba(255,165,0,0.9)"  # Orange for Systematic
            else:
                # Bottom mode: Keep last k components
                for i in range(max(0, n-k), n):
                    colors[i] = "rgba(0,255,150,0.9)"  # Green for Idiosyncratic

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

        # ── K-Factor Loadings Heatmap ─────────────────────
        elif method == "kfactor":
            loadings = _sanitize(d['loadings'])
            fig = px.imshow(
                loadings, text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)

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
                color_threshold=dist_matrix.max() * 0.7  # adjust to separate clusters
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
                xaxis=dict(showticklabels=True, showgrid=False, mirror=True, tickangle=45, tickfont=dict(size=10, color='white')),
                yaxis=dict(
                    title_text="Distance",
                    showticklabels=False,
                    showgrid=False,
                    gridcolor="rgba(255,255,255,0.2)",
                    zeroline=False,
                    mirror=True
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=150)
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.2)", linecolor="white", tickcolor="white", zerolinecolor="white")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.2)", linecolor="white", tickcolor="white", zerolinecolor="white")

            return fig

        # ── MST Network Graph ─────────────────────────────
        elif method == "mst":
            edges = d['edges']
            labels = d['labels']

            # Build adjacency for layout using spring-like positioning
            n = len(labels)
            label_idx = {l: i for i, l in enumerate(labels)}

            # Simple circular layout
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = {l: (np.cos(a), np.sin(a)) for l, a in zip(labels, angles)}

            # Force-directed refinement (simple spring)
            positions = np.column_stack([np.cos(angles), np.sin(angles)])
            for _ in range(len(labels)):
                for edge in edges:
                    i, j = label_idx[edge['source']], label_idx[edge['target']]
                    diff = positions[j] - positions[i]
                    dist = np.linalg.norm(diff) + 1e-9
                    target_dist = edge['weight']
                    force = (dist - target_dist) * 0.01
                    positions[i] += force * diff / dist
                    positions[j] -= force * diff / dist

            pos = {l: (positions[i, 0], positions[i, 1]) for i, l in enumerate(labels)}

            fig = go.Figure()

            # Draw edges
            for edge in edges:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=1.5, color="rgba(255,165,0,0.5)"),
                        hoverinfo="none",
                    showlegend=False
                ))

            # Draw nodes
            node_x = [pos[l][0] for l in labels]
            node_y = [pos[l][1] for l in labels]

            # Color by spillover
            spillover = d.get('spillover', {})
            colors = [spillover.get(l, 0) for l in labels]

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                marker=dict(
                    size=18, color=colors,
                    colorscale="Spectral_r",
                    line=dict(width=1, color="white"),
                    showscale=True,
                    colorbar=dict(title="Spillover")
                    ),
                text=labels,
                textposition="top center",
                textfont=dict(size=9, color="white"),
                hovertext=[f"{l} (spillover: {spillover.get(l, 0):.2f})" for l in labels],
                hoverinfo="text",
                showlegend=False
                ))

            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
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
            corr = res['corr']
            cluster_labels = d['cluster_labels']
            labels = d['labels']            

            order = np.argsort(cluster_labels)
            sorted_labels = [labels[i] for i in order]
            sorted_corr = corr.loc[sorted_labels, sorted_labels]

            fig = px.imshow(
                _sanitize(sorted_corr), text_auto=".2f", aspect="auto",
                color_continuous_scale="Spectral_r",
                zmin=-1, zmax=1,
                template="plotly_dark"
            )
            return _dark_layout(fig, height=500, width=1470)


        # ── MST Spillover Bar Chart ───────────────────────
        if method == "mst":
            spill = d.get('spillover', {})
            # Sort by spillover
            sorted_spill = sorted(spill.items(), key=lambda x: x[1], reverse=True)
            labels = [x[0] for x in sorted_spill]
            vals = [x[1] for x in sorted_spill]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=vals,
                name="Spillover Centrality",
                marker_color="rgba(100,200,250,0.8)"
            ))
            fig.update_xaxes(title_text="Asset")
            fig.update_yaxes(title_text="Spillover (Inverse Path Length)")
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
