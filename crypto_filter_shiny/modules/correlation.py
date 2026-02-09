from shiny import ui, render, reactive
import faicons as fa
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
from src.data import DataManager
from ml_engine.analysis.correlation import CorrelationEngine

def correlation_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Correlation Matrix"),
            ui.input_action_button("btn_gen_corr", "Generate Matrix", class_="btn-primary w-100 mt-3"),
            ui.input_select("corr_interval", "Timeframe", choices=[]),
            ui.input_select("corr_method", "Method", choices=["pearson", "kendall", "spearman"]),
            ui.input_numeric("min_overlap", "Min Overlapping Samples", value=50, min=10, max=2000),
            ui.input_selectize("corr_symbols", "Select Symbols", choices=[], multiple=True),
            title="Correlation Settings"
        ),
        ui.output_ui("correlation_view")
    )

def correlation_server(input, output, session):
    manager = DataManager()
    
    correlation_matrix = reactive.Value(pd.DataFrame())

    @reactive.effect
    def _():
        inventory = manager.get_inventory()
        if not inventory: return
        intervals = sorted(list(set(i for ivs in inventory.values() for i in ivs)))
        ui.update_select("corr_interval", choices=intervals)

    @reactive.effect
    def _():
        interval = input.corr_interval()
        inventory = manager.get_inventory()
        symbols = sorted([s for s, ints in inventory.items() if interval in ints])
        ui.update_selectize("corr_symbols", choices=symbols, selected=symbols[:50])

    @reactive.effect
    @reactive.event(input.btn_gen_corr)
    def _():
        symbols = input.corr_symbols()
        interval = input.corr_interval()
        if not symbols: return
        
        with ui.Progress(min=0, max=len(symbols)) as p:
            p.set(message="Computing correlations...")
            data_map = {}
            for i, sym in enumerate(symbols):
                df = manager.load_data(sym, interval)
                if df is not None and not df.empty:
                    df = df.set_index('open_time')['close']
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ret_series = np.log(df / df.shift(1))
                    ret_series = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
                    data_map[sym] = ret_series
                p.set(i + 1)
            
            if not data_map: return
            
            raw_matrix = CorrelationEngine.calculate_matrix(data_map, method=input.corr_method(), min_periods=input.min_overlap())
            filtered_matrix, _ = CorrelationEngine.filter_blanks(raw_matrix)
            correlation_matrix.set(filtered_matrix)

    @render.ui
    def correlation_view():
        if correlation_matrix.get().empty:
            return ui.div(ui.h4("Select symbols and click 'Generate Matrix'"), class_="text-center mt-5")
        
        return ui.div(
            ui.card(
                ui.card_header("Correlation Matrix"),
                output_widget("corr_chart"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Raw Data"),
                ui.output_data_frame("corr_table")
            )
        )

    @render.data_frame
    def corr_table():
        return correlation_matrix.get()

    @render_widget
    def corr_chart():
        df = correlation_matrix.get()
        if df.empty: return go.Figure()
        
        fig = px.imshow(
            df, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template="plotly_dark"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        return fig
