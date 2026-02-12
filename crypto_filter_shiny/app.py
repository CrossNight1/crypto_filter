import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import src and ml_engine
sys.path.append(str(Path(__file__).parent))

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import faicons as fa

from shinywidgets import output_widget, render_widget
from src.config import APP_TITLE, APP_ICON, WELCOME_TITLE, WELCOME_TEXT, SIDEBAR_INFO, THEME, BG_COLOR, BINANCE_URL
import requests
import webbrowser

from modules.data_loader import data_loader_ui, data_loader_server
from modules.market_radar import market_radar_ui, market_radar_server
from modules.predictive import predictive_ui, predictive_server
from modules.multivariate_analysis import multivariate_analysis_ui, multivariate_analysis_server
from modules.activity_logs import activity_logs_ui, activity_logs_server
from modules.symbol_diagnostics import symbol_diagnostics_ui, symbol_diagnostics_server
from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import BENCHMARK_SYMBOL
from datetime import datetime

# UI definition
app_ui = ui.page_navbar(
    ui.head_content(
        ui.include_css(Path(__file__).parent / "www" / "custom.css"),
        ui.tags.script("""
            document.addEventListener("keydown", function(e) {
                const box = document.getElementById("quick_symbol");
                if (!box) return;
                if (document.activeElement === box && e.key === "Enter") {
                    e.preventDefault();
                    Shiny.setInputValue("quick_symbol_enter", Math.random(), {priority: "event"});
                }
            });
        """)
    ),

    ui.nav_panel("HOME", 
        ui.div(
            ui.div(
                ui.h1("QUANT_TOOLS", class_="main-header"),
                ui.HTML(
                    '<p class="lead mb-5" style="color:white;">> Institutional Predictive Analytics for '
                    '<span style="font-weight:bold ; font-size: 1.2em; text-decoration: underline; text-underline-offset: 0.5em; color:#FCC780;">BINANCE_PERPETUAL</span></p>'
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Diagnostics"),
                        ui.p("Deep Quantitative Symbol Diagnostics"),
                        ui.input_action_button("go_diagnostics", "open_diagnostics", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Market Radar"),
                        ui.p("Real-time relative performance & momentum tracking."),
                        ui.input_action_button("go_radar", "open_radar", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Multivariate"),
                        ui.p("Cross-asset correlation & Multivariate construction."),
                        ui.input_action_button("go_multivariate", "multivariate_analysis", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Machine Learning"),
                        ui.p("ML forecasting with meta-labeling verification protocols."),
                        ui.input_action_button("go_machine_learning", "explore_models", class_="btn-primary w-100")
                    ),
                    col_widths=[3,3,3,3]
                ),
                class_="hero-container text-center py-5"
            ),
            class_="container mt-5"
        )
    ),

    ui.nav_panel("DATA_LOADER", data_loader_ui()),
    ui.nav_panel("DIAGNOSTICS", symbol_diagnostics_ui()),
    ui.nav_panel("MARKET_RADAR", market_radar_ui()),
    ui.nav_panel("MULTIVARIATE", multivariate_analysis_ui()),
    ui.nav_panel("MACHINE_LEARNING", predictive_ui()),
    ui.nav_panel("ACTIVITY_LOGS", activity_logs_ui()),

    ui.nav_spacer(),
    ui.nav_control(ui.output_ui("data_status_")),
    ui.nav_spacer(),
    ui.nav_control(
        ui.div(
            ui.input_text("quick_symbol", None, placeholder="Enter Symbol (e.g. BTCUSDT) ", width="250px"),
            class_="d-flex align-items-center gap-2"
        )
    ),
    id="main_nav"
)


def server(input, output, session):
    # Shared global state if needed
    global_interval = reactive.Value("1d")
    manager = DataManager()
    engine = MetricsEngine()
    
    diag_data = reactive.Value({})
    data_info = reactive.Value({"global": {"oldest": "-", "latest": "-"}})
    
    def get_timestamps(symbol, interval):
        df = manager.load_data(symbol, interval)
        if df is not None and not df.empty and 'open_time' in df.columns:
            ts = pd.to_datetime(df['open_time'])
            return {"oldest": str(ts.min()), "latest": str(ts.max())}
        return {"oldest": "-", "latest": "-"}
    
    @reactive.Effect
    def populate_symbols():
        inventory = manager.get_inventory()
        all_syms = sorted(inventory.keys())
        ui.update_selectize("diag_symbol", choices=all_syms, server=True)
        
        # Set benchmark/global timestamps once
        global_ts = get_timestamps(BENCHMARK_SYMBOL, input.diag_interval())
        data_info.set({"global": global_ts})
    
    @render.ui
    def data_status_():
        d = data_info.get()
        return ui.HTML(f"""
            <div style="font-size: 0.7rem; opacity: 0.7; color: white;">
                <div>Global Data: {d['global']['oldest']} - {d['global']['latest']}</div>
                <div>Current Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        """)
    
    @reactive.Effect
    @reactive.event(input.go_machine_learning)
    def _go_ml():
        ui.update_navset("main_nav", selected="MACHINE_LEARNING")

    @reactive.Effect
    @reactive.event(input.go_radar)
    def _go_radar():
        ui.update_navset("main_nav", selected="MARKET_RADAR")

    @reactive.Effect
    @reactive.event(input.go_multivariate)
    def _go_multivariate():
        ui.update_navset("main_nav", selected="MULTIVARIATE")

    @reactive.Effect
    @reactive.event(input.go_diagnostics)
    def _go_diagnostics():
        ui.update_navset("main_nav", selected="DIAGNOSTICS")

    @reactive.Effect
    @reactive.event(input.quick_symbol_enter)
    def _quick_link():
        symbol = input.quick_symbol().strip().upper()
        if not symbol:
            # ui.notification_show("Enter a symbol", type="warning")
            return
        symbol = symbol.replace("USDT", "")
        symbol = f"{symbol}USDT"
        url = f"https://www.binance.com/en/futures/{symbol}"
        url_cg = "https://legend.coinglass.com/chart/7de78cdae4444cdb9a24cc33d1256a40"
        for u in [url]:
            webbrowser.open(u)
        ui.update_text("quick_symbol", value="")

    data_loader_server(input, output, session)
    market_radar_server(input, output, session, global_interval)
    predictive_server(input, output, session)
    multivariate_analysis_server(input, output, session)
    symbol_diagnostics_server(input, output, session, global_interval)
    activity_logs_server(input, output, session)

app = App(app_ui, server)
