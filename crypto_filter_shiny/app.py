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
from src.config import APP_TITLE, APP_ICON, WELCOME_TITLE, WELCOME_TEXT, SIDEBAR_INFO

from modules.data_loader import data_loader_ui, data_loader_server
from modules.market_radar import market_radar_ui, market_radar_server
from modules.predictive import predictive_ui, predictive_server
from modules.correlation import correlation_ui, correlation_server
from modules.activity_logs import activity_logs_ui, activity_logs_server

# UI definition
app_ui = ui.page_navbar(
    ui.head_content(
        ui.include_css(Path(__file__).parent / "www" / "custom.css")
    ),
    ui.nav_panel("Home", 
        ui.div(
            ui.h1(APP_ICON, class_="text-primary mb-4"),
            ui.markdown("### Welcome to the Advanced Metrics Dashboard"),
            ui.markdown("This application allows you to analyze Binance Perpetual Futures with advanced statistical metrics."),
            class_="container mt-5"
        )
    ),
    ui.nav_panel("Data Loader", data_loader_ui()),
    ui.nav_panel("Market Radar", market_radar_ui()),
    ui.nav_panel("Predictive", predictive_ui()),
    ui.nav_panel("Correlation", correlation_ui()),
    ui.nav_panel("Activity Logs", activity_logs_ui()),
    title=APP_ICON,
    id="main_nav",
    theme=ui.Theme("darkly"),
    navbar_options = ui.navbar_options(
        bg="#ffffff"
    )
)

def server(input, output, session):
    # Shared global state if needed
    global_interval = reactive.Value("1d")
    
    data_loader_server(input, output, session)
    market_radar_server(input, output, session, global_interval)
    predictive_server(input, output, session)
    correlation_server(input, output, session)
    activity_logs_server(input, output, session)

app = App(app_ui, server)
