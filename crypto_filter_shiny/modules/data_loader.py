from shiny import ui, render, reactive, session
import faicons as fa
import pandas as pd
from datetime import datetime, timedelta
from src.data import BinanceFuturesFetcher, DataManager
from src.config import AVAILABLE_INTERVALS, DEFAULT_FETCH_INTERVALS, MANDATORY_CRYPTO

def data_loader_ui():
    return ui.page_fluid(

        ui.row(

            ui.column(
                6,
                ui.card(
                    ui.card_header("Symbol Selection"),

                    ui.layout_columns(
                        ui.h6("Manual Adding Tickers"),
                        ui.input_text("manual_ticker", None, placeholder="BTCUSDT..."),
                        ui.input_action_button("btn_add", "Add", class_="btn-secondary w-100"),
                        col_widths=[4, 5, 3]
                    ),

                    ui.layout_columns(
                        ui.h6("Filtering Top Liquidity Tickers"),
                        ui.input_numeric("top_n", None, value=50, min=0, max=200, step=10),
                        ui.input_action_button("btn_filter", "Filter", class_="btn-secondary w-100"),
                        col_widths=[4, 5, 3]
                    ),

                    ui.layout_columns(
                        ui.output_ui("selection_info"),
                        ui.div(
                            ui.input_selectize("manage_list", None, choices=[], multiple=True),
                            style="max-height:120px; overflow-y:auto;"
                        )
                    ),

                    ui.input_action_button("btn_reset", "Reset", class_="btn-danger w-100")
                )
            ),

            ui.column(
                6,
                ui.card(
                    ui.card_header("Fetch Configuration"),

                    ui.layout_columns(
                        ui.h6("Intervals"),
                        ui.input_selectize(
                            "intervals",
                            None,
                            choices=AVAILABLE_INTERVALS,
                            selected=DEFAULT_FETCH_INTERVALS,
                            multiple=True
                        ),
                        col_widths=[3, 9]
                    ),

                    ui.layout_columns(
                        ui.input_radio_buttons("fetch_mode", None, ["Range", "Limit"], inline=True),
                        ui.panel_conditional(
                            "input.fetch_mode == 'Range'",
                            ui.input_numeric("days_back", None, value=30, min=1)
                        ),
                        ui.panel_conditional(
                            "input.fetch_mode == 'Limit'",
                            ui.input_numeric("limit", None, value=1000, min=100, step=100)
                        ),
                        col_widths=[2, 3, 3, 4]
                    ),

                    ui.input_action_button(
                        "btn_execute",
                        "Execute",
                        class_="btn-primary w-100 mt-1"
                    )
                )
            )
        ),
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Activity Logs"),
                    ui.output_code("fetch_logs"),
                    ui.output_ui("fetch_progress_ui")
                )
            )
        )
    )


def data_loader_server(input, output, session):
    fetcher = BinanceFuturesFetcher()
    manager = DataManager()
    
    selected_symbols = reactive.Value(set(MANDATORY_CRYPTO))
    logs = reactive.Value([])
    is_fetching = reactive.Value(False)
    progress = reactive.Value(0.0)

    @reactive.effect
    @reactive.event(input.btn_filter)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Fetching top symbols...", detail="Please wait")
            new_syms = fetcher.get_top_volume_symbols(top_n=input.top_n())
            selected_symbols.set(selected_symbols.get().union(new_syms))

    @reactive.effect
    @reactive.event(input.btn_add)
    def _():
        val = input.manual_ticker()
        if val:
            new_tickers = [s.strip().upper() for s in val.split(',') if s.strip()]
            selected_symbols.set(selected_symbols.get().union(new_tickers))
            ui.update_text("manual_ticker", value="")

    @reactive.effect
    @reactive.event(input.btn_reset)
    def _():
        selected_symbols.set(set(MANDATORY_CRYPTO))

    @reactive.effect
    def _():
        sorted_syms = sorted(list(selected_symbols.get()))
        ui.update_selectize("manage_list", choices=sorted_syms, selected=sorted_syms)

    @reactive.effect
    @reactive.event(input.manage_list)
    def _():
        current = set(input.manage_list())
        if current != selected_symbols.get():
            selected_symbols.set(current)

    @render.ui
    def selection_info():
        return ui.p(f"Active List: {len(selected_symbols.get())}")

    @render.code
    def fetch_logs():
        return "\n".join(logs.get()[-12:])

    @render.ui
    def fetch_progress_ui():
        if is_fetching.get():
            val = int(progress.get() * 100)
            return ui.div(
                ui.div(
                    ui.div(class_="progress-bar", role="progressbar", style=f"width: {val}%;", aria_valuenow=val, aria_valuemin=0, aria_valuemax=100),
                    class_="progress"
                ),
                class_="mt-2"
            )
        return ui.div()

    @reactive.effect
    @reactive.event(input.btn_execute)
    async def execute_sync():
        if not input.intervals():
            ui.notification_show("Select interval", type="error")
            return
        if not selected_symbols.get():
            ui.notification_show("Select symbols", type="error")
            return
            
        logs.set([])
        is_fetching.set(True)
        progress.set(0.0)
        
        all_syms = sorted(list(selected_symbols.get()))
        total = len(all_syms) * len(input.intervals())
        count_done = 0
        
        current_logs = []
        
        for sym in all_syms:
            for inter in input.intervals():
                now_str = datetime.now().strftime('%H:%M:%S')
                current_logs.append(f"[{now_str}] {sym} {inter}...")
                logs.set(current_logs[:])
                
                try:
                    if input.fetch_mode() == "Range":
                        end_t = datetime.now()
                        start_t = end_t - timedelta(days=input.days_back())
                        df = fetcher.fetch_history(sym, inter, start_time=start_t, end_time=end_t)
                    else:
                        df = fetcher.fetch_candles(sym, inter, limit=input.limit())
                    
                    if not df.empty:
                        manager.save_data(df, sym, inter)
                        current_logs.append(f"  > Saved {len(df)} candles")
                    else:
                        current_logs.append(f"  ! Missing data")
                except Exception as e:
                    current_logs.append(f"  ! Error: {str(e)}")
                
                count_done += 1
                progress.set(count_done / total)
                logs.set(current_logs[:])
                await reactive.flush()
        
        is_fetching.set(False)
        ui.notification_show("Sync Complete", type="message")
