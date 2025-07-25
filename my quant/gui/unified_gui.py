"""
gui/unified_gui.py

Unified GUI for both Backtest and Forward (Live) Test modes:
- Tab 1: Backtest mode (CSV selection, config, run backtest, show results)
- Tab 2: Forward Test mode (SmartAPI login, manual symbol cache refresh, symbol/feed select, start/stop simulated live)
- Tab 3: Status/Logs (view latest logs and info)

All orders are simulated; no real trading occurs in this GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from datetime import datetime
import logging
import yaml

from backtest.backtest_runner import run_backtest
from live.trader import LiveTrader
from utils.cache_manager import load_symbol_cache, refresh_symbol_cache, get_token_for_symbol

LOG_FILENAME = "unified_gui.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedTradingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unified Quantitative Trading System")
        self.geometry("950x700")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.tabControl = ttk.Notebook(self)
        self.tab_backtest = ttk.Frame(self.tabControl)
        self.tab_forward = ttk.Frame(self.tabControl)
        self.tab_status = ttk.Frame(self.tabControl)

        self.tabControl.add(self.tab_backtest, text="Backtest")
        self.tabControl.add(self.tab_forward, text="Forward Test")
        self.tabControl.add(self.tab_status, text="Status / Logs")
        self.tabControl.pack(expand=1, fill="both")

        self._build_backtest_tab()
        self._build_forward_test_tab()
        self._build_status_tab()

        self._backtest_thread = None
        self._forward_thread = None
        self.symbol_token_map = {}  # Initialize simple symbol-to-token mapping

    # --- Backtest Tab ---
    def _build_backtest_tab(self):
        frame = self.tab_backtest
        frame.columnconfigure(1, weight=1)
        row = 0
        
        ttk.Label(frame, text="Select Data CSV:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.bt_data_file = tk.StringVar()
        ttk.Entry(frame, textvariable=self.bt_data_file, width=55).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._bt_browse_csv).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # Strategy Configuration for Backtest
        ttk.Label(frame, text="Strategy Configuration", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky="w", pady=(15,5))
        row += 1

        # Indicator Toggles
        ttk.Label(frame, text="Indicators:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        row += 1
        
        bt_indicators_frame = ttk.Frame(frame)
        bt_indicators_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        self.bt_use_ema_crossover = tk.BooleanVar(value=True)
        self.bt_use_macd = tk.BooleanVar(value=True)
        self.bt_use_vwap = tk.BooleanVar(value=True)
        self.bt_use_rsi_filter = tk.BooleanVar(value=False)
        self.bt_use_htf_trend = tk.BooleanVar(value=True)
        self.bt_use_bollinger_bands = tk.BooleanVar(value=False)
        self.bt_use_stochastic = tk.BooleanVar(value=False)
        self.bt_use_atr = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(bt_indicators_frame, text="EMA Crossover", variable=self.bt_use_ema_crossover).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="MACD", variable=self.bt_use_macd).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="VWAP", variable=self.bt_use_vwap).grid(row=0, column=2, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="RSI Filter", variable=self.bt_use_rsi_filter).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="HTF Trend", variable=self.bt_use_htf_trend).grid(row=1, column=0, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="Bollinger Bands", variable=self.bt_use_bollinger_bands).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="Stochastic", variable=self.bt_use_stochastic).grid(row=1, column=2, sticky="w", padx=5)
        ttk.Checkbutton(bt_indicators_frame, text="ATR", variable=self.bt_use_atr).grid(row=1, column=3, sticky="w", padx=5)
        row += 1

        # Parameters
        ttk.Label(frame, text="Parameters:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=(10,2))
        row += 1
        
        bt_params_frame = ttk.Frame(frame)
        bt_params_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        # EMA Parameters
        ttk.Label(bt_params_frame, text="Fast EMA:").grid(row=0, column=0, sticky="e", padx=2)
        self.bt_fast_ema = tk.StringVar(value="9")
        ttk.Entry(bt_params_frame, textvariable=self.bt_fast_ema, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(bt_params_frame, text="Slow EMA:").grid(row=0, column=2, sticky="e", padx=2)
        self.bt_slow_ema = tk.StringVar(value="21")
        ttk.Entry(bt_params_frame, textvariable=self.bt_slow_ema, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(bt_params_frame, text="EMA Points Threshold:").grid(row=0, column=4, sticky="e", padx=2)
        self.bt_ema_points_threshold = tk.StringVar(value="2")
        ttk.Entry(bt_params_frame, textvariable=self.bt_ema_points_threshold, width=8).grid(row=0, column=5, padx=2)
        
        # MACD Parameters
        ttk.Label(bt_params_frame, text="MACD Fast:").grid(row=1, column=0, sticky="e", padx=2)
        self.bt_macd_fast = tk.StringVar(value="12")
        ttk.Entry(bt_params_frame, textvariable=self.bt_macd_fast, width=8).grid(row=1, column=1, padx=2)
        
        ttk.Label(bt_params_frame, text="MACD Slow:").grid(row=1, column=2, sticky="e", padx=2)
        self.bt_macd_slow = tk.StringVar(value="26")
        ttk.Entry(bt_params_frame, textvariable=self.bt_macd_slow, width=8).grid(row=1, column=3, padx=2)
        
        ttk.Label(bt_params_frame, text="MACD Signal:").grid(row=1, column=4, sticky="e", padx=2)
        self.bt_macd_signal = tk.StringVar(value="9")
        ttk.Entry(bt_params_frame, textvariable=self.bt_macd_signal, width=8).grid(row=1, column=5, padx=2)
        
        # RSI Parameters
        ttk.Label(bt_params_frame, text="RSI Length:").grid(row=2, column=0, sticky="e", padx=2)
        self.bt_rsi_length = tk.StringVar(value="14")
        ttk.Entry(bt_params_frame, textvariable=self.bt_rsi_length, width=8).grid(row=2, column=1, padx=2)
        
        ttk.Label(bt_params_frame, text="RSI Oversold:").grid(row=2, column=2, sticky="e", padx=2)
        self.bt_rsi_oversold = tk.StringVar(value="30")
        ttk.Entry(bt_params_frame, textvariable=self.bt_rsi_oversold, width=8).grid(row=2, column=3, padx=2)
        
        ttk.Label(bt_params_frame, text="RSI Overbought:").grid(row=2, column=4, sticky="e", padx=2)
        self.bt_rsi_overbought = tk.StringVar(value="70")
        ttk.Entry(bt_params_frame, textvariable=self.bt_rsi_overbought, width=8).grid(row=2, column=5, padx=2)
        
        # HTF Parameters
        ttk.Label(bt_params_frame, text="HTF Period:").grid(row=3, column=0, sticky="e", padx=2)
        self.bt_htf_period = tk.StringVar(value="20")
        ttk.Entry(bt_params_frame, textvariable=self.bt_htf_period, width=8).grid(row=3, column=1, padx=2)
        row += 1

        # Risk Management
        ttk.Label(frame, text="Risk Management:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=(10,2))
        row += 1
        
        bt_risk_frame = ttk.Frame(frame)
        bt_risk_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(bt_risk_frame, text="Stop Loss Points:").grid(row=0, column=0, sticky="e", padx=2)
        self.bt_base_sl_points = tk.StringVar(value="15")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_base_sl_points, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(bt_risk_frame, text="TP1 Points:").grid(row=0, column=2, sticky="e", padx=2)
        self.bt_tp1_points = tk.StringVar(value="10")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_tp1_points, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(bt_risk_frame, text="TP2 Points:").grid(row=0, column=4, sticky="e", padx=2)
        self.bt_tp2_points = tk.StringVar(value="25")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_tp2_points, width=8).grid(row=0, column=5, padx=2)
        
        ttk.Label(bt_risk_frame, text="TP3 Points:").grid(row=1, column=0, sticky="e", padx=2)
        self.bt_tp3_points = tk.StringVar(value="50")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_tp3_points, width=8).grid(row=1, column=1, padx=2)
        
        ttk.Label(bt_risk_frame, text="TP4 Points:").grid(row=1, column=2, sticky="e", padx=2)
        self.bt_tp4_points = tk.StringVar(value="100")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_tp4_points, width=8).grid(row=1, column=3, padx=2)
        
        self.bt_use_trail_stop = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_risk_frame, text="Use Trailing Stop", variable=self.bt_use_trail_stop).grid(row=1, column=4, columnspan=2, sticky="w", padx=5)
        
        ttk.Label(bt_risk_frame, text="Trail Activation Points:").grid(row=2, column=0, sticky="e", padx=2)
        self.bt_trail_activation_points = tk.StringVar(value="25")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_trail_activation_points, width=8).grid(row=2, column=1, padx=2)
        
        ttk.Label(bt_risk_frame, text="Trail Distance Points:").grid(row=2, column=2, sticky="e", padx=2)
        self.bt_trail_distance_points = tk.StringVar(value="10")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_trail_distance_points, width=8).grid(row=2, column=3, padx=2)
        
        ttk.Label(bt_risk_frame, text="Risk % per Trade:").grid(row=2, column=4, sticky="e", padx=2)
        self.bt_risk_per_trade_percent = tk.StringVar(value="1.0")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_risk_per_trade_percent, width=8).grid(row=2, column=5, padx=2)
        
        ttk.Label(bt_risk_frame, text="Initial Capital:").grid(row=3, column=0, sticky="e", padx=2)
        self.bt_initial_capital = tk.StringVar(value="100000")
        ttk.Entry(bt_risk_frame, textvariable=self.bt_initial_capital, width=12).grid(row=3, column=1, padx=2)
        row += 1

        ttk.Button(frame, text="Run Backtest", command=self._bt_run_backtest).grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        self.bt_result_box = tk.Text(frame, height=20, state="disabled", wrap="word")
        self.bt_result_box.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        frame.rowconfigure(row, weight=1)

    def _bt_browse_csv(self):
        path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        if path:
            self.bt_data_file.set(path)

    def _bt_run_backtest(self):
        if self._backtest_thread and self._backtest_thread.is_alive():
            messagebox.showinfo("Info", "Backtest already running.")
            return
        
        data_path = self.bt_data_file.get()
        
        if not os.path.isfile(data_path):
            messagebox.showerror("File Error", "Please select a valid CSV data file.")
            return
            
        self.bt_result_box.config(state="normal")
        self.bt_result_box.delete("1.0", "end")
        self.bt_result_box.insert("end", f"Running backtest on {os.path.basename(data_path)}...\n")
        self.bt_result_box.config(state="disabled")
        
        # Create config from GUI inputs (instead of relying on YAML file)
        gui_config = {
            "strategy": {
                "use_ema_crossover": self.bt_use_ema_crossover.get(),
                "use_macd": self.bt_use_macd.get(),
                "use_vwap": self.bt_use_vwap.get(),
                "use_rsi_filter": self.bt_use_rsi_filter.get(),
                "use_htf_trend": self.bt_use_htf_trend.get(),
                "use_bollinger_bands": self.bt_use_bollinger_bands.get(),
                "use_stochastic": self.bt_use_stochastic.get(),
                "use_atr": self.bt_use_atr.get(),
                "fast_ema": int(self.bt_fast_ema.get()),
                "slow_ema": int(self.bt_slow_ema.get()),
                "ema_points_threshold": int(self.bt_ema_points_threshold.get()),
                "macd_fast": int(self.bt_macd_fast.get()),
                "macd_slow": int(self.bt_macd_slow.get()),
                "macd_signal": int(self.bt_macd_signal.get()),
                "rsi_length": int(self.bt_rsi_length.get()),
                "rsi_oversold": int(self.bt_rsi_oversold.get()),
                "rsi_overbought": int(self.bt_rsi_overbought.get()),
                "htf_period": int(self.bt_htf_period.get()),
                "base_sl_points": int(self.bt_base_sl_points.get()),
                "risk_per_trade_percent": float(self.bt_risk_per_trade_percent.get())
            },
            "risk": {
                "base_sl_points": int(self.bt_base_sl_points.get()),
                "use_trail_stop": self.bt_use_trail_stop.get(),
                "trail_activation_points": int(self.bt_trail_activation_points.get()),
                "trail_distance_points": int(self.bt_trail_distance_points.get()),
                "tp_points": [int(self.bt_tp1_points.get()), int(self.bt_tp2_points.get()), 
                             int(self.bt_tp3_points.get()), int(self.bt_tp4_points.get())],
                "tp_percents": [0.25, 0.25, 0.25, 0.25],
                "risk_per_trade_percent": float(self.bt_risk_per_trade_percent.get()),
                "commission_percent": 0.1,
                "commission_per_trade": 0.0,
                "buy_buffer": 0
            },
            "capital": {
                "initial_capital": int(self.bt_initial_capital.get())
            },
            "session": {
                "is_intraday": True,
                "intraday_start_hour": 9,
                "intraday_start_min": 15,
                "intraday_end_hour": 15,
                "intraday_end_min": 15,
                "exit_before_close": 20
            },
            "backtest": {
                "max_drawdown_pct": 0,
                "allow_short": False,
                "close_at_session_end": True,
                "save_results": True,
                "results_dir": "backtest_results",
                "log_level": "INFO"
            },
            "instrument": {
                "symbol": "NIFTY",
                "exchange": "NSE_FO",
                "lot_size": 50,
                "tick_size": 0.05,
                "product_type": "INTRADAY"
            }
        }
        
        self._backtest_thread = threading.Thread(target=self._bt_worker, args=(gui_config, data_path))
        self._backtest_thread.start()

    def _bt_worker(self, config_dict, data_path):
        try:
            trades_df, metrics = run_backtest(config_dict, data_path)
            summary = (
                f"---- BACKTEST SUMMARY ----\n"
                f"Total Trades: {metrics['total_trades']}\n"
                f"Win Rate: {metrics['win_rate']:.2f}%\n"
                f"Total P&L: ₹{metrics['total_pnl']:.2f}\n"
                f"Avg Win: ₹{metrics['avg_win']:.2f}\n"
                f"Avg Loss: ₹{metrics['avg_loss']:.2f}\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                f"Max Win: ₹{metrics['max_win']:.2f}\n"
                f"Max Loss: ₹{metrics['max_loss']:.2f}\n"
                f"Total Commission: ₹{metrics['total_commission']:.2f}\n"
                f"Trade log written to 'backtest_trades.csv'\n"
            )
            self.bt_result_box.config(state="normal")
            self.bt_result_box.delete("1.0", "end")
            self.bt_result_box.insert("end", summary)
            self.bt_result_box.config(state="disabled")
        except Exception as e:
            self.bt_result_box.config(state="normal")
            self.bt_result_box.insert("end", f"\nBacktest failed: {e}\n")
            self.bt_result_box.config(state="disabled")

    # --- Forward Test Tab ---
    def _build_forward_test_tab(self):
        frame = self.tab_forward
        frame.columnconfigure(1, weight=1)
        row = 0

        # Symbol Cache Section
        ttk.Label(frame, text="Symbol Management", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10,5))
        row += 1
        
        # Authentication Status
        ttk.Label(frame, text="SmartAPI Authentication: Automatic (uses saved session)", font=('Arial', 9), foreground="blue").grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        ttk.Label(frame, text="Note: If no session exists, system will run in paper trading mode", font=('Arial', 8), foreground="gray").grid(row=row+1, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        row += 2
        
        ttk.Button(frame, text="Refresh Symbol Cache", command=self._ft_refresh_cache).grid(row=row, column=0, pady=3)
        self.ft_cache_status = tk.StringVar(value="Cache not loaded")
        ttk.Label(frame, textvariable=self.ft_cache_status).grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        row += 1

        ttk.Label(frame, text="Exchange:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        self.ft_exchange = tk.StringVar(value="NSE_FO")
        exchanges = ["NSE_FO", "NSE_CM", "BSE_CM"]
        ttk.Combobox(frame, textvariable=self.ft_exchange, values=exchanges, width=10, state='readonly').grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

        # Replace the combobox implementation with a listbox approach like angelalgo windsurf
        symbol_frame = ttk.Frame(frame)
        symbol_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        # Top row with label, entry and filter button
        ttk.Label(symbol_frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.ft_symbol = tk.StringVar()
        self.ft_symbol_entry = ttk.Entry(symbol_frame, textvariable=self.ft_symbol, width=32)
        self.ft_symbol_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(symbol_frame, text="Filter & Load Symbols", command=self._ft_load_symbols).grid(row=0, column=2, padx=5, pady=2)
        
        # Second row with listbox for filtered symbols
        self.ft_symbols_listbox = tk.Listbox(symbol_frame, width=50, height=6)
        self.ft_symbols_listbox.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        self.ft_symbols_listbox.bind("<<ListboxSelect>>", self._ft_update_symbol_details)
        
        # Add scrollbar to the listbox
        listbox_scrollbar = ttk.Scrollbar(symbol_frame, orient="vertical", command=self.ft_symbols_listbox.yview)
        listbox_scrollbar.grid(row=1, column=3, sticky="ns")
        self.ft_symbols_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        row += 2  # Increase row count to account for the listbox
        
        # Token field (auto-filled, read-only)
        ttk.Label(frame, text="Token:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        self.ft_token = tk.StringVar()
        ttk.Entry(frame, textvariable=self.ft_token, width=20, state="readonly").grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

        ttk.Label(frame, text="Feed Type:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        self.ft_feed_type = tk.StringVar(value="Quote")
        feed_types = ["LTP", "Quote", "SnapQuote"]
        ttk.Combobox(frame, textvariable=self.ft_feed_type, values=feed_types, width=12, state='readonly').grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

        # Strategy Configuration
        ttk.Label(frame, text="Strategy Configuration", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky="w", pady=(15,5))
        row += 1

        # Indicator Toggles
        ttk.Label(frame, text="Indicators:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        row += 1
        
        indicators_frame = ttk.Frame(frame)
        indicators_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        self.ft_use_ema_crossover = tk.BooleanVar(value=True)
        self.ft_use_macd = tk.BooleanVar(value=True)
        self.ft_use_vwap = tk.BooleanVar(value=True)
        self.ft_use_rsi_filter = tk.BooleanVar(value=False)
        self.ft_use_htf_trend = tk.BooleanVar(value=True)
        self.ft_use_bollinger_bands = tk.BooleanVar(value=False)
        self.ft_use_stochastic = tk.BooleanVar(value=False)
        self.ft_use_atr = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(indicators_frame, text="EMA Crossover", variable=self.ft_use_ema_crossover).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="MACD", variable=self.ft_use_macd).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="VWAP", variable=self.ft_use_vwap).grid(row=0, column=2, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="RSI Filter", variable=self.ft_use_rsi_filter).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="HTF Trend", variable=self.ft_use_htf_trend).grid(row=1, column=0, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="Bollinger Bands", variable=self.ft_use_bollinger_bands).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="Stochastic", variable=self.ft_use_stochastic).grid(row=1, column=2, sticky="w", padx=5)
        ttk.Checkbutton(indicators_frame, text="ATR", variable=self.ft_use_atr).grid(row=1, column=3, sticky="w", padx=5)
        row += 1

        # Parameters
        ttk.Label(frame, text="Parameters:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=(10,2))
        row += 1
        
        params_frame = ttk.Frame(frame)
        params_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        # EMA Parameters
        ttk.Label(params_frame, text="Fast EMA:").grid(row=0, column=0, sticky="e", padx=2)
        self.ft_fast_ema = tk.StringVar(value="9")
        ttk.Entry(params_frame, textvariable=self.ft_fast_ema, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(params_frame, text="Slow EMA:").grid(row=0, column=2, sticky="e", padx=2)
        self.ft_slow_ema = tk.StringVar(value="21")
        ttk.Entry(params_frame, textvariable=self.ft_slow_ema, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(params_frame, text="EMA Points Threshold:").grid(row=0, column=4, sticky="e", padx=2)
        self.ft_ema_points_threshold = tk.StringVar(value="2")
        ttk.Entry(params_frame, textvariable=self.ft_ema_points_threshold, width=8).grid(row=0, column=5, padx=2)
        
        # MACD Parameters
        ttk.Label(params_frame, text="MACD Fast:").grid(row=1, column=0, sticky="e", padx=2)
        self.ft_macd_fast = tk.StringVar(value="12")
        ttk.Entry(params_frame, textvariable=self.ft_macd_fast, width=8).grid(row=1, column=1, padx=2)
        
        ttk.Label(params_frame, text="MACD Slow:").grid(row=1, column=2, sticky="e", padx=2)
        self.ft_macd_slow = tk.StringVar(value="26")
        ttk.Entry(params_frame, textvariable=self.ft_macd_slow, width=8).grid(row=1, column=3, padx=2)
        
        ttk.Label(params_frame, text="MACD Signal:").grid(row=1, column=4, sticky="e", padx=2)
        self.ft_macd_signal = tk.StringVar(value="9")
        ttk.Entry(params_frame, textvariable=self.ft_macd_signal, width=8).grid(row=1, column=5, padx=2)
        
        # RSI Parameters
        ttk.Label(params_frame, text="RSI Length:").grid(row=2, column=0, sticky="e", padx=2)
        self.ft_rsi_length = tk.StringVar(value="14")
        ttk.Entry(params_frame, textvariable=self.ft_rsi_length, width=8).grid(row=2, column=1, padx=2)
        
        ttk.Label(params_frame, text="RSI Oversold:").grid(row=2, column=2, sticky="e", padx=2)
        self.ft_rsi_oversold = tk.StringVar(value="30")
        ttk.Entry(params_frame, textvariable=self.ft_rsi_oversold, width=8).grid(row=2, column=3, padx=2)
        
        ttk.Label(params_frame, text="RSI Overbought:").grid(row=2, column=4, sticky="e", padx=2)
        self.ft_rsi_overbought = tk.StringVar(value="70")
        ttk.Entry(params_frame, textvariable=self.ft_rsi_overbought, width=8).grid(row=2, column=5, padx=2)
        
        # HTF Parameters
        ttk.Label(params_frame, text="HTF Period:").grid(row=3, column=0, sticky="e", padx=2)
        self.ft_htf_period = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.ft_htf_period, width=8).grid(row=3, column=1, padx=2)
        row += 1

        # Risk Management
        ttk.Label(frame, text="Risk Management:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=(10,2))
        row += 1
        
        risk_frame = ttk.Frame(frame)
        risk_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(risk_frame, text="Stop Loss Points:").grid(row=0, column=0, sticky="e", padx=2)
        self.ft_base_sl_points = tk.StringVar(value="15")
        ttk.Entry(risk_frame, textvariable=self.ft_base_sl_points, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(risk_frame, text="TP1 Points:").grid(row=0, column=2, sticky="e", padx=2)
        self.ft_tp1_points = tk.StringVar(value="10")
        ttk.Entry(risk_frame, textvariable=self.ft_tp1_points, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(risk_frame, text="TP2 Points:").grid(row=0, column=4, sticky="e", padx=2)
        self.ft_tp2_points = tk.StringVar(value="25")
        ttk.Entry(risk_frame, textvariable=self.ft_tp2_points, width=8).grid(row=0, column=5, padx=2)
        
        ttk.Label(risk_frame, text="TP3 Points:").grid(row=1, column=0, sticky="e", padx=2)
        self.ft_tp3_points = tk.StringVar(value="50")
        ttk.Entry(risk_frame, textvariable=self.ft_tp3_points, width=8).grid(row=1, column=1, padx=2)
        
        ttk.Label(risk_frame, text="TP4 Points:").grid(row=1, column=2, sticky="e", padx=2)
        self.ft_tp4_points = tk.StringVar(value="100")
        ttk.Entry(risk_frame, textvariable=self.ft_tp4_points, width=8).grid(row=1, column=3, padx=2)
        
        self.ft_use_trail_stop = tk.BooleanVar(value=True)
        ttk.Checkbutton(risk_frame, text="Use Trailing Stop", variable=self.ft_use_trail_stop).grid(row=1, column=4, columnspan=2, sticky="w", padx=5)
        
        ttk.Label(risk_frame, text="Trail Activation Points:").grid(row=2, column=0, sticky="e", padx=2)
        self.ft_trail_activation_points = tk.StringVar(value="25")
        ttk.Entry(risk_frame, textvariable=self.ft_trail_activation_points, width=8).grid(row=2, column=1, padx=2)
        
        ttk.Label(risk_frame, text="Trail Distance Points:").grid(row=2, column=2, sticky="e", padx=2)
        self.ft_trail_distance_points = tk.StringVar(value="10")
        ttk.Entry(risk_frame, textvariable=self.ft_trail_distance_points, width=8).grid(row=2, column=3, padx=2)
        
        ttk.Label(risk_frame, text="Risk % per Trade:").grid(row=2, column=4, sticky="e", padx=2)
        self.ft_risk_per_trade_percent = tk.StringVar(value="1.0")
        ttk.Entry(risk_frame, textvariable=self.ft_risk_per_trade_percent, width=8).grid(row=2, column=5, padx=2)
        row += 1

        # Trading Controls
        ttk.Label(frame, text="Trading Controls", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=3, sticky="w", pady=(15,5))
        row += 1
        
        self.ft_paper_trading = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Simulation Mode (No Real Orders)", variable=self.ft_paper_trading, state="disabled").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(frame, text="Order execution is simulation-only.", foreground="red").grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        row += 2

        ttk.Button(frame, text="Start Forward Test", command=self._ft_run_forward_test).grid(row=row, column=0, pady=5)
        ttk.Button(frame, text="Stop", command=self._ft_stop_forward_test).grid(row=row, column=1, pady=5)
        row += 1

        self.ft_result_box = tk.Text(frame, height=16, width=110, state="disabled")
        self.ft_result_box.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        frame.rowconfigure(row, weight=1)

    def _ft_refresh_cache(self):
        try:
            count = refresh_symbol_cache()
            self.ft_cache_status.set(f"Cache refreshed: {count} symbols loaded.")
            messagebox.showinfo("Cache Refreshed", f"Symbol cache updated successfully. Loaded {count} symbols.")
        except Exception as e:
            error_msg = f"Cache refresh failed: {e}"
            self.ft_cache_status.set(error_msg)
            messagebox.showerror("Cache Error", f"Could not refresh cache: {e}")

    def _ft_load_symbols(self):
        """Load symbols exactly like angelalgo windsurf approach"""
        try:
            # Load the simple symbol:token mapping
            symbol_token_map = load_symbol_cache()
            
            if not symbol_token_map:
                messagebox.showwarning("No Symbols", "No symbols found. Try refreshing the cache first.")
                return
            
            # Store the mapping for later use
            self.symbol_token_map = symbol_token_map
            
            # Get text typed in symbol box
            typed_text = self.ft_symbol.get().upper()
            
            # Filter symbols matching the typed text
            if typed_text:
                # First try exact matches
                exact_matches = [s for s in symbol_token_map.keys() if s.upper() == typed_text]
                if exact_matches:
                    matching_symbols = exact_matches
                else:
                    # Then try contains matches
                    matching_symbols = [s for s in symbol_token_map.keys() if typed_text in s.upper()]
                    
                # Sort and limit results
                matching_symbols = sorted(matching_symbols)[:100]  # Limit to 100 for performance
            else:
                # If no text entered, show first 50 symbols as preview
                matching_symbols = sorted(symbol_token_map.keys())[:50]
                
            # Update dropdown with filtered symbols
            self.ft_symbols_listbox.delete(0, tk.END)  # Clear existing entries
            for symbol in matching_symbols:
                self.ft_symbols_listbox.insert(tk.END, symbol)
            
            # Show dropdown if there are matches
            if matching_symbols:
                self.ft_symbols_listbox.event_generate('<Down>')
                
            # Update status
            if typed_text:
                self.ft_cache_status.set(f"Found {len(matching_symbols)} symbols matching '{typed_text}'")
            else:
                self.ft_cache_status.set(f"Loaded {len(symbol_token_map)} symbols. Type to search...")
            
        except Exception as e:
            error_msg = f"Failed to load symbols: {e}"
            self.ft_cache_status.set(error_msg)
            logger.error(error_msg)

    def _ft_update_symbol_details(self, event=None):
        """Update token field when symbol is selected (exactly like angelalgo windsurf)"""
        try:
            selected_symbol = self.ft_symbols_listbox.get(self.ft_symbols_listbox.curselection())
            
            # Clear token first
            self.ft_token.set("")
            
            if hasattr(self, 'symbol_token_map') and selected_symbol in self.symbol_token_map:
                # Direct match found
                token = self.symbol_token_map[selected_symbol]
                self.ft_token.set(token)
                logger.info(f"Selected symbol: {selected_symbol}, Token: {token}")
            elif hasattr(self, 'symbol_token_map') and selected_symbol:
                # Try to find exact matches first
                exact_matches = [s for s in self.symbol_token_map.keys() if s.upper() == selected_symbol.upper()]
                if exact_matches:
                    exact_symbol = exact_matches[0]
                    token = self.symbol_token_map[exact_symbol]
                    self.ft_symbol.set(exact_symbol)  # Update to exact symbol name
                    self.ft_token.set(token)
                    logger.info(f"Auto-corrected to exact match: {exact_symbol}, Token: {token}")
                else:
                    # Try partial matches if no exact match found
                    matching_symbols = [s for s in self.symbol_token_map.keys() if selected_symbol.upper() in s.upper()]
                    if len(matching_symbols) == 1:
                        single_match = matching_symbols[0]
                        token = self.symbol_token_map[single_match]
                        self.ft_symbol.set(single_match)  # Update to matched symbol
                        self.ft_token.set(token)
                        logger.info(f"Auto-corrected to partial match: {single_match}, Token: {token}")
                    elif len(matching_symbols) > 1:
                        # Multiple matches - update dropdown with these matches
                        self.ft_symbols_listbox.delete(0, tk.END)  # Clear existing entries
                        for symbol in sorted(matching_symbols)[:100]:
                            self.ft_symbols_listbox.insert(tk.END, symbol)
                        self.ft_symbols_listbox.event_generate('<Down>')  # Show dropdown
                        logger.info(f"Found {len(matching_symbols)} matches for '{selected_symbol}'")
                    else:
                        logger.warning(f"Symbol '{selected_symbol}' not found in cache")
                        
        except Exception as e:
            logger.warning(f"Symbol details update error: {e}")
            self.ft_token.set("")

    def _ft_run_forward_test(self):
        if self._forward_thread and self._forward_thread.is_alive():
            messagebox.showinfo("Test Running", "Forward test already running.")
            return
        
        # Get symbol and token from the GUI (simple approach)
        selected_symbol = self.ft_symbol.get()
        token = self.ft_token.get()
        
        if not selected_symbol or not token:
            messagebox.showerror("Symbol Error", "Please select a valid symbol and load symbols first.")
            return
        
        self.ft_result_box.config(state="normal")
        self.ft_result_box.delete("1.0", "end")
        self.ft_result_box.insert("end", f"Starting forward test for {selected_symbol} (Token: {token}) - simulated orders only...\n")
        self.ft_result_box.config(state="disabled")
            
        # Create config dictionary for LiveTrader
        settings = {
            "live": {
                "paper_trading": True  # Always True for safety
            },
            "instrument": {
                "symbol": selected_symbol,
                "token": token,  # Include token for SmartAPI
                "exchange": self.ft_exchange.get(),
                "feed_type": self.ft_feed_type.get(),
                "lot_size": 1,
                "tick_size": 0.05
            },
            "strategy": {
                "use_ema_crossover": self.ft_use_ema_crossover.get(),
                "use_macd": self.ft_use_macd.get(),
                "use_vwap": self.ft_use_vwap.get(),
                "use_rsi_filter": self.ft_use_rsi_filter.get(),
                "use_htf_trend": self.ft_use_htf_trend.get(),
                "use_bollinger_bands": self.ft_use_bollinger_bands.get(),
                "use_stochastic": self.ft_use_stochastic.get(),
                "use_atr": self.ft_use_atr.get(),
                "fast_ema": int(self.ft_fast_ema.get()),
                "slow_ema": int(self.ft_slow_ema.get()),
                "ema_points_threshold": int(self.ft_ema_points_threshold.get()),
                "macd_fast": int(self.ft_macd_fast.get()),
                "macd_slow": int(self.ft_macd_slow.get()),
                "macd_signal": int(self.ft_macd_signal.get()),
                "rsi_length": int(self.ft_rsi_length.get()),
                "rsi_oversold": int(self.ft_rsi_oversold.get()),
                "rsi_overbought": int(self.ft_rsi_overbought.get()),
                "htf_period": int(self.ft_htf_period.get()),
                "base_sl_points": int(self.ft_base_sl_points.get()),
                "risk_per_trade_percent": float(self.ft_risk_per_trade_percent.get())
            },
            "risk": {
                "base_sl_points": int(self.ft_base_sl_points.get()),
                "use_trail_stop": self.ft_use_trail_stop.get(),
                "trail_activation_points": int(self.ft_trail_activation_points.get()),
                "trail_distance_points": int(self.ft_trail_distance_points.get()),
                "tp_points": [int(self.ft_tp1_points.get()), int(self.ft_tp2_points.get()), 
                             int(self.ft_tp3_points.get()), int(self.ft_tp4_points.get())],
                "tp_percents": [0.25, 0.25, 0.25, 0.25],
                "risk_per_trade_percent": float(self.ft_risk_per_trade_percent.get()),
                "commission_percent": 0.1,
                "commission_per_trade": 0.0,
                "buy_buffer": 0
            },
            "capital": {
                "initial_capital": 100000
            },
            "session": {
                "is_intraday": True,
                "intraday_start_hour": 9,
                "intraday_start_min": 15,
                "intraday_end_hour": 15,
                "intraday_end_min": 15,
                "exit_before_close": 20
            }
        }
        
        self.ft_result_box.config(state="normal")
        self.ft_result_box.delete("1.0", "end")
        self.ft_result_box.insert("end", f"Starting forward test for {selected_symbol} (Token: {token}) - simulated orders only...\n")
        self.ft_result_box.config(state="disabled")
        
        self._forward_thread = threading.Thread(target=self._ft_run_forward_worker, args=(settings,))
        self._forward_thread.start()

    def _ft_run_forward_worker(self, settings):
        try:
            trader = LiveTrader(config_dict=settings)
            trader.start(run_once=False, result_box=self.ft_result_box)
        except Exception as e:
            self.ft_result_box.config(state="normal")
            self.ft_result_box.insert("end", f"\nForward test failed: {e}\n")
            self.ft_result_box.config(state="disabled")

    def _ft_stop_forward_test(self):
        messagebox.showinfo("Stop", "Forward test stop functionality not yet implemented.")

    # --- Status Tab ---
    def _build_status_tab(self):
        frame = self.tab_status
        row = 0
        
        ttk.Label(frame, text="Unified Trading System Status & Logs", font=('Arial', 14, 'bold')).grid(row=row, column=0, sticky="w", pady=10)
        row += 1
        
        ttk.Label(frame, text=f"Log File: {LOG_FILENAME}").grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        
        ttk.Button(frame, text="View Log File", command=self._open_log_file).grid(row=row, column=0, sticky="w", pady=5)
        row += 1
        
        self.status_text = tk.Text(frame, height=25, width=110, state='disabled')
        self.status_text.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.rowconfigure(row, weight=1)
        frame.columnconfigure(0, weight=1)
        
        self._update_status_box()

    def _update_status_box(self):
        self.status_text.config(state="normal")
        self.status_text.delete("1.0", "end")
        
        status_info = (
            "UNIFIED TRADING SYSTEM STATUS\n"
            "=" * 50 + "\n\n"
            "✅ System Status: Ready\n"
            "🔒 Trading Mode: Simulation Only (No Real Orders)\n"
            "📊 Available Modes: Backtest & Forward Test\n\n"
            f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "BACKTEST MODE:\n"
            "- Load historical CSV data\n"
            "- Configure strategy parameters\n"
            "- Run complete simulation\n"
            "- View detailed results\n\n"
            "FORWARD TEST MODE:\n"
            "- Connect to SmartAPI for live data\n"
            "- Manual symbol cache management\n"
            "- Real-time tick processing\n"
            "- Simulated order execution\n\n"
            "Recent Activity:\n"
        )
        
        self.status_text.insert("end", status_info)
        
        # Add recent log entries
        try:
            if os.path.exists(LOG_FILENAME):
                with open(LOG_FILENAME, "r") as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                    for line in lines:
                        self.status_text.insert("end", line)
        except Exception:
            self.status_text.insert("end", "No recent log entries.\n")
            
        self.status_text.config(state="disabled")

    def _open_log_file(self):
        import subprocess
        import sys
        
        if os.path.exists(LOG_FILENAME):
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.call(('open', LOG_FILENAME))
            elif os.name == 'nt':  # Windows
                os.startfile(LOG_FILENAME)
            elif os.name == 'posix':  # Linux
                subprocess.call(('xdg-open', LOG_FILENAME))
        else:
            messagebox.showinfo("Log File", "No log file found yet.")

    def _on_close(self):
        if messagebox.askokcancel("Exit", "Do you want to quit?"):
            # Stop any running threads
            if self._backtest_thread and self._backtest_thread.is_alive():
                logger.info("Stopping backtest thread...")
            if self._forward_thread and self._forward_thread.is_alive():
                logger.info("Stopping forward test thread...")
            self.destroy()

if __name__ == "__main__":
    app = UnifiedTradingGUI()
    app.mainloop()
