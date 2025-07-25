"""
gui/parameter_gui.py - Unified Strategy Parameter GUI

Interactive interface for configuring strategy backtest parameters and triggering backtests.
Compatible with long-only, intraday use for equity or F&O symbols.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import os
import pandas as pd
from tabulate import tabulate

from backtest import run_backtest_from_file
from utils import create_results_directory, generate_timestamp_str

class StrategyParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Strategy Parameter GUI")
        self.root.geometry("700x900")

        # === Core Parameters ===
        self.data_file = tk.StringVar()
        self.init_capital = tk.IntVar(value=100000)
        self.exit_before_close = tk.IntVar(value=20)
        self.symbol = tk.StringVar(value="NIFTY24DECFUT")
        self.lot_size = tk.IntVar(value=15)
        self.tick_size = tk.DoubleVar(value=0.05)

        # === Indicator Toggles ===
        self.use_ema = tk.BooleanVar(value=True)
        self.use_macd = tk.BooleanVar(value=True)
        self.use_rsi = tk.BooleanVar(value=False)
        self.use_vwap = tk.BooleanVar(value=True)
        self.use_bb = tk.BooleanVar(value=False)
        self.use_htf = tk.BooleanVar(value=True)
        self.use_stoch = tk.BooleanVar(value=False)
        self.use_ma = tk.BooleanVar(value=False)
        self.use_atr = tk.BooleanVar(value=True)

        # === Indicator Parameters ===
        self.fast_ema = tk.IntVar(value=9)
        self.slow_ema = tk.IntVar(value=21)
        self.ema_thresh = tk.IntVar(value=2)

        self.macd_fast = tk.IntVar(value=12)
        self.macd_slow = tk.IntVar(value=26)
        self.macd_signal = tk.IntVar(value=9)

        self.rsi_len = tk.IntVar(value=14)
        self.rsi_oversold = tk.IntVar(value=30)
        self.rsi_overbought = tk.IntVar(value=70)

        self.bb_period = tk.IntVar(value=20)
        self.bb_std = tk.DoubleVar(value=2.0)

        self.htf_period = tk.IntVar(value=20)

        # === Risk Management ===
        self.base_sl = tk.IntVar(value=15)
        self.tp1 = tk.IntVar(value=10)
        self.tp2 = tk.IntVar(value=25)
        self.tp3 = tk.IntVar(value=50)
        self.tp4 = tk.IntVar(value=100)

        self.tp_percents = [tk.DoubleVar(value=0.25) for _ in range(4)]

        self.trail_enable = tk.BooleanVar(value=True)
        self.trail_activate = tk.IntVar(value=25)
        self.trail_dist = tk.IntVar(value=10)

        self.risk_perc = tk.DoubleVar(value=1.0)
        self.commission_pct = tk.DoubleVar(value=0.03)
        self.commission_per_trade = tk.DoubleVar(value=0.0)

        self._build_gui()

    def _build_gui(self):
        row = 0

        def check(label, var):
            ttk.Checkbutton(self.root, text=label, variable=var).grid(row=row, column=0, sticky="w")

        def entry(label, var, width=8):
            nonlocal row
            ttk.Label(self.root, text=label).grid(row=row, column=0, sticky="e")
            ttk.Entry(self.root, textvariable=var, width=width).grid(row=row, column=1)
            row += 1

        # === File Picker ===
        ttk.Label(self.root, text="Data File (.csv):").grid(row=row, column=0, sticky="e")
        ttk.Entry(self.root, textvariable=self.data_file, width=40).grid(row=row, column=1)
        ttk.Button(self.root, text="Browse", command=self.browse_file).grid(row=row, column=2)
        row += 1

        # === Indicators ===
        ttk.Label(self.root, text="Indicators:").grid(row=row, column=0, sticky="w"); row += 1
        for label, var in [
            ("EMA Crossover", self.use_ema),
            ("MACD", self.use_macd),
            ("RSI", self.use_rsi),
            ("VWAP", self.use_vwap),
            ("Bollinger Bands", self.use_bb),
            ("HTF Trend", self.use_htf),
            ("Stochastic", self.use_stoch),
            ("SMA/MA", self.use_ma),
            ("ATR", self.use_atr),
        ]:
            check(label, var); row += 1

        # === Indicator Params ===
        for label, var in [
            ("Fast EMA", self.fast_ema),
            ("Slow EMA", self.slow_ema),
            ("EMA Threshold", self.ema_thresh),
            ("MACD Fast", self.macd_fast),
            ("MACD Slow", self.macd_slow),
            ("MACD Signal", self.macd_signal),
            ("RSI Length", self.rsi_len),
            ("RSI Oversold", self.rsi_oversold),
            ("RSI Overbought", self.rsi_overbought),
            ("BB Period", self.bb_period),
            ("BB StdDev", self.bb_std),
            ("HTF EMA Period", self.htf_period),
        ]:
            entry(label, var)

        # === Risk Settings ===
        ttk.Label(self.root, text="RISK / SL / TP:").grid(row=row, column=0, sticky="w"); row += 1
        for label, var in [
            ("Base Stop (points)", self.base_sl),
            ("TP1 (pts)", self.tp1),
            ("TP2", self.tp2),
            ("TP3", self.tp3),
            ("TP4", self.tp4),
        ]:
            entry(label, var)

        for i, percent in enumerate(self.tp_percents):
            entry(f"TP{i+1} %", percent)

        ttk.Checkbutton(self.root, text="Enable Trailing SL", variable=self.trail_enable).grid(row=row, column=0, sticky="w"); row += 1
        entry("Trail Activation (pts)", self.trail_activate)
        entry("Trail Distance (pts)", self.trail_dist)

        # === Capital and Session ===
        entry("Risk % per Trade", self.risk_perc)
        entry("Commission %", self.commission_pct)
        entry("Commission per Trade", self.commission_per_trade)
        entry("Initial Capital", self.init_capital)
        entry("Exit N min before Close", self.exit_before_close)
        entry("Symbol", self.symbol)
        entry("Lot Size", self.lot_size)
        entry("Tick Size", self.tick_size)

        # === Backtest Button ===
        ttk.Button(self.root, text="Run Backtest", command=self.run_backtest).grid(row=row, column=0, columnspan=2, pady=10)

    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            self.data_file.set(file)

    def run_backtest(self):
        if not self.data_file.get().endswith(".csv"):
            messagebox.showerror("File Error", "Please select a .csv file")
            return

        # Build unified config dict
        params = {
            "strategy_version": "research",
            "symbol": self.symbol.get(),
            "lot_size": self.lot_size.get(),
            "tick_size": self.tick_size.get(),
            "initial_capital": self.init_capital.get(),
            "exit_before_close": self.exit_before_close.get(),

            "use_ema_crossover": self.use_ema.get(),
            "fast_ema": self.fast_ema.get(),
            "slow_ema": self.slow_ema.get(),
            "ema_points_threshold": self.ema_thresh.get(),

            "use_macd": self.use_macd.get(),
            "macd_fast": self.macd_fast.get(),
            "macd_slow": self.macd_slow.get(),
            "macd_signal": self.macd_signal.get(),

            "use_rsi_filter": self.use_rsi.get(),
            "rsi_length": self.rsi_len.get(),
            "rsi_oversold": self.rsi_oversold.get(),
            "rsi_overbought": self.rsi_overbought.get(),

            "use_vwap": self.use_vwap.get(),
            "use_bollinger_bands": self.use_bb.get(),
            "bb_period": self.bb_period.get(),
            "bb_std": self.bb_std.get(),
            "use_htf_trend": self.use_htf.get(),
            "htf_period": self.htf_period.get(),

            "base_sl_points": self.base_sl.get(),
            "tp_points": [self.tp1.get(), self.tp2.get(), self.tp3.get(), self.tp4.get()],
            "tp_percents": [v.get() for v in self.tp_percents],

            "use_trail_stop": self.trail_enable.get(),
            "trail_activation_points": self.trail_activate.get(),
            "trail_distance_points": self.trail_dist.get(),

            "risk_per_trade_percent": self.risk_perc.get(),
            "commission_percent": self.commission_pct.get(),
            "commission_per_trade": self.commission_per_trade.get(),
        }

        data_path = self.data_file.get()
        try:
            results, saved = run_backtest_from_file(data_path, params, data_type="csv")
            self.display_results(results)
        except Exception as e:
            messagebox.showerror("Backtest Error", f"Backtest failed:\n{e}")

    def display_results(self, results: dict):
        metrics = results.get("metrics", {})
        trades_df = results.get("trades_df", pd.DataFrame())

        summary = [
            ["Total Trades", metrics.get("total_trades", 0)],
            ["Win Rate (%)", f"{metrics.get('win_rate', 0):.2f}"],
            ["Net PnL", f"₹{metrics.get('net_pnl', 0):,.2f}"],
            ["Return (%)", f"{metrics.get('return_percent', 0):.2f}"],
            ["Best Trade ₹", f"{metrics.get('best_trade', 0):,.2f}"],
            ["Worst Trade ₹", f"{metrics.get('worst_trade', 0):,.2f}"],
            ["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"],
        ]

        print("\n=== STRATEGY SUMMARY ===")
        print(tabulate(summary, tablefmt="github"))

        if not trades_df.empty:
            path = os.path.join(create_results_directory(), f"trades_{generate_timestamp_str()}.csv")
            trades_df.to_csv(path, index=False)
            print(f"[✓] Trade log saved: {path}")

            print("\n=== SAMPLE TRADES ===")
            print(tabulate(trades_df.head(10), headers="keys", tablefmt="github"))
            messagebox.showinfo("Backtest Complete", f"Backtest completed. Details printed in console and saved.")

# Entry Point
if __name__ == "__main__":
    root = tk.Tk()
    app = StrategyParameterGUI(root)
    root.mainloop()
