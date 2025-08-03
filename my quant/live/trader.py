"""
live/trader.py

Unified forward-test/live simulation runner.
- Loads configuration and strategy logic.
- Connects to SmartAPI for live tick data (or mock data if in simulation).
- Processes incoming bar/tick data with your core strategy and position manager.
- Simulates all trades: never sends real orders.
"""

import time
import yaml
import logging
import importlib
from core.position_manager import PositionManager
from live.broker_adapter import BrokerAdapter
from utils.time_utils import now_ist

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_strategy(params):
    version = params.get("strategy_version", "live").lower()
    if version == "research":
        strat_module = importlib.import_module("core.researchStrategy")
    else:
        strat_module = importlib.import_module("core.liveStrategy")
    ind_mod = importlib.import_module("core.indicators")
    return strat_module.ModularIntradayStrategy(params, ind_mod)

class LiveTrader:
    def __init__(self, config_path: str = None, config_dict: dict = None):
        if config_dict is not None:
            config = config_dict
        else:
            config = load_config(config_path or "config/strategy_config.yaml")
        self.config = config
        strategy_params = config.get('strategy', config)
        risk_params = config.get('risk', {})
        session_params = config.get('session', {})
        instrument_params = config.get('instrument', {})
        capital = config.get('capital', {}).get('initial_capital', 100000)

        self.strategy = get_strategy(strategy_params)
        self.position_manager = PositionManager({
            **strategy_params, **risk_params,
            **instrument_params,
            "initial_capital": capital,
            "session": session_params,
        })
        self.broker = BrokerAdapter(config_path or "config/strategy_config.yaml")
        self.is_running = False
        self.active_position_id = None

    def start(self, run_once=False, result_box=None):
        self.is_running = True
        logger = logging.getLogger(__name__)
        self.broker.connect()
        logger.info("🟢 Forward testing session started.")
        try:
            while self.is_running:
                tick = self.broker.get_next_tick()
                if not tick:
                    time.sleep(0.1)
                    continue
                now = tick['timestamp'] if 'timestamp' in tick else now_ist()
                # Aggregate bars
                bars = self.broker.get_recent_bars(last_n=100)
                if bars.empty:
                    continue
                df_ind = self.strategy.calculate_indicators(bars)
                row = df_ind.iloc[-1]
                # Session end enforcement
                if hasattr(self.strategy, "should_exit_session"):
                    if self.strategy.should_exit_session(now):
                        self.close_position("Session End")
                        logger.info("Session end: all positions flattened.")
                        break
                # Entry logic
                if not self.active_position_id and getattr(self.strategy, "can_enter_new_position", lambda t: True)(now) \
                   and getattr(self.strategy, "should_enter_long", lambda r, n: False)(row, now):
                    self.active_position_id = self.strategy.open_long(row, now, self.position_manager)
                    if self.active_position_id:
                        qty = self.position_manager.positions[self.active_position_id].current_quantity
                        logger.info(f"[SIM] ENTERED LONG at ₹{row['close']} ({qty} contracts)")
                        if result_box:
                            result_box.config(state="normal")
                            result_box.insert("end", f"Simulated BUY: {qty} @ {row['close']:.2f}\n")
                            result_box.see("end")
                            result_box.config(state="disabled")
                # Position manager processes TP/SL/trail exits
                self.position_manager.process_positions(row, now)
                # Check if position closed (by SL/TP/Trailing/Session)
                if self.active_position_id and self.active_position_id not in self.position_manager.positions:
                    logger.info("Position closed (TP/SL/trailing/session).")
                    if result_box:
                        result_box.config(state="normal")
                        result_box.insert("end", f"Position closed at {row['close']:.2f}\n")
                        result_box.see("end")
                        result_box.config(state="disabled")
                    self.active_position_id = None
                if run_once:
                    self.is_running = False
        except KeyboardInterrupt:
            logger.info("Forward test interrupted by user.")
            self.close_position("Keyboard Interrupt")
        except Exception as e:
            logger.exception(f"Error in trading loop: {e}")
            self.close_position("Error Occurred")
        finally:
            self.broker.disconnect()
            logger.info("Session ended, data connection closed.")

    def close_position(self, reason: str = "Manual"):
        if self.active_position_id and self.active_position_id in self.position_manager.positions:
            last_price = self.broker.get_last_price()
            now = now_ist()
            self.position_manager.close_position_full(self.active_position_id, last_price, now, reason)
            logger = logging.getLogger(__name__)
            logger.info(f"[SIM] Position closed at {last_price} for reason: {reason}")
            self.active_position_id = None

if __name__ == "__main__":
    import argparse
    from utils.logging_utils import setup_logging, get_log_file_path
    log_file = get_log_file_path("forward_test")
    setup_logging(log_level="INFO", log_file=log_file)
    parser = argparse.ArgumentParser(description="Unified Forward Test Runner (Paper Trading)")
    parser.add_argument("--config", default="config/strategy_config.yaml", help="Config YAML path")
    args = parser.parse_args()

    bot = LiveTrader(args.config)
    bot.start()
