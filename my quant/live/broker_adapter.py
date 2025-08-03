"""
live/broker_adapter.py

Unified SmartAPI broker/tick data adapter for live trading and forward test simulation.
- Handles SmartAPI login/session (via login.py/session manager).
- Streams live ticks via websocket or falls back to polling mode.
- Buffers ticks, generates 1-min OHLCV bars.
- Simulates orders in paper trading; never sends real orders.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import os

from utils.config_loader import load_config
from utils.time_utils import now_ist, normalize_datetime_to_ist

logger = logging.getLogger(__name__)

class BrokerAdapter:
    def __init__(self, config_path: str):
        config = load_config(config_path)
        self.params = config
        self.live_params = config.get("live", {})
        self.instrument = config.get("instrument", {})
        self.symbol = self.instrument.get("symbol", "")
        self.exchange = self.instrument.get("exchange", "NSE_FO")
        self.lot_size = self.instrument.get("lot_size", 1)
        self.tick_size = self.instrument.get("tick_size", 0.05)
        self.product_type = self.instrument.get("product_type", "INTRADAY")
        self.paper_trading = self.live_params.get("paper_trading", True)

        self.tick_buffer: List[Dict] = []
        self.df_tick = pd.DataFrame(columns=["timestamp", "price", "volume"])
        self.last_price: float = 0.0
        self.connection = None
        self.feed_active = False
        self.session_manager = None

        # Dynamic imports for SmartAPI
        try:
            from SmartApi import SmartConnect
            self.SmartConnect = SmartConnect
        except ImportError:
            self.SmartConnect = None
            logger.warning("SmartAPI not installed; running in simulated mode.")

    def connect(self):
        """Authenticate and establish live SmartAPI session (skips in sim mode)."""
        if self.paper_trading:
            logger.info("Paper trading mode: skipping SmartAPI live connection.")
            return
        if not self.SmartConnect:
            logger.warning("SmartAPI package missing. Cannot connect.")
            return
        try:
            # Use automatic login system if credentials are not in config
            live = self.live_params
            if not all(key in live for key in ["api_key", "client_code", "pin", "totp_token"]):
                # Use the session manager for automatic authentication
                from live.login import SmartAPISessionManager
                # Try to load existing session or prompt for login
                try:
                    # Check if session file exists and is valid
                    self.session_manager = SmartAPISessionManager("", "", "", "")
                    session_info = self.session_manager.load_session_from_file()
                    if session_info and self.session_manager.is_session_valid():
                        self.connection = self.session_manager.get_smartconnect()
                        self.feed_active = True
                        logger.info("Connected using existing SmartAPI session")
                        return
                    else:
                        logger.info("No valid session found. Running in paper trading mode.")
                        self.paper_trading = True
                        return
                except Exception as e:
                    logger.warning(f"Could not load SmartAPI session: {e}. Running in paper trading mode.")
                    self.paper_trading = True
                    return
            else:
                # Use credentials from config (legacy mode)
                self.connection = self.SmartConnect(api_key=live.get("api_key"))
                client_code = live.get("client_code")
                pin = live.get("pin")
                totp = live.get("totp_token")
                session = self.connection.generateSession(client_code, pin, totp)
                self.auth_token = session["data"]["jwtToken"]
                self.feed_token = self.connection.getfeedToken()
                self.feed_active = True
                logger.info(f"Connected to SmartAPI as {client_code}")
        except Exception as e:
            logger.error(f"Failed to connect to SmartAPI: {e}")
            logger.info("Falling back to paper trading mode")
            self.paper_trading = True

    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        """Fetch next tick—using SmartAPI polling, or simulated if in paper mode."""
        if self.paper_trading or not self.connection:
            # Simulate tick by quick micro-oscillation
            last = self.last_price or 22000.0
            direction = 1 if int(time.time() * 10) % 2 == 0 else -1
            price = last + direction * self.tick_size
            tick = {"timestamp": now_ist(), "price": price, "volume": 1500}
            self.last_price = price
            self._buffer_tick(tick)
            return tick
        try:
            ltp = self.connection.ltpData(self.exchange, self.symbol, self.instrument.get("instrument_token", ""))
            price = float(ltp["data"]["ltp"])
            tick = {"timestamp": now_ist(), "price": price, "volume": 1000}
            self.last_price = price
            self._buffer_tick(tick)
            return tick
        except Exception as e:
            logger.error(f"Error fetching SmartAPI LTP: {e}")
            return None

    def _buffer_tick(self, tick: Dict[str, Any]):
        """Buffer each tick and limit rolling window for memory safety."""
        self.tick_buffer.append(tick)
        self.df_tick = pd.concat([self.df_tick, pd.DataFrame([tick])], ignore_index=True)
        if len(self.df_tick) > 2500:
            self.df_tick = self.df_tick.iloc[-1000:]

    def get_recent_bars(self, last_n: int = 100) -> pd.DataFrame:
        """Aggregate tick buffer into most recent N 1-min OHLCV bars."""
        if self.df_tick.empty:
            now = now_ist()
            return pd.DataFrame({
                "open": [self.last_price],
                "high": [self.last_price],
                "low": [self.last_price],
                "close": [self.last_price],
                "volume": [0]
            }, index=[now])
        df = self.df_tick.copy()
        df.index = pd.to_datetime(df["timestamp"])
        df = df.sort_index()
        ohlc = df["price"].resample("1T").ohlc()
        volume = df["volume"].resample("1T").sum()
        bars = pd.concat([ohlc, volume], axis=1).dropna()
        bars.rename(columns={"sum": "volume"}, inplace=True)
        return bars.tail(last_n)

    def place_order(self, side: str, price: float, quantity: int, order_type: str = "MARKET") -> str:
        """Simulate all orders by default. Never sends real order in paper/forward test."""
        logger.info(f"Simulated order: {side} {quantity} @ {price} ({order_type})")
        return f"PAPER_{side}_{int(time.time())}"

    def get_last_price(self) -> float:
        """Return last known tick price (latest or simulated)."""
        return self.last_price or 0.0

    def disconnect(self):
        """Graceful SmartAPI logout (no effect in simulation)."""
        if self.connection and not self.paper_trading:
            try:
                client_code = self.live_params.get("client_code")
                self.connection.terminateSession(client_code)
                logger.info("Broker session terminated.")
            except Exception as e:
                logger.warning(f"Error shutting down session: {e}")
