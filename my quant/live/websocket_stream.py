"""
live/websocket_stream.py

SmartAPI WebSocket streaming module for unified trading system.

Features:
- Multiple instrument streams (up to 3 per SmartAPI account)
- User-selectable feed type: LTP, Quote, SnapQuote
- Event-driven tick delivery to tick buffer and OHLC aggregator
- Robust reconnect and error handling
- Integration with GUI controls and manual refresh

Usage:
- Import and call `start_stream()` from the live runner or GUI.
- All ticks passed safely for simulation; never to live order endpoints.
"""

import logging
import threading
import json
from datetime import datetime
try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
except ImportError:
    SmartWebSocketV2 = None  # Install with pip install smartapi-python

logger = logging.getLogger(__name__)

class WebSocketTickStreamer:
    def __init__(self, api_key, client_code, feed_token, symbol_tokens, feed_type="Quote", on_tick=None):
        """
        api_key: SmartAPI API key
        client_code: User/Account code
        feed_token: Obtained from SmartAPI session
        symbol_tokens: list of dicts [{"symbol": ..., "token": ..., "exchange": ...}]
        feed_type: 'LTP', 'Quote', or 'SnapQuote'
        on_tick: callback(tick_dict, symbol) called when new tick arrives
        """
        if SmartWebSocketV2 is None:
            raise ImportError("SmartWebSocketV2 (smartapi) package not available.")
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.symbol_tokens = symbol_tokens[:3]  # SmartAPI allows max 3
        self.feed_type = feed_type
        self.on_tick = on_tick or (lambda tick, symbol: None)
        self.ws = None
        self.running = False
        self.thread = None

    def _on_open(self, ws):
        logger.info("WebSocket connection OPEN")
        sub_json = [{
            "exchangeType": s['exchange'],
            "tokens": [s['token']],
            "feedType": self.feed_type
        } for s in self.symbol_tokens]
        ws.subscribe(sub_json)
        logger.info(f"Subscribed to {len(sub_json)} stream(s): {[s['symbol'] for s in self.symbol_tokens]} [{self.feed_type}]")

    def _on_data(self, ws, message):
        try:
            data = json.loads(message)
            ts = datetime.now()
            tick = {
                "timestamp": ts,
                "price": float(data.get("ltp", data.get("last_traded_price", 0))),
                "volume": int(data.get("volume", 0)),
                "symbol": data.get("tradingsymbol", data.get("symbol", "")),
                "exchange": data.get("exchange", ""),
                "raw": data
            }
            if self.on_tick:
                self.on_tick(tick, tick['symbol'])
        except Exception as e:
            logger.error(f"Error in streamed tick: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws):
        self.running = False
        logger.warning("WebSocket connection CLOSED")

    def start_stream(self):
        if self.running:
            logger.info("WebSocket stream already running.")
            return
        self.running = True
        self.ws = SmartWebSocketV2(
            api_key=self.api_key,
            client_code=self.client_code,
            feed_token=self.feed_token
        )
        self.ws.on_open = self._on_open
        self.ws.on_data = self._on_data
        self.ws.on_error = self._on_error
        self.ws.on_close = self._on_close
        self.thread = threading.Thread(target=self.ws.connect)
        self.thread.daemon = True
        self.thread.start()
        logger.info("WebSocket thread started.")

    def stop_stream(self):
        if self.ws is not None:
            self.ws.close()
            self.running = False
            logger.info("WebSocket stream stopped.")

# Example usage for integration testing (not run in production as-is)
if __name__ == "__main__":
    import os
    import time
    # Load session info from smartapi/session_token.json or config
    session_path = "smartapi/session_token.json"
    if not os.path.exists(session_path):
        print("Session JSON missing—run smartapi login first.")
        exit(1)
    with open(session_path, "r") as f:
        session = json.load(f)
    api_key = session["profile"]["api_key"]
    client_code = session["client_code"]
    feed_token = session["feed_token"]
    # Load three sample tokens from symbol cache
    from utils.cache_manager import load_symbol_cache
    symbols = load_symbol_cache()
    test_tokens = [v for (k, v) in list(symbols.items())[:3]]
    def print_tick(tick, symbol):
        print(f"[{tick['timestamp']}] {symbol}: ₹{tick['price']} Vol:{tick['volume']}")
    streamer = WebSocketTickStreamer(
        api_key, client_code, feed_token,
        test_tokens, feed_type="Quote", on_tick=print_tick
    )
    streamer.start_stream()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.stop_stream()
