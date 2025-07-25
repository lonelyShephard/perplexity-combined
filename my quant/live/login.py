"""
live/login.py

SmartAPI login and session management for the unified trading system.

- Handles API key, client code, PIN, TOTP for Angel One session creation.
- Provides method for authenticated SmartConnect client (for streaming/tick data).
- Designed for GUI- or CLI-driven login with explicit user control and safety.
- Saves and loads session tokens from disk to avoid repeated logins.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None
    logging.warning("SmartAPI client not installed. install with `pip install smartapi-python`.")

SESSION_FILE = "smartapi/session_token.json"

class SmartAPISessionManager:
    def __init__(self, api_key: str, client_code: str, pin: str, totp_token: str):
        self.api_key = api_key
        self.client_code = client_code
        self.pin = pin
        self.totp_token = totp_token
        self.session = None
        self.session_info = {}
        if SmartConnect is None:
            raise ImportError("SmartAPI client is not installed.")

    def login(self) -> dict:
        """
        Authenticate with SmartAPI, save JWT and feed token to SESSION_FILE.
        Returns:
            dict with tokens, may raise Exception on auth failure
        """
        smartapi = SmartConnect(api_key=self.api_key)
        try:
            res = smartapi.generateSession(self.client_code, self.pin, self.totp_token)
            jwt_token = res["data"]["jwtToken"]
            feed_token = smartapi.getfeedToken()
            profile = smartapi.getProfile(self.client_code)["data"]
            self.session_info = {
                "jwt_token": jwt_token,
                "feed_token": feed_token,
                "client_code": self.client_code,
                "refresh_time": datetime.now().isoformat(),
                "profile": profile,
            }
            # Ensure directory exists before writing file
            session_dir = os.path.dirname(SESSION_FILE)
            if session_dir and not os.path.exists(session_dir):
                os.makedirs(session_dir, exist_ok=True)
                logging.info(f"Created session directory: {session_dir}")
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(self.session_info, f, indent=2)
            logging.info(f"SmartAPI login successful for {self.client_code}. Session token saved.")
            self.session = smartapi
            return self.session_info
        except Exception as e:
            logging.error(f"SmartAPI login failed: {e}")
            raise

    def load_session_from_file(self) -> Optional[dict]:
        """
        Loads session tokens from local file if available.
        """
        if not os.path.exists(SESSION_FILE):
            logging.warning("No session token file found.")
            return None
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            self.session_info = json.load(f)
        logging.info(f"Loaded SmartAPI session from file (client {self.session_info.get('client_code')}).")
        return self.session_info

    def is_session_valid(self) -> bool:
        """
        Check if the loaded token is still valid (valid for 1 trading day).
        """
        # For strict robustness, always relogin daily or on session error
        info = self.session_info
        if not info:
            return False
        try:
            last_refresh = datetime.fromisoformat(info["refresh_time"])
            age = (datetime.now() - last_refresh).total_seconds() / 3600
            return age < 20  # Assume <20h = valid
        except Exception:
            return False

    def get_smartconnect(self):
        """
        Public API to get SmartConnect object with valid session.
        """
        if self.session is not None:
            return self.session
        elif self.load_session_from_file() and self.is_session_valid():
            smartapi = SmartConnect(api_key=self.api_key)
            smartapi.feed_token = self.session_info["feed_token"]
            self.session = smartapi
            return smartapi
        else:
            self.login()
            return self.session

# CLI entry point for manual login and session check
if __name__ == "__main__":
    import getpass
    print("SmartAPI Login Utility for Unified Trading System")
    api_key = input("API Key: ").strip()
    client_code = input("Client Code: ").strip()
    pin = getpass.getpass("PIN: ")
    totp_token = getpass.getpass("TOTP Token: ")
    mgr = SmartAPISessionManager(api_key, client_code, pin, totp_token)
    try:
        info = mgr.login()
        print("Login successful. Session token saved.")
        print(f"Feed Token: {info['feed_token']}")
        print(f"JWT Token: (hidden)")
    except Exception as e:
        print(f"Login failed: {e}")
