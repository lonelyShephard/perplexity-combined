"""
utils/config_loader.py

Centralized configuration loader for the unified trading system.
- Loads and validates YAML configuration files
- Provides default values and parameter validation
- Ensures consistent config access across all modules
- Supports environment variable overrides for sensitive data
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/strategy_config.yaml"

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation and defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If required parameters are missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    if not config:
        raise ValueError(f"Configuration file {config_path} is empty or invalid")
    
    # Apply environment variable overrides for sensitive data
    config = _apply_env_overrides(config)
    
    # Validate and apply defaults
    config = _validate_and_apply_defaults(config)
    
    logger.info(f"Configuration loaded successfully from {config_path}")
    return config

def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides for sensitive configuration.
    
    Environment variables follow pattern: TRADING_SECTION_KEY
    Example: TRADING_LIVE_API_KEY overrides config['live']['api_key']
    """
    # SmartAPI credentials from environment
    live_config = config.setdefault('live', {})
    
    env_mappings = {
        'TRADING_LIVE_API_KEY': ('live', 'api_key'),
        'TRADING_LIVE_CLIENT_CODE': ('live', 'client_code'),
        'TRADING_LIVE_PIN': ('live', 'pin'),
        'TRADING_LIVE_TOTP_SECRET': ('live', 'totp_secret'),
        'TRADING_INITIAL_CAPITAL': ('capital', 'initial_capital'),
    }
    
    for env_var, (section, key) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            if section not in config:
                config[section] = {}
            
            # Convert to appropriate type
            if key == 'initial_capital':
                config[section][key] = int(env_value)
            else:
                config[section][key] = env_value
            
            logger.info(f"Applied environment override for {section}.{key}")
    
    return config

def _validate_and_apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and apply default values where needed.
    """
    # Ensure all required sections exist
    required_sections = ['strategy', 'risk', 'capital', 'session', 'instrument']
    for section in required_sections:
        if section not in config:
            config[section] = {}
    
    # Apply strategy defaults
    strategy_defaults = {
        'use_ema_crossover': True,
        'use_macd': True,
        'use_vwap': True,
        'use_rsi_filter': False,
        'use_htf_trend': True,
        'use_bollinger_bands': False,
        'use_stochastic': False,
        'use_atr': True,
        'fast_ema': 9,
        'slow_ema': 21,
        'ema_points_threshold': 2,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_length': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'htf_period': 20,
        'strategy_version': 'live'
    }
    
    for key, default_value in strategy_defaults.items():
        config['strategy'].setdefault(key, default_value)
    
    # Apply risk management defaults
    risk_defaults = {
        'base_sl_points': 15,
        'tp_points': [10, 25, 50, 100],
        'tp_percents': [0.25, 0.25, 0.25, 0.25],
        'use_trail_stop': True,
        'trail_activation_points': 25,
        'trail_distance_points': 10,
        'risk_per_trade_percent': 1.0,
        'max_position_value_percent': 95,
        'commission_percent': 0.03,
        'commission_per_trade': 0.0,
        'stt_percent': 0.025,
        'exchange_charges_percent': 0.0019,
        'gst_percent': 18.0,
        'slippage_points': 1
    }
    
    for key, default_value in risk_defaults.items():
        config['risk'].setdefault(key, default_value)
    
    # Apply capital defaults
    capital_defaults = {
        'initial_capital': 100000
    }
    
    for key, default_value in capital_defaults.items():
        config['capital'].setdefault(key, default_value)
    
    # Apply session defaults
    session_defaults = {
        'is_intraday': True,
        'intraday_start_hour': 9,
        'intraday_start_min': 15,
        'intraday_end_hour': 15,
        'intraday_end_min': 15,
        'exit_before_close': 20,
        'timezone': 'Asia/Kolkata'
    }
    
    for key, default_value in session_defaults.items():
        config['session'].setdefault(key, default_value)
    
    # Apply instrument defaults
    instrument_defaults = {
        'symbol': 'NIFTY24DECFUT',
        'exchange': 'NSE_FO',
        'lot_size': 15,
        'tick_size': 0.05,
        'product_type': 'INTRADAY'
    }
    
    for key, default_value in instrument_defaults.items():
        config['instrument'].setdefault(key, default_value)
    
    # Apply live trading defaults
    live_defaults = {
        'paper_trading': True,  # Always default to simulation
        'exchange_type': 'NSE_FO',
        'feed_type': 'Quote',
        'log_ticks': False,
        'visual_indicator': True
    }
    
    config.setdefault('live', {})
    for key, default_value in live_defaults.items():
        config['live'].setdefault(key, default_value)
    
    # Validate critical parameters
    _validate_config_values(config)
    
    return config

def _validate_config_values(config: Dict[str, Any]) -> None:
    """
    Validate critical configuration values.
    
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    # Validate strategy parameters
    strategy = config['strategy']
    if strategy.get('fast_ema', 0) >= strategy.get('slow_ema', 0):
        errors.append("fast_ema must be less than slow_ema")
    
    if strategy.get('rsi_oversold', 0) >= strategy.get('rsi_overbought', 100):
        errors.append("rsi_oversold must be less than rsi_overbought")
    
    # Validate risk parameters
    risk = config['risk']
    if risk.get('risk_per_trade_percent', 0) <= 0:
        errors.append("risk_per_trade_percent must be positive")
    
    if risk.get('base_sl_points', 0) <= 0:
        errors.append("base_sl_points must be positive")
    
    # Validate capital
    capital = config['capital']
    if capital.get('initial_capital', 0) <= 0:
        errors.append("initial_capital must be positive")
    
    # Validate session
    session = config['session']
    start_hour = session.get('intraday_start_hour', 9)
    start_min = session.get('intraday_start_min', 15)
    end_hour = session.get('intraday_end_hour', 15)
    end_min = session.get('intraday_end_min', 15)
    
    start_time = start_hour * 60 + start_min
    end_time = end_hour * 60 + end_min
    
    if start_time >= end_time:
        errors.append("intraday_start_time must be before intraday_end_time")
    
    # Validate instrument
    instrument = config['instrument']
    if instrument.get('lot_size', 0) <= 0:
        errors.append("lot_size must be positive")
    
    if instrument.get('tick_size', 0) <= 0:
        errors.append("tick_size must be positive")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

def save_config(config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Output file path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        raise IOError(f"Failed to save configuration to {config_path}: {e}")

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'strategy.fast_ema')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        value = get_config_value(config, 'strategy.fast_ema', 9)
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'strategy.fast_ema')
        value: Value to set
        
    Example:
        set_config_value(config, 'strategy.fast_ema', 12)
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to parent dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def create_config_template(output_path: str = "config/strategy_config_template.yaml") -> None:
    """
    Create a template configuration file with all possible parameters.
    
    Args:
        output_path: Path for the template file
    """
    template = {
        'strategy': {
            'use_ema_crossover': True,
            'use_macd': True,
            'use_vwap': True,
            'use_rsi_filter': False,
            'use_htf_trend': True,
            'use_bollinger_bands': False,
            'use_stochastic': False,
            'use_atr': True,
            'fast_ema': 9,
            'slow_ema': 21,
            'ema_points_threshold': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_length': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'htf_period': 20,
            'strategy_version': 'live'
        },
        'risk': {
            'base_sl_points': 15,
            'tp_points': [10, 25, 50, 100],
            'tp_percents': [0.25, 0.25, 0.25, 0.25],
            'use_trail_stop': True,
            'trail_activation_points': 25,
            'trail_distance_points': 10,
            'risk_per_trade_percent': 1.0,
            'commission_percent': 0.03
        },
        'capital': {
            'initial_capital': 100000
        },
        'session': {
            'is_intraday': True,
            'intraday_start_hour': 9,
            'intraday_start_min': 15,
            'intraday_end_hour': 15,
            'intraday_end_min': 15,
            'exit_before_close': 20
        },
        'instrument': {
            'symbol': 'NIFTY24DECFUT',
            'exchange': 'NSE_FO',
            'lot_size': 15,
            'tick_size': 0.05
        },
        'live': {
            'paper_trading': True,
            'feed_type': 'Quote',
            'log_ticks': False
        }
    }
    
    save_config(template, output_path)
    print(f"Configuration template created at {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Create a sample config for testing
    test_config_path = "test_config.yaml"
    create_config_template(test_config_path)
    
    # Load and validate
    try:
        config = load_config(test_config_path)
        print("✅ Configuration loaded and validated successfully")
        
        # Test nested access
        fast_ema = get_config_value(config, 'strategy.fast_ema')
        print(f"Fast EMA: {fast_ema}")
        
        # Test modification
        set_config_value(config, 'strategy.fast_ema', 12)
        modified_ema = get_config_value(config, 'strategy.fast_ema')
        print(f"Modified Fast EMA: {modified_ema}")
        
        print("✅ Config loader test completed successfully")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_config_path):
            os.remove(test_config_path)
