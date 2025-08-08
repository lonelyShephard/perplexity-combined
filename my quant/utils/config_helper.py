import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigAccessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_strategy_param(self, key: str, default: Any = None) -> Any:
        return self.config.get('strategy', {}).get(key, default)

    def get_risk_param(self, key: str, default: Any = None) -> Any:
        return self.config.get('risk', {}).get(key, default)

    def get_capital_param(self, key: str, default: Any = None) -> Any:
        return self.config.get('capital', {}).get(key, default)

    def get_instrument_param(self, key: str, default: Any = None) -> Any:
        return self.config.get('instrument', {}).get(key, default)

    def get_session_param(self, key: str, default: Any = None) -> Any:
        return self.config.get('session', {}).get(key, default)

    def get_backtest_param(self, key: str, default: Any = None) -> Any:
        return self.config.get('backtest', {}).get(key, default)

    def validate_required_params(self) -> Dict[str, Any]:
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        required_sections = ['strategy', 'risk', 'capital', 'instrument', 'session']
        for section in required_sections:
            if section not in self.config:
                validation['errors'].append(f"Missing required section: {section}")
                validation['valid'] = False
        critical_params = [
            ('strategy', 'use_ema_crossover'),
            ('risk', 'base_sl_points'),
            ('capital', 'initial_capital'),
            ('session', 'intraday_start_hour')
        ]
        for section, param in critical_params:
            if section in self.config and param not in self.config[section]:
                validation['warnings'].append(f"Missing parameter: {section}.{param}")
        return validation

def create_nested_config_from_flat(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    nested_config = {
        'strategy': {},
        'risk': {},
        'capital': {},
        'instrument': {},
        'session': {},
        'backtest': {}
    }
    strategy_params = [
        'use_ema_crossover', 'use_macd', 'use_vwap', 'use_rsi_filter', 'use_htf_trend',
        'use_bollinger_bands', 'use_stochastic', 'use_atr', 'fast_ema', 'slow_ema',
        'macd_fast', 'macd_slow', 'macd_signal', 'rsi_length', 'rsi_overbought',
        'rsi_oversold', 'htf_period', 'strategy_version', 'indicator_update_mode'
    ]
    risk_params = [
        'base_sl_points', 'use_trail_stop', 'trail_activation_points', 
        'trail_distance_points', 'tp_points', 'tp_percents', 'risk_per_trade_percent',
        'commission_percent', 'commission_per_trade', 'buy_buffer'
    ]
    capital_params = ['initial_capital']
    instrument_params = ['symbol', 'exchange', 'lot_size', 'tick_size', 'product_type']
    session_params = [
        'intraday_start_hour', 'intraday_start_min', 'intraday_end_hour',
        'intraday_end_min', 'exit_before_close', 'timezone'
    ]
    backtest_params = [
        'max_drawdown_pct', 'allow_short', 'close_at_session_end',
        'save_results', 'results_dir', 'log_level'
    ]
    param_mapping = {
        'strategy': strategy_params,
        'risk': risk_params,
        'capital': capital_params,
        'instrument': instrument_params,
        'session': session_params,
        'backtest': backtest_params
    }
    for section, param_list in param_mapping.items():
        for param in param_list:
            if param in flat_config:
                nested_config[section][param] = flat_config[param]
    for section in nested_config.keys():
        if section in flat_config and isinstance(flat_config[section], dict):
            nested_config[section].update(flat_config[section])
    logger.info(f"Converted flat config to nested: {len(flat_config)} → {sum(len(v) for v in nested_config.values())} parameters")
    return nested_config