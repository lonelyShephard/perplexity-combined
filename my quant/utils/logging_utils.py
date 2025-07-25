"""
utils/logging_utils.py

Centralized logging configuration and utilities for the unified trading system.

Features:
- Consistent logging format across all modules
- File and console logging with rotation
- Performance logging for backtests and live trading
- Thread-safe logging for GUI applications
- Configurable log levels per module
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading

# Global logging configuration
LOG_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO

# Thread lock for thread-safe logging configuration
_config_lock = threading.Lock()
_configured_loggers: Dict[str, logging.Logger] = {}

def get_log_file_path(base_name: str = "trading", 
                      log_dir: str = "logs", 
                      use_timestamp: bool = True) -> str:
    """
    Generate a standardized log file path.
    
    Args:
        base_name: Base name for the log file
        log_dir: Directory to store log files
        use_timestamp: Whether to include timestamp in filename
        
    Returns:
        Full path to log file
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.log"
    else:
        filename = f"{base_name}.log"
    
    return str(log_path / filename)

def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  console_output: bool = True,
                  file_rotation: bool = True,
                  max_file_size: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5,
                  module_name: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the trading system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        file_rotation: Whether to use rotating file handler
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup files to keep
        module_name: Specific module name for logger
        
    Returns:
        Configured logger instance
    """
    with _config_lock:
        # Determine logger name
        logger_name = module_name or "trading_system"
        
        # Return existing logger if already configured
        if logger_name in _configured_loggers:
            return _configured_loggers[logger_name]
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper(), DEFAULT_LOG_LEVEL))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            if file_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, 
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        logger.propagate = False
        
        # Cache configured logger
        _configured_loggers[logger_name] = logger
        
        logger.info(f"Logging configured for {logger_name}")
        return logger

def get_module_logger(module_name: str, 
                      log_level: str = "INFO",
                      log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific module with consistent configuration.
    
    Args:
        module_name: Name of the module
        log_level: Logging level for this module
        log_file: Optional separate log file for this module
        
    Returns:
        Configured logger for the module
    """
    return setup_logging(
        log_level=log_level,
        log_file=log_file,
        module_name=module_name
    )

def setup_performance_logging(log_dir: str = "logs/performance") -> logging.Logger:
    """
    Set up specialized logging for performance metrics.
    
    Args:
        log_dir: Directory for performance logs
        
    Returns:
        Performance logger instance
    """
    log_file = get_log_file_path("performance", log_dir, use_timestamp=True)
    
    perf_logger = setup_logging(
        log_level="INFO",
        log_file=log_file,
        console_output=False,
        module_name="performance"
    )
    
    return perf_logger

def setup_trade_logging(log_dir: str = "logs/trades") -> logging.Logger:
    """
    Set up specialized logging for trade execution.
    
    Args:
        log_dir: Directory for trade logs
        
    Returns:
        Trade logger instance
    """
    log_file = get_log_file_path("trades", log_dir, use_timestamp=True)
    
    trade_logger = setup_logging(
        log_level="INFO",
        log_file=log_file,
        console_output=True,
        module_name="trades"
    )
    
    return trade_logger

def setup_tick_logging(log_dir: str = "logs/ticks", 
                       log_file_name: str = "price_ticks.log") -> logging.Logger:
    """
    Set up specialized logging for price tick data.
    
    Args:
        log_dir: Directory for tick logs
        log_file_name: Name of the tick log file
        
    Returns:
        Tick logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_file_name)
    
    # Custom formatter for tick data
    tick_formatter = logging.Formatter("%(asctime)s,%(message)s", "%Y-%m-%d %H:%M:%S.%f")
    
    tick_logger = logging.getLogger("tick_data")
    tick_logger.setLevel(logging.INFO)
    tick_logger.handlers.clear()
    
    # File handler for tick data
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(tick_formatter)
    tick_logger.addHandler(file_handler)
    tick_logger.propagate = False
    
    return tick_logger

class PerformanceTimer:
    """Context manager for timing operations and logging performance."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger("performance")
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}")
    
    def get_duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

def log_system_info(logger: logging.Logger):
    """Log system information for debugging purposes."""
    import platform
    import psutil
    
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {psutil.cpu_count()} cores")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

def log_trade_execution(logger: logging.Logger, 
                       action: str, 
                       symbol: str, 
                       quantity: int, 
                       price: float, 
                       timestamp: datetime,
                       reason: str = "",
                       order_id: str = ""):
    """
    Log trade execution in a standardized format.
    
    Args:
        logger: Logger instance
        action: Trade action (BUY/SELL)
        symbol: Trading symbol
        quantity: Quantity traded
        price: Execution price
        timestamp: Execution timestamp
        reason: Reason for trade
        order_id: Order ID if available
    """
    log_msg = (
        f"TRADE_EXECUTION | "
        f"Action: {action} | "
        f"Symbol: {symbol} | "
        f"Qty: {quantity} | "
        f"Price: {price:.2f} | "
        f"Time: {timestamp} | "
        f"Reason: {reason} | "
        f"OrderID: {order_id}"
    )
    logger.info(log_msg)

def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of performance metrics
    """
    logger.info("=== PERFORMANCE METRICS ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=== END METRICS ===")

def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """
    Clean up log files older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days of logs to retain
    """
    import time
    
    if not os.path.exists(log_dir):
        return
    
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        print(f"Removed old log file: {file_path}")
                    except OSError as e:
                        print(f"Error removing {file_path}: {e}")

# Convenience functions for common logging scenarios
def log_backtest_start(logger: logging.Logger, config: Dict[str, Any]):
    """Log backtest start with configuration."""
    logger.info("=== BACKTEST STARTED ===")
    logger.info(f"Strategy: {config.get('strategy', {}).get('name', 'Unknown')}")
    logger.info(f"Capital: ₹{config.get('capital', {}).get('initial_capital', 0):,}")
    logger.info(f"Data source: {config.get('data_source', 'Unknown')}")

def log_backtest_end(logger: logging.Logger, results: Dict[str, Any]):
    """Log backtest completion with key results."""
    logger.info("=== BACKTEST COMPLETED ===")
    logger.info(f"Total trades: {results.get('total_trades', 0)}")
    logger.info(f"Win rate: {results.get('win_rate', 0):.2f}%")
    logger.info(f"Total P&L: ₹{results.get('total_pnl', 0):.2f}")
    logger.info(f"Profit factor: {results.get('profit_factor', 0):.2f}")

def log_live_session_start(logger: logging.Logger, config: Dict[str, Any]):
    """Log live trading session start."""
    logger.info("=== LIVE SESSION STARTED ===")
    logger.info(f"Symbol: {config.get('instrument', {}).get('symbol', 'Unknown')}")
    logger.info(f"Exchange: {config.get('instrument', {}).get('exchange', 'Unknown')}")
    logger.info(f"Paper trading: {config.get('live', {}).get('paper_trading', True)}")

def log_live_session_end(logger: logging.Logger, session_stats: Dict[str, Any]):
    """Log live trading session end with statistics."""
    logger.info("=== LIVE SESSION ENDED ===")
    logger.info(f"Session duration: {session_stats.get('duration', 'Unknown')}")
    logger.info(f"Trades executed: {session_stats.get('trades', 0)}")
    logger.info(f"Session P&L: ₹{session_stats.get('pnl', 0):.2f}")

# Example usage and testing
if __name__ == "__main__":
    # Test basic logging setup
    logger = setup_logging(
        log_level="INFO",
        log_file="logs/test_logging.log",
        console_output=True
    )
    
    logger.info("Testing logging configuration")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance timer
    with PerformanceTimer("test_operation", logger) as timer:
        import time
        time.sleep(1)  # Simulate work
    
    print(f"Operation took {timer.get_duration():.3f} seconds")
    
    # Test specialized loggers
    trade_logger = setup_trade_logging()
    trade_logger.info("Test trade log message")
    
    tick_logger = setup_tick_logging()
    tick_logger.info("22000.50,1500")  # Sample tick data
    
    # Test system info logging
    log_system_info(logger)
    
    print("✅ Logging utilities test completed successfully!")
