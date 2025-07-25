"""
utils/time_utils.py

Time and session-related utilities for the unified trading system.

Features:
- Market session management and timing
- Timezone-aware datetime operations
- Trading session validation
- Intraday trading time calculations
- Market holiday and weekend handling
"""

import pytz
from datetime import datetime, time, timedelta, date
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Indian market timezone
IST = pytz.timezone('Asia/Kolkata')

# Market session defaults
DEFAULT_MARKET_OPEN = time(9, 15)  # 9:15 AM
DEFAULT_MARKET_CLOSE = time(15, 30)  # 3:30 PM
DEFAULT_PRE_MARKET_OPEN = time(9, 0)  # 9:00 AM
DEFAULT_POST_MARKET_CLOSE = time(15, 40)  # 3:40 PM

def now_kolkata() -> datetime:
    """Get current time in IST/Asia-Kolkata timezone."""
    return datetime.now(IST)

def to_kolkata(dt: datetime) -> datetime:
    """Convert datetime to IST timezone."""
    if dt.tzinfo is None:
        # Assume UTC if no timezone info
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(IST)

def format_timestamp(dt: datetime, include_timezone: bool = False) -> str:
    """
    Format datetime for file/log naming.
    
    Args:
        dt: Datetime to format
        include_timezone: Whether to include timezone in output
        
    Returns:
        Formatted timestamp string
    """
    if include_timezone:
        return dt.strftime("%Y%m%d_%H%M%S_%Z")
    return dt.strftime("%Y%m%d_%H%M%S")

def get_market_session_times(trading_date: date, 
                            open_time: time = DEFAULT_MARKET_OPEN,
                            close_time: time = DEFAULT_MARKET_CLOSE) -> Tuple[datetime, datetime]:
    """
    Get market open and close times for a given trading date.
    
    Args:
        trading_date: Date for which to get session times
        open_time: Market opening time (default 9:15 AM)
        close_time: Market closing time (default 3:30 PM)
        
    Returns:
        Tuple of (market_open_datetime, market_close_datetime) in IST
    """
    market_open = IST.localize(datetime.combine(trading_date, open_time))
    market_close = IST.localize(datetime.combine(trading_date, close_time))
    
    return market_open, market_close

def is_market_session(current_time: Optional[datetime] = None,
                     open_time: time = DEFAULT_MARKET_OPEN,
                     close_time: time = DEFAULT_MARKET_CLOSE) -> bool:
    """
    Check if current time is within market trading session.
    
    Args:
        current_time: Time to check (default: current IST time)
        open_time: Market opening time
        close_time: Market closing time
        
    Returns:
        True if within trading session
    """
    if current_time is None:
        current_time = now_kolkata()
    else:
        current_time = to_kolkata(current_time)
    
    current_time_only = current_time.time()
    return open_time <= current_time_only <= close_time

def is_weekday(dt: Optional[datetime] = None) -> bool:
    """
    Check if given date is a weekday (Monday-Friday).
    
    Args:
        dt: Date to check (default: current IST date)
        
    Returns:
        True if weekday
    """
    if dt is None:
        dt = now_kolkata()
    else:
        dt = to_kolkata(dt)
    
    return dt.weekday() < 5  # 0-4 are Monday-Friday

def get_market_close_time(dt: Optional[datetime] = None, 
                         close_hour: int = 15, 
                         close_minute: int = 30) -> datetime:
    """
    Get the market closing datetime for the given day.
    
    Args:
        dt: Date for which to get close time (default: current date)
        close_hour: Market close hour (default: 15)
        close_minute: Market close minute (default: 30)
        
    Returns:
        Market close datetime in IST
    """
    if dt is None:
        dt = now_kolkata()
    else:
        dt = to_kolkata(dt)
    
    market_date = dt.date()
    close_time = time(close_hour, close_minute)
    
    return IST.localize(datetime.combine(market_date, close_time))

def is_time_to_exit(current_time: Optional[datetime] = None, 
                   minutes_before_close: int = 20,
                   close_hour: int = 15, 
                   close_minute: int = 30) -> bool:
    """
    Check if it's time to exit all positions (N minutes before market close).
    
    Args:
        current_time: Current time (default: current IST time)
        minutes_before_close: Minutes before close to trigger exit
        close_hour: Market close hour
        close_minute: Market close minute
        
    Returns:
        True if should exit positions
    """
    if current_time is None:
        current_time = now_kolkata()
    else:
        current_time = to_kolkata(current_time)
    
    close_dt = get_market_close_time(current_time, close_hour, close_minute)
    exit_time = close_dt - timedelta(minutes=minutes_before_close)
    
    return current_time >= exit_time

def get_session_remaining_minutes(current_time: Optional[datetime] = None,
                                close_hour: int = 15,
                                close_minute: int = 30) -> int:
    """
    Get remaining minutes in the current trading session.
    
    Args:
        current_time: Current time (default: current IST time)
        close_hour: Market close hour
        close_minute: Market close minute
        
    Returns:
        Remaining minutes until market close (0 if market closed)
    """
    if current_time is None:
        current_time = now_kolkata()
    else:
        current_time = to_kolkata(current_time)
    
    close_dt = get_market_close_time(current_time, close_hour, close_minute)
    
    if current_time >= close_dt:
        return 0
    
    remaining = close_dt - current_time
    return int(remaining.total_seconds() / 60)

def calculate_session_progress(current_time: Optional[datetime] = None,
                             open_hour: int = 9, open_minute: int = 15,
                             close_hour: int = 15, close_minute: int = 30) -> float:
    """
    Calculate how much of the trading session has elapsed (0.0 to 1.0).
    
    Args:
        current_time: Current time (default: current IST time)
        open_hour: Market open hour
        open_minute: Market open minute
        close_hour: Market close hour
        close_minute: Market close minute
        
    Returns:
        Session progress as fraction (0.0 = just opened, 1.0 = closed)
    """
    if current_time is None:
        current_time = now_kolkata()
    else:
        current_time = to_kolkata(current_time)
    
    open_dt = IST.localize(datetime.combine(current_time.date(), time(open_hour, open_minute)))
    close_dt = IST.localize(datetime.combine(current_time.date(), time(close_hour, close_minute)))
    
    if current_time <= open_dt:
        return 0.0
    elif current_time >= close_dt:
        return 1.0
    else:
        session_duration = close_dt - open_dt
        elapsed = current_time - open_dt
        return elapsed.total_seconds() / session_duration.total_seconds()

def get_next_trading_day(dt: Optional[datetime] = None) -> date:
    """
    Get the next trading day (skipping weekends).
    
    Args:
        dt: Reference date (default: current IST date)
        
    Returns:
        Next trading day
    """
    if dt is None:
        dt = now_kolkata()
    else:
        dt = to_kolkata(dt)
    
    next_day = dt.date() + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
        next_day += timedelta(days=1)
    
    return next_day

def get_previous_trading_day(dt: Optional[datetime] = None) -> date:
    """
    Get the previous trading day (skipping weekends).
    
    Args:
        dt: Reference date (default: current IST date)
        
    Returns:
        Previous trading day
    """
    if dt is None:
        dt = now_kolkata()
    else:
        dt = to_kolkata(dt)
    
    prev_day = dt.date() - timedelta(days=1)
    
    # Skip weekends
    while prev_day.weekday() >= 5:  # Saturday=5, Sunday=6
        prev_day -= timedelta(days=1)
    
    return prev_day

def is_pre_market(current_time: Optional[datetime] = None,
                 pre_market_start: time = DEFAULT_PRE_MARKET_OPEN,
                 market_open: time = DEFAULT_MARKET_OPEN) -> bool:
    """
    Check if current time is in pre-market session.
    
    Args:
        current_time: Time to check (default: current IST time)
        pre_market_start: Pre-market start time
        market_open: Regular market open time
        
    Returns:
        True if in pre-market session
    """
    if current_time is None:
        current_time = now_kolkata()
    else:
        current_time = to_kolkata(current_time)
    
    current_time_only = current_time.time()
    return pre_market_start <= current_time_only < market_open

def is_post_market(current_time: Optional[datetime] = None,
                  market_close: time = DEFAULT_MARKET_CLOSE,
                  post_market_end: time = DEFAULT_POST_MARKET_CLOSE) -> bool:
    """
    Check if current time is in post-market session.
    
    Args:
        current_time: Time to check (default: current IST time)
        market_close: Regular market close time
        post_market_end: Post-market end time
        
    Returns:
        True if in post-market session
    """
    if current_time is None:
        current_time = now_kolkata()
    else:
        current_time = to_kolkata(current_time)
    
    current_time_only = current_time.time()
    return market_close < current_time_only <= post_market_end

def wait_for_market_open(poll_interval: int = 60) -> None:
    """
    Block execution until market opens (useful for automated systems).
    
    Args:
        poll_interval: How often to check (seconds)
    """
    import time as time_module
    
    while not is_market_session():
        current = now_kolkata()
        next_open = get_market_session_times(current.date())[0]
        
        if current >= next_open:
            # Market should be open, check next day
            next_trading_day = get_next_trading_day(current)
            next_open = get_market_session_times(next_trading_day)[0]
        
        wait_seconds = (next_open - current).total_seconds()
        logger.info(f"Waiting for market open. Next session: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"Time remaining: {wait_seconds/3600:.1f} hours")
        
        time_module.sleep(min(poll_interval, wait_seconds))

def get_trading_session_info(dt: Optional[datetime] = None) -> dict:
    """
    Get comprehensive information about the current trading session.
    
    Args:
        dt: Reference datetime (default: current IST time)
        
    Returns:
        Dictionary with session information
    """
    if dt is None:
        dt = now_kolkata()
    else:
        dt = to_kolkata(dt)
    
    open_dt, close_dt = get_market_session_times(dt.date())
    
    return {
        'current_time': dt,
        'trading_date': dt.date(),
        'market_open': open_dt,
        'market_close': close_dt,
        'is_trading_day': is_weekday(dt),
        'is_market_session': is_market_session(dt),
        'is_pre_market': is_pre_market(dt),
        'is_post_market': is_post_market(dt),
        'session_progress': calculate_session_progress(dt),
        'remaining_minutes': get_session_remaining_minutes(dt),
        'should_exit_positions': is_time_to_exit(dt),
        'next_trading_day': get_next_trading_day(dt),
        'previous_trading_day': get_previous_trading_day(dt)
    }

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_market_calendar(year: int) -> list:
    """
    Get basic market calendar for the year (excluding major holidays).
    Note: This is a basic implementation. For production, use a proper market calendar service.
    
    Args:
        year: Year for calendar
        
    Returns:
        List of trading days (excludes weekends and major holidays)
    """
    # Basic holidays (this should be expanded with actual NSE holiday calendar)
    basic_holidays = [
        date(year, 1, 26),  # Republic Day
        date(year, 8, 15),  # Independence Day
        date(year, 10, 2),  # Gandhi Jayanti
    ]
    
    trading_days = []
    current_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    while current_date <= end_date:
        # Skip weekends and basic holidays
        if current_date.weekday() < 5 and current_date not in basic_holidays:
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    return trading_days

# Example usage and testing
if __name__ == "__main__":
    # Test current time functions
    current = now_kolkata()
    print(f"Current IST time: {current}")
    print(f"Formatted timestamp: {format_timestamp(current)}")
    
    # Test market session functions
    session_info = get_trading_session_info()
    print(f"\nMarket session info:")
    for key, value in session_info.items():
        print(f"  {key}: {value}")
    
    # Test time calculations
    print(f"\nTime calculations:")
    print(f"Is weekday: {is_weekday()}")
    print(f"Is market session: {is_market_session()}")
    print(f"Time to exit: {is_time_to_exit()}")
    print(f"Session progress: {calculate_session_progress():.1%}")
    print(f"Remaining minutes: {get_session_remaining_minutes()}")
    
    # Test market close time
    close_time = get_market_close_time()
    print(f"Market close time: {close_time}")
    
    # Test next/previous trading days
    print(f"Next trading day: {get_next_trading_day()}")
    print(f"Previous trading day: {get_previous_trading_day()}")
    
    print("\n✅ Time utilities test completed successfully!")
