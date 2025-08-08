"""
core/position_manager.py - Unified Position Manager for Backtest & Live Trading

Handles:
- Long-only, intraday-only position management
- F&O support with lot sizes and tick sizes
- Advanced order type simulation
- Tiered take-profits with partial exits
- Trailing stop loss management
- Commission and cost modeling
- Risk management and capital tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
from utils.config_helper import ConfigAccessor
from utils.time_utils import now_ist

logger = logging.getLogger(__name__)

class PositionType(Enum):
    LONG = "LONG"  # Only long supported

class PositionStatus(Enum):
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"

class ExitReason(Enum):
    TAKE_PROFIT_1 = "Take Profit 1"
    TAKE_PROFIT_2 = "Take Profit 2"
    TAKE_PROFIT_3 = "Take Profit 3"
    TAKE_PROFIT_4 = "Take Profit 4"
    STOP_LOSS = "Stop Loss"
    TRAILING_STOP = "Trailing Stop"
    SESSION_END = "Session End"
    STRATEGY_EXIT = "Strategy Exit"

@dataclass
class Position:
    position_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    initial_quantity: int
    current_quantity: int
    lot_size: int
    tick_size: float

    stop_loss_price: float
    tp_levels: List[float] = field(default_factory=list)
    tp_percentages: List[float] = field(default_factory=list)
    tp_executed: List[bool] = field(default_factory=list)

    trailing_enabled: bool = False
    trailing_activation_points: float = 0.0
    trailing_distance_points: float = 0.0
    trailing_activated: bool = False
    trailing_stop_price: Optional[float] = None
    highest_price: float = 0.0

    status: PositionStatus = PositionStatus.OPEN
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0

    exit_transactions: List[Dict] = field(default_factory=list)

    def update_unrealized_pnl(self, current_price: float):
        if self.current_quantity > 0:
            self.unrealized_pnl = (current_price - self.entry_price) * self.current_quantity
        else:
            self.unrealized_pnl = 0.0

    def get_total_pnl(self, current_price: float) -> float:
        self.update_unrealized_pnl(current_price)
        return self.realized_pnl + self.unrealized_pnl

    def update_trailing_stop(self, current_price: float):
        if not self.trailing_enabled or self.current_quantity == 0:
            return
        if current_price > self.highest_price:
            self.highest_price = current_price
        if not self.trailing_activated:
            profit_points = current_price - self.entry_price
            if profit_points >= self.trailing_activation_points:
                self.trailing_activated = True
                self.trailing_stop_price = current_price - self.trailing_distance_points
                logger.info(f"Trailing stop activated for {self.position_id} at {self.trailing_stop_price}")
        elif self.trailing_activated:
            new_stop = self.highest_price - self.trailing_distance_points
            if new_stop > (self.trailing_stop_price or 0):
                self.trailing_stop_price = new_stop

@dataclass
class Trade:
    trade_id: str
    position_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    commission: float
    net_pnl: float
    exit_reason: str
    duration_minutes: int
    lot_size: int

class PositionManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_accessor = ConfigAccessor(config)
        self.initial_capital = self.config_accessor.get_capital_param('initial_capital', 100000)
        self.current_capital = self.initial_capital
        self.reserved_margin = 0.0
        self.risk_per_trade_percent = self.config_accessor.get_risk_param('risk_per_trade_percent', 1.0)
        self.max_position_value_percent = self.config_accessor.get_risk_param('max_position_value_percent', 95)
        self.base_sl_points = self.config_accessor.get_risk_param('base_sl_points', 15)
        self.tp_points = self.config_accessor.get_risk_param('tp_points', [10, 25, 50, 100])
        self.tp_percentages = self.config_accessor.get_risk_param('tp_percents', [0.25, 0.25, 0.25, 0.25])
        self.use_trailing_stop = self.config_accessor.get_risk_param('use_trail_stop', True)
        self.trailing_activation_points = self.config_accessor.get_risk_param('trail_activation_points', 25)
        self.trailing_distance_points = self.config_accessor.get_risk_param('trail_distance_points', 10)
        self.commission_percent = self.config_accessor.get_risk_param('commission_percent', 0.03)
        self.commission_per_trade = self.config_accessor.get_risk_param('commission_per_trade', 0.0)
        self.stt_percent = self.config_accessor.get_risk_param('stt_percent', 0.025)
        self.exchange_charges_percent = self.config_accessor.get_risk_param('exchange_charges_percent', 0.0019)
        self.gst_percent = self.config_accessor.get_risk_param('gst_percent', 18.0)
        self.slippage_points = self.config_accessor.get_risk_param('slippage_points', 1)
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.session_params = config.get('session', {})
        logger.info(f"PositionManager initialized with capital: {self.initial_capital:,}")

    def _ensure_timezone(self, dt):
        """Ensure datetime is timezone-aware for consistent comparisons"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            import pytz
            return pytz.timezone('Asia/Kolkata').localize(dt)
        return dt

    def calculate_lot_aligned_quantity(self, desired_quantity: int, lot_size: int) -> int:
        if lot_size <= 1:  # Equity
            return max(1, desired_quantity)
        lots = max(1, round(desired_quantity / lot_size))
        return lots * lot_size

    def calculate_position_size(self, entry_price: float, stop_loss_price: float, lot_size: int = 1) -> int:
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            return 0
        max_risk_amount = self.current_capital * (self.risk_per_trade_percent / 100)
        raw_quantity = int(max_risk_amount / risk_per_unit)
        if raw_quantity <= 0:
            return 0
        aligned_quantity = self.calculate_lot_aligned_quantity(raw_quantity, lot_size)
        position_value = aligned_quantity * entry_price
        max_position_value = self.current_capital * (self.max_position_value_percent / 100)
        if position_value > max_position_value:
            max_lots = int(max_position_value / (lot_size * entry_price))
            aligned_quantity = max(1, max_lots) * lot_size
        return aligned_quantity

    def calculate_total_costs(self, price: float, quantity: int, is_buy: bool = True) -> Dict[str, float]:
        turnover = price * quantity
        commission = max(self.commission_per_trade, turnover * (self.commission_percent / 100))
        stt = turnover * (self.stt_percent / 100) if not is_buy else 0.0
        exchange_charges = turnover * (self.exchange_charges_percent / 100)
        taxable_amount = commission + exchange_charges
        gst = taxable_amount * (self.gst_percent / 100)
        total_costs = commission + stt + exchange_charges + gst
        return {
            'commission': commission,
            'stt': stt,
            'exchange_charges': exchange_charges,
            'gst': gst,
            'total_costs': total_costs,
            'turnover': turnover
        }

    def open_position(self, symbol: str, entry_price: float, timestamp: datetime,
                      lot_size: int = 1, tick_size: float = 0.05,
                      order_type: OrderType = OrderType.MARKET) -> Optional[str]:
        if order_type == OrderType.MARKET:
            actual_entry_price = entry_price + self.slippage_points * tick_size
        else:
            actual_entry_price = entry_price
        stop_loss_price = actual_entry_price - self.base_sl_points * tick_size
        quantity = self.calculate_position_size(actual_entry_price, stop_loss_price, lot_size)
        if quantity <= 0:
            logger.warning("Cannot open position: invalid quantity calculated")
            return None
        entry_costs = self.calculate_total_costs(actual_entry_price, quantity, is_buy=True)
        required_capital = entry_costs['turnover'] + entry_costs['total_costs']
        if required_capital > self.current_capital:
            logger.warning(f"Insufficient capital: required {required_capital:,.2f}, available {self.current_capital:,.2f}")
            return None
        position_id = str(uuid.uuid4())[:8]
        tp_levels = [actual_entry_price + (tp * tick_size) for tp in self.tp_points]
        position = Position(
            position_id=position_id,
            symbol=symbol,
            entry_time=timestamp,
            entry_price=actual_entry_price,
            initial_quantity=quantity,
            current_quantity=quantity,
            lot_size=lot_size,
            tick_size=tick_size,
            stop_loss_price=stop_loss_price,
            tp_levels=tp_levels,
            tp_percentages=self.tp_percentages.copy(),
            tp_executed=[False] * len(self.tp_points),
            trailing_enabled=self.use_trailing_stop,
            trailing_activation_points=self.trailing_activation_points * tick_size,
            trailing_distance_points=self.trailing_distance_points * tick_size,
            highest_price=actual_entry_price,
            total_commission=entry_costs['total_costs']
        )
        self.current_capital -= required_capital
        self.reserved_margin += required_capital
        self.positions[position_id] = position
        logger.info(f"Opened LONG position {position_id}: {quantity} {symbol} @ {actual_entry_price}")
        logger.info(f"SL: {stop_loss_price}, TPs: {tp_levels}")
        return position_id

    def close_position_partial(self, position_id: str, exit_price: float,
                               quantity_to_close: int, timestamp: datetime,
                               exit_reason: str) -> bool:
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return False
        position = self.positions[position_id]
        if quantity_to_close <= 0 or quantity_to_close > position.current_quantity:
            logger.error(f"Invalid quantity to close: {quantity_to_close}")
            return False
        exit_costs = self.calculate_total_costs(exit_price, quantity_to_close, is_buy=False)
        gross_pnl = (exit_price - position.entry_price) * quantity_to_close
        net_pnl = gross_pnl - exit_costs['total_costs']
        proceeds = exit_costs['turnover'] - exit_costs['total_costs']
        self.current_capital += proceeds
        position.current_quantity -= quantity_to_close
        position.realized_pnl += net_pnl
        position.total_commission += exit_costs['total_costs']
        duration = int((timestamp - position.entry_time).total_seconds() / 60)
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            position_id=position_id,
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=quantity_to_close,
            gross_pnl=gross_pnl,
            commission=exit_costs['total_costs'],
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            duration_minutes=duration,
            lot_size=position.lot_size
        )
        self.completed_trades.append(trade)
        position.exit_transactions.append({
            'timestamp': timestamp,
            'price': exit_price,
            'quantity': quantity_to_close,
            'reason': exit_reason,
            'pnl': net_pnl
        })
        if position.current_quantity == 0:
            position.status = PositionStatus.CLOSED
            self.reserved_margin -= (position.initial_quantity * position.entry_price)
            del self.positions[position_id]
            logger.info(f"Fully closed position {position_id}")
        else:
            position.status = PositionStatus.PARTIALLY_CLOSED
            logger.info(f"Partially closed position {position_id}: {quantity_to_close} @ ₹{exit_price}")
        self.daily_pnl += net_pnl
        return True

    def close_position_full(self, position_id: str, exit_price: float,
                            timestamp: datetime, exit_reason: str) -> bool:
        if position_id not in self.positions:
            return False
        position = self.positions[position_id]
        return self.close_position_partial(position_id, exit_price, position.current_quantity, timestamp, exit_reason)

    def check_exit_conditions(self, position_id: str, current_price: float, timestamp: datetime) -> List[Tuple[int, str]]:
        if position_id not in self.positions:
            return []
        position = self.positions[position_id]
        exits = []
        position.update_trailing_stop(current_price)
        if current_price <= position.stop_loss_price:
            exits.append((position.current_quantity, ExitReason.STOP_LOSS.value))
            return exits
        if (position.trailing_activated and position.trailing_stop_price and current_price <= position.trailing_stop_price):
            exits.append((position.current_quantity, ExitReason.TRAILING_STOP.value))
            return exits
        for i, (tp_level, tp_percentage, tp_executed) in enumerate(zip(position.tp_levels, position.tp_percentages, position.tp_executed)):
            if not tp_executed and current_price >= tp_level:
                position.tp_executed[i] = True
                exit_quantity = int(position.initial_quantity * tp_percentage) if i < len(position.tp_levels) - 1 else position.current_quantity
                exit_quantity = min(exit_quantity, position.current_quantity)
                if exit_quantity > 0:
                    reason = f"Take Profit {i+1}"
                    exits.append((exit_quantity, reason))
        return exits

    def process_positions(self, row, timestamp, session_params=None):
        """Enhanced position processing with session awareness"""
        current_price = row['close']
        
        # Check for session exit if parameters provided
        if session_params:
            from utils.time_utils import is_time_to_exit
            exit_buffer = session_params.get('exit_before_close', 20)
            end_hour = session_params.get('intraday_end_hour', 15)
            end_min = session_params.get('intraday_end_min', 30)
            
            if is_time_to_exit(timestamp, exit_buffer, end_hour, end_min):
                # Close all positions for session end
                for position_id in list(self.positions.keys()):
                    self.close_position_full(position_id, current_price, timestamp, "Session End")
                return
        
        # Ensure timezone-aware
        timestamp = self._ensure_timezone(timestamp)
        
        for position_id in list(self.positions.keys()):
            position = self.positions.get(position_id)
            if not position or position.status == PositionStatus.CLOSED:
                continue
            if session_end:
                self.close_position_full(position_id, current_price, timestamp, ExitReason.SESSION_END.value)
                continue
            exits = self.check_exit_conditions(position_id, current_price, timestamp)
            for exit_quantity, exit_reason in exits:
                if exit_quantity > 0:
                    self.close_position_partial(position_id, current_price, exit_quantity, timestamp, exit_reason)
                if position_id not in self.positions:
                    break

    def get_portfolio_value(self, current_price: float) -> float:
        total_value = self.current_capital
        for position in self.positions.values():
            if position.status != PositionStatus.CLOSED:
                position.update_unrealized_pnl(current_price)
                total_value += position.unrealized_pnl
        return total_value

    def get_open_positions(self) -> List[Dict[str, Any]]:
        open_positions = []
        for position in self.positions.values():
            if position.status != PositionStatus.CLOSED:
                open_positions.append({
                    'id': position.position_id,
                    'symbol': position.symbol,
                    'type': 'long',
                    'quantity': position.current_quantity,
                    'entry_price': position.entry_price,
                    'entry_time': position.entry_time,
                    'stop_loss': position.stop_loss_price,
                    'take_profits': position.tp_levels,
                    'trailing_active': position.trailing_activated,
                    'trailing_stop': position.trailing_stop_price,
                    'unrealized_pnl': position.unrealized_pnl
                })
        return open_positions

    def get_trade_history(self) -> List[Dict[str, Any]]:
        trades = []
        for trade in self.completed_trades:
            trades.append({
                'trade_id': trade.trade_id,
                'position_id': trade.position_id,
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'gross_pnl': trade.gross_pnl,
                'commission': trade.commission,
                'net_pnl': trade.net_pnl,
                'exit_reason': trade.exit_reason,
                'duration_minutes': trade.duration_minutes,
                'return_percent': (trade.net_pnl / (trade.entry_price * trade.quantity)) * 100
            })
        return trades

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'total_commission': 0.0
            }
        winning_trades = [t for t in self.completed_trades if t.net_pnl > 0]
        losing_trades = [t for t in self.completed_trades if t.net_pnl < 0]
        total_pnl = sum(t.net_pnl for t in self.completed_trades)
        total_commission = sum(t.commission for t in self.completed_trades)
        gross_profit = sum(t.net_pnl for t in winning_trades)
        gross_loss = abs(sum(t.net_pnl for t in losing_trades))
        return {
            'total_trades': len(self.completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(self.completed_trades)) * 100,
            'total_pnl': total_pnl,
            'avg_win': gross_profit / len(winning_trades) if winning_trades else 0,
            'avg_loss': gross_loss / len(losing_trades) if losing_trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'max_win': max(t.net_pnl for t in self.completed_trades),
            'max_loss': min(t.net_pnl for t in self.completed_trades),
            'total_commission': total_commission
        }

    def reset(self, initial_capital: Optional[float] = None):
        if initial_capital:
            self.initial_capital = initial_capital
        self.current_capital = self.initial_capital
        self.reserved_margin = 0.0
        self.daily_pnl = 0.0
        self.positions.clear()
        self.completed_trades.clear()
        logger.info(f"Position Manager reset with capital: {self.initial_capital:,}")

    # Legacy compatibility methods for backtest engine
    def enter_position(self, side: str, price: float, quantity: int, timestamp: datetime,
                       **kwargs) -> Optional[str]:
        if side.upper() != 'BUY':
            logger.warning("This system only supports LONG positions")
            return None
        symbol = kwargs.get('symbol', 'NIFTY')
        lot_size = kwargs.get('lot_size', 1)
        tick_size = kwargs.get('tick_size', 0.05)
        return self.open_position(symbol, price, timestamp, lot_size, tick_size)

    def exit_position(self, position_id: str, price: float, timestamp: datetime, reason: str):
        self.close_position_full(position_id, price, timestamp, reason)

    def can_enter_position(self) -> bool:
        return len(self.positions) < self.config_accessor.get_strategy_param('max_positions_per_day', 10)

    def calculate_position_size_gui_driven(self, entry_price: float, stop_loss_price: float, 
                                     user_capital: float, user_risk_pct: float, 
                                     user_lot_size: int) -> dict:
        """
        GUI-driven position sizing with comprehensive feedback
        
        Returns:
            dict with position details and capital analysis
        """
        
        if entry_price <= 0 or stop_loss_price <= 0:
            return {"error": "Invalid price inputs"}
        
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        # Risk-based calculation
        max_risk_amount = user_capital * (user_risk_pct / 100)
        risk_based_quantity = int(max_risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
        
        # Capital-constrained calculation  
        usable_capital = user_capital * 0.95  # 95% utilization limit
        max_affordable_shares = int(usable_capital / entry_price)
        capital_based_quantity = (max_affordable_shares // user_lot_size) * user_lot_size
        
        # Hybrid selection (conservative approach)
        final_quantity = min(risk_based_quantity, capital_based_quantity)
        final_lots = final_quantity // user_lot_size
        aligned_quantity = final_lots * user_lot_size
        
        # Calculate all metrics for GUI display
        position_value = aligned_quantity * entry_price
        actual_risk = aligned_quantity * risk_per_unit
        actual_risk_pct = (actual_risk / user_capital) * 100 if user_capital > 0 else 0
        capital_utilization = (position_value / user_capital) * 100 if user_capital > 0 else 0
        
        return {
            "recommended_quantity": aligned_quantity,
            "recommended_lots": final_lots,
            "position_value": position_value,
            "capital_utilization_pct": capital_utilization,
            "actual_risk_amount": actual_risk,
            "actual_risk_pct": actual_risk_pct,
            "max_affordable_lots": max_affordable_shares // user_lot_size,
            "risk_based_lots": risk_based_quantity // user_lot_size,
            "approach_used": "risk_limited" if final_quantity == risk_based_quantity else "capital_limited"
        }

if __name__ == "__main__":
    # Example usage and testing
    from datetime import datetime
    test_config = {
        'initial_capital': 500000,
        'risk_per_trade_percent': 1.0,
        'base_sl_points': 15,
        'tp_points': [10, 25, 50, 100],
        'tp_percents': [0.25, 0.25, 0.25, 0.25],
        'use_trail_stop': True,
        'trail_activation_points': 25,
        'trail_distance_points': 10,
        'commission_percent': 0.03,
        'stt_percent': 0.025
    }
    pm = PositionManager(test_config)
    position_id = pm.open_position("NIFTY24DECFUT", 22000.0, now_ist(), 15, 0.05)
    print(f"Opened position ID: {position_id}")
    # Simulate price move to hit take profit
    pm.process_positions(pd.Series({'close': 22035.0, 'session_exit': False}), now_ist())
    summary = pm.get_performance_summary()
    print("Performance Summary:", summary)
    print("✅ Position Manager test completed!")


"""
CONFIGURATION PARAMETER NAMING CONVENTION:
- Constructor parameter: __init__(self, config: Dict[str, Any])
- Internal storage: self.config = config
- All parameter access: config.get('parameter_name', default)
- Session parameters: self.session_params = config.get('session', {})

PARAMETER STRUCTURE EXPECTATION:
This class expects a consolidated configuration dictionary containing:
- Strategy parameters (direct keys in config)
- Risk management parameters (direct keys in config)
- Instrument parameters (direct keys in config)
- Capital parameters (direct keys in config)
- Session parameters under 'session' key

CRITICAL: Calling code must pass consolidated config, not separate parameter dictionaries
"""
