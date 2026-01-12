#!/usr/bin/env python3
"""
Simple test script to verify cash handling fix.

Run directly: python python/trading/tests/test_cash_fix_simple.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from trading.state import Account, PositionManager, OrderBook, Order, OrderSide, OrderType
from trading.execution import Executor
from trading.storage import MemoryStorage


def test_buy_order_without_frozen_cash():
    """Test that buy order execution works when frozen_cash is 0."""
    print("Test 1: Buy order without frozen cash")
    
    # Setup
    storage = MemoryStorage()
    account = Account(
        account_id="test_001",
        available_cash=100000.0,
        frozen_cash=0.0,
        initial_cash=100000.0,
    )
    position_manager = PositionManager(account, storage)
    order_book = OrderBook(storage)
    executor = Executor(position_manager, order_book, commission_rate=0.0015, min_commission=5.0)
    
    # Create order
    order = Order(
        symbol="000001.SZ",
        side=OrderSide.BUY,
        amount=1000,
        order_type=OrderType.MARKET,
        date=pd.Timestamp("2025-07-01"),
    )
    order_book.submit(order)
    
    # Verify initial state
    assert account.frozen_cash == 0.0, f"Expected frozen_cash=0.0, got {account.frozen_cash}"
    assert account.available_cash == 100000.0, f"Expected available_cash=100000.0, got {account.available_cash}"
    
    # Create market data
    market_data = pd.DataFrame({
        '$close': [10.0],
        '$open': [10.0],
        '$high': [10.5],
        '$low': [9.5],
        '$volume': [1000000],
    }, index=pd.MultiIndex.from_product([
        ['000001.SZ'],
        [pd.Timestamp('2025-07-01')]
    ], names=['instrument', 'datetime']))
    
    # Execute order
    fill_info = executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
    
    # Verify
    assert fill_info is not None, "Order should be filled"
    assert fill_info.symbol == "000001.SZ", f"Expected symbol=000001.SZ, got {fill_info.symbol}"
    assert fill_info.amount == 1000, f"Expected amount=1000, got {fill_info.amount}"
    
    # Verify cash deduction
    expected_cost = 1000 * 10.0 * (1 + 0.0015) + 5.0  # 10015 + 5 = 10020
    assert abs(account.available_cash - (100000.0 - expected_cost)) < 0.01, \
        f"Expected available_cash={100000.0 - expected_cost:.2f}, got {account.available_cash:.2f}"
    assert account.frozen_cash == 0.0, f"Expected frozen_cash=0.0, got {account.frozen_cash}"
    
    # Verify position
    assert "000001.SZ" in position_manager.all_positions, "Position should be created"
    position = position_manager.all_positions["000001.SZ"]
    assert position.amount == 1000, f"Expected position amount=1000, got {position.amount}"
    assert position.entry_price == 10.0, f"Expected entry_price=10.0, got {position.entry_price}"
    
    print("  ✓ Test 1 passed: Buy order executed successfully without frozen cash")
    print(f"    Available cash after: {account.available_cash:.2f}")
    print(f"    Frozen cash: {account.frozen_cash:.2f}")


def test_cash_availability_check():
    """Test that cash availability check uses available_cash + frozen_cash."""
    print("\nTest 2: Cash availability check uses total cash")
    
    # Setup
    storage = MemoryStorage()
    account = Account(
        account_id="test_002",
        available_cash=50000.0,
        frozen_cash=50000.0,
        initial_cash=100000.0,
    )
    position_manager = PositionManager(account, storage)
    order_book = OrderBook(storage)
    executor = Executor(position_manager, order_book, commission_rate=0.0015, min_commission=5.0)
    
    # Create order with target_value > available_cash but < total cash
    order = Order(
        symbol="000001.SZ",
        side=OrderSide.BUY,
        amount=0,
        target_value=80000.0,  # More than available_cash (50k) but less than total (100k)
        order_type=OrderType.MARKET,
        date=pd.Timestamp("2025-07-01"),
    )
    order_book.submit(order)
    
    # Create market data
    market_data = pd.DataFrame({
        '$close': [10.0],
        '$open': [10.0],
        '$high': [10.5],
        '$low': [9.5],
        '$volume': [1000000],
    }, index=pd.MultiIndex.from_product([
        ['000001.SZ'],
        [pd.Timestamp('2025-07-01')]
    ], names=['instrument', 'datetime']))
    
    # Execute order - should succeed because total cash (100k) > target (80k)
    fill_info = executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
    
    # Verify
    assert fill_info is not None, "Order should be filled when total cash is sufficient"
    assert order.status.value != "REJECTED", f"Order should not be rejected, status={order.status.value}"
    
    print("  ✓ Test 2 passed: Order filled using total cash (available + frozen)")
    print(f"    Available cash: {account.available_cash:.2f}")
    print(f"    Frozen cash: {account.frozen_cash:.2f}")


def test_insufficient_cash_rejects_order():
    """Test that order is rejected when total cash is insufficient."""
    print("\nTest 3: Insufficient cash rejects order")
    
    # Setup with limited cash
    storage = MemoryStorage()
    account = Account(
        account_id="test_003",
        available_cash=1000.0,
        frozen_cash=0.0,
        initial_cash=1000.0,
    )
    position_manager = PositionManager(account, storage)
    order_book = OrderBook(storage)
    executor = Executor(position_manager, order_book, commission_rate=0.0015, min_commission=5.0)
    
    # Create order that requires more cash than available
    order = Order(
        symbol="000001.SZ",
        side=OrderSide.BUY,
        amount=0,
        target_value=50000.0,  # Much more than available (1k)
        order_type=OrderType.MARKET,
        date=pd.Timestamp("2025-07-01"),
    )
    order_book.submit(order)
    
    initial_cash = account.available_cash
    
    # Create market data
    market_data = pd.DataFrame({
        '$close': [10.0],
        '$open': [10.0],
        '$high': [10.5],
        '$low': [9.5],
        '$volume': [1000000],
    }, index=pd.MultiIndex.from_product([
        ['000001.SZ'],
        [pd.Timestamp('2025-07-01')]
    ], names=['instrument', 'datetime']))
    
    # Execute order - should be rejected
    fill_info = executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
    
    # Verify
    assert fill_info is None, "Order should be rejected when cash is insufficient"
    assert order.status.value == "REJECTED", f"Order should be rejected, status={order.status.value}"
    assert account.available_cash == initial_cash, "Cash should not be deducted"
    
    print("  ✓ Test 3 passed: Order rejected when cash is insufficient")
    print(f"    Available cash unchanged: {account.available_cash:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Executor Cash Handling Fix")
    print("=" * 60)
    
    try:
        test_buy_order_without_frozen_cash()
        test_cash_availability_check()
        test_insufficient_cash_rejects_order()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
