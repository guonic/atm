"""
Test to verify Qlib's actual data format behavior.

This test checks what format Qlib actually returns when querying:
1. Single instrument as list: [symbol]
2. Multiple instruments: [symbol1, symbol2, ...]

The goal is to understand if Qlib's behavior is consistent or varies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from qlib.data import D


def test_single_instrument_format():
    """Test format when querying single instrument."""
    print("=" * 60)
    print("Test: Single instrument query format")
    print("=" * 60)
    
    try:
        # Get a sample instrument
        instruments = D.instruments()
        if len(instruments) == 0:
            print("  ⚠ No instruments available, skipping test")
            return
        
        sample_symbol = instruments[0]
        print(f"  Testing with symbol: {sample_symbol}")
        
        # Query single instrument (as list with one element)
        data = D.features(
            instruments=[sample_symbol],
            fields=["$close", "$high", "$low", "$volume"],
            start_time="2025-07-01",
            end_time="2025-07-10",
        )
        
        print(f"  Data shape: {data.shape}")
        print(f"  Index type: {type(data.index)}")
        print(f"  Columns type: {type(data.columns)}")
        print(f"  Is MultiIndex columns: {isinstance(data.columns, pd.MultiIndex)}")
        print(f"  Columns: {list(data.columns)}")
        
        if isinstance(data.columns, pd.MultiIndex):
            print(f"  MultiIndex levels: {data.columns.nlevels}")
            print(f"  Level 0 (instruments): {data.columns.get_level_values(0).unique().tolist()}")
            print(f"  Level 1 (fields): {data.columns.get_level_values(1).unique().tolist()}")
        else:
            print(f"  Single level columns: {data.columns.tolist()}")
        
        print(f"  Sample data (first 3 rows):")
        print(data.head(3))
        print()
        
        return data
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_instruments_format():
    """Test format when querying multiple instruments."""
    print("=" * 60)
    print("Test: Multiple instruments query format")
    print("=" * 60)
    
    try:
        # Get sample instruments
        instruments = D.instruments()
        if len(instruments) < 2:
            print("  ⚠ Less than 2 instruments available, skipping test")
            return
        
        sample_symbols = instruments[:2]
        print(f"  Testing with symbols: {sample_symbols}")
        
        # Query multiple instruments
        data = D.features(
            instruments=sample_symbols,
            fields=["$close", "$high", "$low", "$volume"],
            start_time="2025-07-01",
            end_time="2025-07-10",
        )
        
        print(f"  Data shape: {data.shape}")
        print(f"  Index type: {type(data.index)}")
        print(f"  Columns type: {type(data.columns)}")
        print(f"  Is MultiIndex columns: {isinstance(data.columns, pd.MultiIndex)}")
        
        if isinstance(data.columns, pd.MultiIndex):
            print(f"  MultiIndex levels: {data.columns.nlevels}")
            print(f"  Level 0 (instruments): {data.columns.get_level_values(0).unique().tolist()}")
            print(f"  Level 1 (fields): {data.columns.get_level_values(1).unique().tolist()}")
        else:
            print(f"  Single level columns: {data.columns.tolist()}")
        
        print(f"  Sample data (first 3 rows):")
        print(data.head(3))
        print()
        
        return data
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_formats():
    """Compare formats between single and multiple instrument queries."""
    print("=" * 60)
    print("Comparison: Single vs Multiple instrument formats")
    print("=" * 60)
    
    single_data = test_single_instrument_format()
    multiple_data = test_multiple_instruments_format()
    
    if single_data is None or multiple_data is None:
        print("  ⚠ Cannot compare - one or both tests failed")
        return
    
    single_is_multi = isinstance(single_data.columns, pd.MultiIndex)
    multiple_is_multi = isinstance(multiple_data.columns, pd.MultiIndex)
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Single instrument query: {'MultiIndex' if single_is_multi else 'Single level'} columns")
    print(f"  Multiple instruments query: {'MultiIndex' if multiple_is_multi else 'Single level'} columns")
    
    if single_is_multi == multiple_is_multi:
        print("  ✓ Formats are CONSISTENT")
        if single_is_multi:
            print("  → Qlib ALWAYS returns MultiIndex columns (even for single instrument)")
        else:
            print("  → Qlib ALWAYS returns single level columns (even for multiple instruments)")
    else:
        print("  ⚠ Formats are INCONSISTENT")
        print("  → Qlib returns different formats based on number of instruments")
        print("  → This requires format detection in our code")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Qlib Data Format Behavior Test")
    print("=" * 60)
    print()
    
    compare_formats()
    
    print()
    print("=" * 60)
    print("Conclusion:")
    print("=" * 60)
    print("Based on the test results above, we can determine:")
    print("1. Whether Qlib's format is consistent or varies")
    print("2. What format our code should expect")
    print("3. Whether our current format detection logic is correct")
    print()


if __name__ == "__main__":
    main()
