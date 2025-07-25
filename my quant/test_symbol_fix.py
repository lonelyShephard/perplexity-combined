#!/usr/bin/env python3
"""
Test script to validate the symbol management fixes
"""

import sys
import os
sys.path.append('.')

from utils.cache_manager import load_symbol_cache

def test_symbol_loading():
    print("Testing symbol cache loading...")
    
    try:
        symbols = load_symbol_cache()
        print(f"✅ Successfully loaded {len(symbols)} symbols")
        
        # Test a few sample symbols
        sample_count = min(5, len(symbols))
        sample_symbols = list(symbols.items())[:sample_count]
        
        print(f"\nSample symbols ({sample_count}):")
        for symbol, details in sample_symbols:
            token = details.get('token', 'N/A')
            exchange = details.get('exchange', 'N/A')
            instrumenttype = details.get('instrumenttype', 'N/A')
            print(f"  {symbol}")
            print(f"    Token: {token}")
            print(f"    Exchange: {exchange}")
            print(f"    Type: {instrumenttype}")
        
        # Test filtering by exchange
        print(f"\nTesting exchange filtering:")
        exchanges = ['NSE', 'NFO', 'BSE']
        for exch in exchanges:
            matching = [s for s, d in symbols.items() if exch in d.get('exchange', '')][:5]
            print(f"  {exch}: {len(matching)} symbols (showing first 5)")
            for sym in matching:
                print(f"    {sym} -> {symbols[sym].get('token')}")
        
        # Test NIFTY symbols specifically
        nifty_symbols = [s for s in symbols.keys() if 'NIFTY' in s.upper()][:10]
        print(f"\nFound {len(nifty_symbols)} NIFTY-related symbols (showing first 10):")
        for sym in nifty_symbols:
            details = symbols[sym]
            display_text = f"{sym} (Token: {details.get('token', 'N/A')}) [{details.get('exchange', 'N/A')}]"
            print(f"  {display_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading symbols: {e}")
        return False

def test_symbol_filtering():
    print("\n" + "="*50)
    print("Testing symbol filtering functionality...")
    
    try:
        symbols = load_symbol_cache()
        
        # Simulate the GUI filtering logic
        exchange = "NSE_FO"
        filtered_symbols = []
        symbol_to_token = {}
        
        for symbol, details in symbols.items():
            symbol_exchange = details.get("exchange", "")
            
            # Test exchange matching logic
            exchange_match = False
            if exchange == "NSE_FO" and ("NSE" in symbol_exchange or "NFO" in symbol_exchange):
                exchange_match = True
            elif exchange == "NSE_CM" and ("NSE" in symbol_exchange and "NFO" not in symbol_exchange):
                exchange_match = True
            elif exchange == "BSE_CM" and "BSE" in symbol_exchange:
                exchange_match = True
            elif exchange in symbol_exchange:
                exchange_match = True
            
            if exchange_match:
                display_text = f"{symbol} (Token: {details.get('token', 'N/A')}) [{symbol_exchange}]"
                filtered_symbols.append(display_text)
                symbol_to_token[display_text] = {
                    'token': details.get('token', ''),
                    'symbol': symbol,
                    'exchange': symbol_exchange,
                    'instrumenttype': details.get('instrumenttype', ''),
                    'expiry': details.get('expiry', '')
                }
        
        print(f"✅ Filtered {len(filtered_symbols)} symbols for {exchange}")
        
        # Show sample filtered symbols
        sample_filtered = sorted(filtered_symbols)[:10]
        print(f"\nSample filtered symbols for {exchange}:")
        for display_text in sample_filtered:
            details = symbol_to_token[display_text]
            print(f"  {display_text}")
            print(f"    Actual Symbol: {details['symbol']}")
            print(f"    Token: {details['token']}")
        
        # Test autocomplete filtering
        print(f"\nTesting autocomplete filtering with 'NIFTY':")
        typed_value = "NIFTY"
        matching_symbols = []
        for s in filtered_symbols:
            symbol_upper = s.upper()
            if (typed_value in symbol_upper or 
                typed_value in symbol_to_token[s]['symbol'].upper() or
                typed_value in symbol_to_token[s]['token']):
                matching_symbols.append(s)
        
        print(f"  Found {len(matching_symbols)} matches for '{typed_value}'")
        for match in matching_symbols[:5]:  # Show first 5
            print(f"    {match}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in filtering test: {e}")
        return False

if __name__ == "__main__":
    print("Symbol Management Testing")
    print("=" * 50)
    
    success1 = test_symbol_loading()
    success2 = test_symbol_filtering()
    
    if success1 and success2:
        print(f"\n✅ All tests passed! Symbol management should work correctly.")
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
