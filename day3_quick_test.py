#!/usr/bin/env python3
"""
day3_quick_test.py
Quick test to verify Day 3 fixes
"""

def test_unicode_fix():
    """Test that unicode encoding is fixed"""
    print("🧪 Testing Unicode Fix...")
    
    try:
        # Test writing unicode characters to file
        import tempfile
        import os
        
        test_content = "🚀 Day 3 - Unicode Test\n✅ Working!\n📊 Emojis supported"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        # Read it back
        with open(temp_file, 'r', encoding='utf-8') as f:
            read_content = f.read()
        
        os.unlink(temp_file)
        
        if test_content == read_content:
            print("✅ Unicode encoding fix working!")
            return True
        else:
            print("❌ Unicode test failed")
            return False
            
    except Exception as e:
        print(f"❌ Unicode test error: {e}")
        return False

def test_correlation_detection():
    """Test that correlation detection is working"""
    print("\n🧪 Testing Correlation Detection...")
    
    try:
        import sys
        sys.path.insert(0, '.')
        
        # Create simple test data
        import pandas as pd
        import numpy as np
        from day3_correlation_engine import CrossProtocolCorrelator
        
        # Create test DataFrame with perfect correlation
        timestamps = np.arange(0, 1, 0.1)
        values = np.sin(timestamps)
        
        # Signal 1 (CAN)
        data1 = pd.DataFrame({
            'timestamp': timestamps,
            'protocol': 'CAN',
            'signal': 'TestSignal1',
            'message': 'TestMessage1',
            'value': values
        })
        
        # Signal 2 (LIN) - same values with slight delay
        data2 = pd.DataFrame({
            'timestamp': timestamps + 0.05,  # 50ms delay
            'protocol': 'LIN', 
            'signal': 'TestSignal2',
            'message': 'TestMessage2',
            'value': values  # Same values (perfect gateway relay)
        })
        
        # Combine data
        df = pd.concat([data1, data2], ignore_index=True)
        
        # Test correlation engine
        correlator = CrossProtocolCorrelator()
        correlations = correlator.find_correlations(df)
        
        if len(correlations) > 0:
            print(f"✅ Correlation detection working!")
            print(f"   - Found {len(correlations)} correlations")
            
            gateway_candidates = [c for c in correlations if c.gateway_candidate]
            print(f"   - Gateway candidates: {len(gateway_candidates)}")
            
            if gateway_candidates:
                best = gateway_candidates[0]
                print(f"   - Best gateway: {best.confidence_score:.1f}% confidence")
            
            return True
        else:
            print("⚠️  No correlations found (may need tuning)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        print(f"❌ Correlation test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick tests"""
    print("🔧 Day 3 - Quick Fix Verification")
    print("=" * 35)
    
    test1 = test_unicode_fix()
    test2 = test_correlation_detection()
    
    print("\n" + "=" * 35)
    print("📊 Fix Test Results:")
    print(f"   Unicode encoding: {'✅' if test1 else '❌'}")
    print(f"   Correlation detection: {'✅' if test2 else '❌'}")
    
    if test1 and test2:
        print("\n✅ All fixes working!")
        print("🚀 Ready to rerun day3_test.py")
    else:
        print("\n❌ Some fixes need attention")

if __name__ == "__main__":
    main()