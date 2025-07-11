#!/usr/bin/env python3
"""
Day 1 Clean Test - Verify everything works
"""

import tempfile
import os
import sys
from pathlib import Path

def create_test_files():
    """Create test DBC and ASC files"""
    
    # Create test DBC
    dbc_content = """
VERSION ""

NS_ : 
    NS_DESC_
    CM_
    BA_DEF_
    BA_
    VAL_
    CAT_DEF_
    CAT_
    FILTER
    BA_DEF_DEF_
    EV_DATA_
    ENVVAR_DATA_
    SGTYPE_
    SGTYPE_VAL_
    BA_DEF_SGTYPE_
    BA_SGTYPE_
    SIG_VALTYPE_
    SIGTYPE_VALTYPE_
    BO_TX_BU_
    BA_DEF_REL_
    BA_REL_
    BA_DEF_DEF_REL_
    BU_SG_REL_
    BU_EV_REL_
    BU_BO_REL_
    SG_MUL_VAL_

BS_:

BU_:

BO_ 1234 TestMessage: 8 Vector__XXX
 SG_ TestSignal : 0|16@1+ (0.01,0) [0|655.35] "V" Vector__XXX
 SG_ AnotherSignal : 16|16@1+ (0.1,0) [0|6553.5] "A" Vector__XXX
"""
    
    # Create test ASC
    asc_content = """date Wed Nov 15 14:30:00 2023
base hex  timestamps absolute
internal events logged
// version 9.0.0
Begin Triggerblock Wed Nov 15 14:30:00 2023
   0.000000 1  4D2             Rx   d 8  10 00 64 00 00 00 00 00
   0.100000 1  4D2             Rx   d 8  11 00 65 00 00 00 00 00
   0.200000 1  4D2             Rx   d 8  12 00 66 00 00 00 00 00
   0.300000 1  4D2             Rx   d 8  13 00 67 00 00 00 00 00
   0.400000 1  4D2             Rx   d 8  14 00 68 00 00 00 00 00
End TriggerBlock
"""
    
    # Write files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dbc', delete=False) as f:
        f.write(dbc_content)
        dbc_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
        f.write(asc_content)
        asc_file = f.name
    
    return dbc_file, asc_file

def main():
    """Run clean Day 1 test"""
    print("🧪 Day 1 - Clean Test")
    print("=" * 30)
    
    # Create test files
    dbc_file, asc_file = create_test_files()
    
    try:
        # Test the analyzer
        sys.path.insert(0, '.')
        from can_analyzer import SimpleCANAnalyzer
        
        analyzer = SimpleCANAnalyzer()
        df = analyzer.analyze_file(dbc_file, asc_file)
        
        if not df.empty:
            print(f"\n✅ SUCCESS!")
            print(f"   📊 {len(df)} signal samples extracted")
            print(f"   🎯 Signals found: {list(df['signal'].unique())}")
            print(f"   ⏰ Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}s")
            
            # Show sample data
            print(f"\n📋 Sample Data:")
            print(df.head())
            
            return True
        else:
            print("❌ FAILED - No data extracted")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    finally:
        # Cleanup
        os.unlink(dbc_file)
        os.unlink(asc_file)

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Day 1 COMPLETE!")
        print("✅ Your automotive network analyzer is working!")
        print("🚀 Ready for Day 2: Signal Quality & Multi-Protocol")
    else:
        print("\n🔧 Something needs fixing...")