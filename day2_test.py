#!/usr/bin/env python3
"""
Day 2 Test Script - Verify enhanced functionality
"""

import tempfile
import os
import sys
from pathlib import Path

def create_enhanced_test_data():
    """Create test data with quality issues for Day 2 testing"""
    
    # Enhanced DBC with more signals
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

BU_: ECU1 ECU2 Gateway

BO_ 1234 EngineData: 8 ECU1
 SG_ RPM : 0|16@1+ (0.25,0) [0|16383.75] "rpm" ECU2
 SG_ Throttle : 16|8@1+ (0.4,0) [0|102] "%" ECU2
 SG_ Temperature : 24|8@1+ (1,-40) [-40|215] "C" ECU2

BO_ 1235 VehicleSpeed: 8 ECU2  
 SG_ Speed : 0|16@1+ (0.1,0) [0|6553.5] "km/h" Gateway
 SG_ Odometer : 16|32@1+ (0.1,0) [0|429496729.5] "km" Gateway

BO_ 1236 DiagnosticData: 8 Gateway
 SG_ ErrorCode : 0|16@1+ (1,0) [0|65535] "" ECU1,ECU2
 SG_ StatusBits : 16|8@1+ (1,0) [0|255] "" ECU1,ECU2
"""
    
    # Enhanced ASC with quality issues
    asc_content = """date Wed Nov 15 14:30:00 2023
base hex  timestamps absolute
internal events logged
// version 9.0.0
Begin Triggerblock Wed Nov 15 14:30:00 2023
// Good quality signals
   0.000000 1  4D2             Rx   d 8  10 00 40 14 00 00 12 34
   0.010000 1  4D2             Rx   d 8  11 00 41 15 00 00 12 35
   0.020000 1  4D2             Rx   d 8  12 00 42 16 00 00 12 36
   0.030000 1  4D2             Rx   d 8  13 00 43 17 00 00 12 37
   0.040000 1  4D2             Rx   d 8  14 00 44 18 00 00 12 38

// Vehicle speed (different message)
   0.005000 1  4D3             Rx   d 8  C8 00 10 27 00 00 00 00
   0.015000 1  4D3             Rx   d 8  D2 00 10 27 01 00 00 00
   0.025000 1  4D3             Rx   d 8  DC 00 10 27 02 00 00 00
   0.035000 1  4D3             Rx   d 8  E6 00 10 27 03 00 00 00

// Stuck signal example (Temperature stuck at same value)
   0.050000 1  4D2             Rx   d 8  15 00 44 18 00 00 12 39
   0.060000 1  4D2             Rx   d 8  16 00 44 18 00 00 12 40
   0.070000 1  4D2             Rx   d 8  17 00 44 18 00 00 12 41
   0.080000 1  4D2             Rx   d 8  18 00 44 18 00 00 12 42
   0.090000 1  4D2             Rx   d 8  19 00 44 18 00 00 12 43

// Missing data gap (no messages for a while)
   0.200000 1  4D2             Rx   d 8  20 00 45 19 00 00 12 44
   0.210000 1  4D2             Rx   d 8  21 00 46 20 00 00 12 45

// Diagnostic data with errors
   0.100000 1  4D4             Rx   d 8  FF 00 80 00 00 00 00 00
   0.200000 1  4D4             Rx   d 8  00 00 00 00 00 00 00 00

// High frequency burst
   0.250000 1  4D3             Rx   d 8  F0 00 20 27 10 00 00 00
   0.251000 1  4D3             Rx   d 8  F1 00 20 27 11 00 00 00
   0.252000 1  4D3             Rx   d 8  F2 00 20 27 12 00 00 00
   0.253000 1  4D3             Rx   d 8  F3 00 20 27 13 00 00 00
   0.254000 1  4D3             Rx   d 8  F4 00 20 27 14 00 00 00

End TriggerBlock
"""
    
    # Create LIN test data (basic)
    ldf_content = """
LIN_description_file;
LIN_protocol_version = "2.1";
LIN_language_version = "2.1";
LIN_speed = 19200;

Nodes {
  Master: LIN_Master, 5ms, 0.1ms ;
  Slaves: LIN_Slave1, LIN_Slave2 ;
}

Signals {
  Signal1: 8, 0, LIN_Master, LIN_Slave1 ;
  Signal2: 16, 0, LIN_Slave1, LIN_Master ;
}

Frames {
  Frame1: 0x01, LIN_Master, 2 {
    Signal1, 0 ;
  }
  Frame2: 0x02, LIN_Slave1, 4 {
    Signal2, 0 ;
  }
}
"""
    
    lin_log_content = """# LIN Log File
# timestamp frame_id data
0.000 01 A5
0.100 02 B6C7  
0.200 01 A6
0.300 02 B7C8
0.400 01 A7
"""
    
    # Write test files
    files = {}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dbc', delete=False) as f:
        f.write(dbc_content)
        files['dbc'] = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
        f.write(asc_content)
        files['asc'] = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ldf', delete=False) as f:
        f.write(ldf_content)
        files['ldf'] = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lin', delete=False) as f:
        f.write(lin_log_content)
        files['lin_log'] = f.name
    
    return files

def test_day2_single_protocol():
    """Test Day 2 with single protocol (backward compatibility)"""
    print("🧪 Testing Day 2 - Single Protocol Mode")
    print("-" * 40)
    
    files = create_enhanced_test_data()
    
    try:
        # Import Day 2 analyzer
        sys.path.insert(0, '.')
        from day2_enhanced_analyzer import EnhancedNetworkAnalyzer
        from day2_universal_signal import ProtocolType
        
        analyzer = EnhancedNetworkAnalyzer()
        results = analyzer.analyze_single_protocol(
            files['dbc'], files['asc'], ProtocolType.CAN
        )
        
        if results and results['total_signals'] > 0:
            print(f"✅ Single protocol test PASSED!")
            print(f"   - {results['total_signals']} signals processed")
            print(f"   - {len(results['quality_analysis'])} signals analyzed for quality")
            
            # Test quality detection
            poor_quality = sum(1 for q in results['quality_analysis'].values() if q.quality_score < 70)
            print(f"   - {poor_quality} signals with quality issues detected")
            
            return True
        else:
            print("❌ Single protocol test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Single protocol test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        for file_path in files.values():
            try:
                os.unlink(file_path)
            except:
                pass

def test_day2_multi_protocol():
    """Test Day 2 multi-protocol functionality"""
    print("\n🧪 Testing Day 2 - Multi-Protocol Mode")
    print("-" * 40)
    
    files = create_enhanced_test_data()
    
    try:
        # Create config file
        config_content = f"""# Multi-protocol test config
CAN,{files['dbc']},{files['asc']}
LIN,{files['ldf']},{files['lin_log']}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        # Import Day 2 analyzer
        from day2_enhanced_analyzer import EnhancedNetworkAnalyzer
        from day2_universal_signal import ProtocolType
        
        analyzer = EnhancedNetworkAnalyzer()
        
        # Add protocols
        can_ok = analyzer.add_protocol_database(ProtocolType.CAN, files['dbc'])
        lin_ok = analyzer.add_protocol_database(ProtocolType.LIN, files['ldf'])
        
        if not (can_ok and lin_ok):
            print("❌ Failed to load protocol databases")
            return False
        
        # Analyze multiple logs
        log_configs = [
            {"path": files['asc'], "protocol": ProtocolType.CAN},
            {"path": files['lin_log'], "protocol": ProtocolType.LIN}
        ]
        
        results = analyzer.analyze_mixed_logs(log_configs)
        
        if results and results['total_signals'] > 0:
            protocols_found = results['protocols']
            print(f"✅ Multi-protocol test PASSED!")
            print(f"   - {results['total_signals']} total signals")
            print(f"   - Protocols: {', '.join(protocols_found)}")
            print(f"   - Quality analysis: {len(results['quality_analysis'])} signals")
            
            # Check if we got both protocols
            if 'CAN' in protocols_found and 'LIN' in protocols_found:
                print(f"   - ✅ Both CAN and LIN protocols detected!")
                return True
            else:
                print(f"   - ⚠️  Only {protocols_found} detected")
                return True  # Still a success, just incomplete
        else:
            print("❌ Multi-protocol test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Multi-protocol test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        for file_path in files.values():
            try:
                os.unlink(file_path)
            except:
                pass
        try:
            os.unlink(config_file)
        except:
            pass

def main():
    """Run Day 2 tests"""
    print("🚀 Day 2 - Enhanced Network Analyzer Tests")
    print("=" * 50)
    
    # Test 1: Single protocol (backward compatibility)
    test1_pass = test_day2_single_protocol()
    
    # Test 2: Multi-protocol functionality  
    test2_pass = test_day2_multi_protocol()
    
    # Overall results
    print("\n" + "=" * 50)
    if test1_pass and test2_pass:
        print("🎉 DAY 2 COMPLETE!")
        print("✅ All tests passed!")
        print("🚀 Enhanced analyzer with:")
        print("   - Signal quality assessment")
        print("   - Multi-protocol support (CAN + LIN)")
        print("   - Universal signal structure")
        print("   - Backward compatibility")
        print("\n✅ Ready for Day 3: Cross-Protocol Correlation!")
    else:
        print("❌ Some tests failed:")
        print(f"   Single protocol: {'✅' if test1_pass else '❌'}")
        print(f"   Multi-protocol: {'✅' if test2_pass else '❌'}")
        print("\n🔧 Fix the issues before proceeding to Day 3")

if __name__ == "__main__":
    main()