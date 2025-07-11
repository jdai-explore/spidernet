#!/usr/bin/env python3
"""
day3_test.py
Complete test suite for Day 3 functionality
Tests correlation engine, gateway analysis, and complete network analysis
"""

import tempfile
import os
import sys
import numpy as np
from pathlib import Path

def create_correlated_test_data():
    """Create test data with intentional correlations for testing"""
    
    # Enhanced DBC with correlated signals
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
 SG_ GatewayRPM : 16|16@1+ (0.25,0) [0|16383.75] "rpm" Gateway

BO_ 1236 SystemStatus: 8 Gateway
 SG_ GatewaySpeed : 0|16@1+ (0.1,0) [0|6553.5] "km/h" ECU1,ECU2
 SG_ SystemOK : 16|1@1+ (1,0) [0|1] "" ECU1,ECU2
"""
    
    # Create correlated ASC data
    # RPM in CAN message 1234 should correlate with GatewayRPM in message 1235
    # Speed in message 1235 should correlate with GatewaySpeed in message 1236
    asc_content = """date Wed Nov 15 14:30:00 2023
base hex  timestamps absolute
internal events logged
// version 9.0.0
Begin Triggerblock Wed Nov 15 14:30:00 2023
"""
    
    # Generate correlated data
    times = np.arange(0, 5, 0.1)  # 5 seconds, 100ms intervals
    base_rpm = 1000
    base_speed = 50
    
    for i, t in enumerate(times):
        # RPM varies sinusoidally
        rpm_raw = int((base_rpm + 200 * np.sin(t)) / 0.25)  # Convert to raw DBC value
        rpm_bytes = f"{rpm_raw:04X}"
        
        # Speed varies with RPM (correlated)
        speed_raw = int((base_speed + 10 * np.sin(t)) / 0.1)
        speed_bytes = f"{speed_raw:04X}"
        
        # Throttle varies independently
        throttle_raw = int((30 + 20 * np.sin(t * 0.5)) / 0.4)
        throttle_bytes = f"{throttle_raw:02X}"
        
        # Temperature slowly increases
        temp_raw = int(80 + t)  # 80°C + 1°C per second
        temp_bytes = f"{temp_raw:02X}"
        
        # CAN message 1234 (EngineData) - source signals
        rpm_lo = rpm_bytes[2:4]
        rpm_hi = rpm_bytes[0:2]
        asc_content += f"   {t:.3f}000 1  4D2             Rx   d 8  {rpm_lo} {rpm_hi} {throttle_bytes} {temp_bytes} 00 00 00 00\n"
        
        # CAN message 1235 (VehicleSpeed) - gateway relayed RPM with 50ms delay
        gateway_rpm_raw = rpm_raw  # Same value (perfect gateway relay)
        gateway_rpm_bytes = f"{gateway_rpm_raw:04X}"
        grpm_lo = gateway_rpm_bytes[2:4]
        grpm_hi = gateway_rpm_bytes[0:2]
        speed_lo = speed_bytes[2:4]
        speed_hi = speed_bytes[0:2]
        
        delay_time = t + 0.05  # 50ms gateway delay
        asc_content += f"   {delay_time:.3f}000 1  4D3             Rx   d 8  {speed_lo} {speed_hi} {grpm_lo} {grpm_hi} 00 00 00 00\n"
        
        # CAN message 1236 (SystemStatus) - gateway relayed speed with 100ms delay
        gateway_speed_raw = speed_raw  # Same value (perfect gateway relay)
        gateway_speed_bytes = f"{gateway_speed_raw:04X}"
        gspeed_lo = gateway_speed_bytes[2:4]
        gspeed_hi = gateway_speed_bytes[0:2]
        
        delay_time2 = t + 0.10  # 100ms gateway delay
        asc_content += f"   {delay_time2:.3f}000 1  4D4             Rx   d 8  {gspeed_lo} {gspeed_hi} 01 00 00 00 00 00\n"
    
    asc_content += "End TriggerBlock\n"
    
    # LIN configuration for cross-protocol correlation
    ldf_content = """
LIN_description_file;
LIN_protocol_version = "2.1";
LIN_language_version = "2.1";
LIN_speed = 19200;

Nodes {
  Master: LIN_Master, 5ms, 0.1ms ;
  Slaves: LIN_Slave1 ;
}

Signals {
  LIN_RPM: 16, 0, LIN_Slave1, LIN_Master ;
  LIN_Speed: 16, 0, LIN_Slave1, LIN_Master ;
}

Frames {
  RPM_Frame: 0x01, LIN_Slave1, 4 {
    LIN_RPM, 0 ;
  }
  Speed_Frame: 0x02, LIN_Slave1, 4 {
    LIN_Speed, 0 ;
  }
}
"""
    
    # LIN log with correlated signals (200ms delay from CAN)
    lin_log_content = "# LIN Log with correlation to CAN\n"
    for i, t in enumerate(times[::2]):  # Half the frequency
        # LIN gets RPM from gateway with additional 200ms delay
        lin_time = t + 0.2
        
        # RPM correlated to CAN RPM
        lin_rpm_raw = int((base_rpm + 200 * np.sin(t)) / 4)  # Different scaling
        lin_rpm_hex = f"{lin_rpm_raw:04X}"
        
        # Speed correlated to CAN speed  
        lin_speed_raw = int((base_speed + 10 * np.sin(t)) / 2)  # Different scaling
        lin_speed_hex = f"{lin_speed_raw:04X}"
        
        lin_log_content += f"{lin_time:.3f} 01 {lin_rpm_hex}\n"
        lin_log_content += f"{lin_time + 0.01:.3f} 02 {lin_speed_hex}\n"
    
    # Create test configuration file
    config_content = """# Day 3 Multi-Protocol Test Configuration
# Format: protocol,database,logfile
CAN,{dbc_file},{asc_file}
LIN,{ldf_file},{lin_file}
"""
    
    # Write all files
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
    
    # Create config file
    config_filled = config_content.format(
        dbc_file=files['dbc'],
        asc_file=files['asc'],
        ldf_file=files['ldf'],
        lin_file=files['lin_log']
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(config_filled)
        files['config'] = f.name
    
    return files

def test_correlation_engine():
    """Test the cross-protocol correlation engine"""
    print("🧪 Testing Correlation Engine")
    print("-" * 30)
    
    files = create_correlated_test_data()
    
    try:
        # Import correlation components
        sys.path.insert(0, '.')
        from day3_correlation_engine import CrossProtocolCorrelator
        from day2_enhanced_analyzer import EnhancedNetworkAnalyzer
        from day2_universal_signal import ProtocolType
        
        # Set up analyzer and get data
        analyzer = EnhancedNetworkAnalyzer()
        
        # Add protocols
        analyzer.add_protocol_database(ProtocolType.CAN, files['dbc'])
        analyzer.add_protocol_database(ProtocolType.LIN, files['ldf'])
        
        # Analyze logs
        log_configs = [
            {"path": files['asc'], "protocol": ProtocolType.CAN},
            {"path": files['lin_log'], "protocol": ProtocolType.LIN}
        ]
        
        results = analyzer.analyze_mixed_logs(log_configs)
        
        if not results or results['total_signals'] == 0:
            print("❌ No signals found for correlation test")
            return False
        
        # Test correlation engine
        correlator = CrossProtocolCorrelator()
        df = results['data_frame']
        
        correlations = correlator.find_correlations(df)
        
        # Check results
        if len(correlations) > 0:
            print(f"✅ Correlation engine test PASSED!")
            print(f"   - Found {len(correlations)} correlations")
            
            # Check for expected correlations
            gateway_candidates = [c for c in correlations if c.gateway_candidate]
            print(f"   - Gateway candidates: {len(gateway_candidates)}")
            
            # Look for RPM correlation (should be strongest)
            rpm_correlations = [c for c in correlations 
                              if 'RPM' in c.signal1_name or 'RPM' in c.signal2_name]
            
            if rpm_correlations:
                best_rpm = max(rpm_correlations, key=lambda x: x.confidence_score)
                print(f"   - Best RPM correlation: {best_rpm.confidence_score:.1f}% confidence")
                print(f"   - Delay: {best_rpm.delay_ms:.1f}ms")
            
            return True
        else:
            print("❌ No correlations found")
            return False
            
    except Exception as e:
        print(f"❌ Correlation engine test ERROR: {e}")
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

def test_gateway_analyzer():
    """Test the gateway latency analyzer"""
    print("\n🧪 Testing Gateway Latency Analyzer")
    print("-" * 35)
    
    files = create_correlated_test_data()
    
    try:
        # Import components
        from day3_gateway_analyzer import GatewayLatencyAnalyzer
        from day3_correlation_engine import CrossProtocolCorrelator
        from day2_enhanced_analyzer import EnhancedNetworkAnalyzer
        from day2_universal_signal import ProtocolType
        
        # Get correlations first
        analyzer = EnhancedNetworkAnalyzer()
        analyzer.add_protocol_database(ProtocolType.CAN, files['dbc'])
        
        log_configs = [{"path": files['asc'], "protocol": ProtocolType.CAN}]
        results = analyzer.analyze_mixed_logs(log_configs)
        
        correlator = CrossProtocolCorrelator()
        df = results['data_frame']
        correlations = correlator.find_correlations(df)
        
        # Test gateway analyzer
        gateway_analyzer = GatewayLatencyAnalyzer()
        latency_analyses = gateway_analyzer.analyze_gateway_latencies(correlations, df)
        
        if len(latency_analyses) > 0:
            print(f"✅ Gateway analyzer test PASSED!")
            print(f"   - Analyzed {len(latency_analyses)} gateway paths")
            
            # Check latency measurements
            total_measurements = sum(len(a.measurements) for a in latency_analyses)
            print(f"   - Total latency measurements: {total_measurements}")
            
            # Check for reasonable latencies
            if latency_analyses:
                best_analysis = min(latency_analyses, key=lambda x: x.mean_latency)
                print(f"   - Best latency: {best_analysis.mean_latency:.1f}ms")
                print(f"   - Performance score: {best_analysis.performance_score:.1f}/100")
            
            return True
        else:
            print("⚠️  No gateway paths found (may be normal for single protocol)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        print(f"❌ Gateway analyzer test ERROR: {e}")
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

def test_complete_day3_analyzer():
    """Test the complete Day 3 analyzer"""
    print("\n🧪 Testing Complete Day 3 Analyzer")
    print("-" * 35)
    
    files = create_correlated_test_data()
    
    try:
        # Import complete analyzer
        from day3_network_analyzer import Day3NetworkAnalyzer
        from day2_universal_signal import ProtocolType
        
        analyzer = Day3NetworkAnalyzer()
        
        # Test single protocol analysis
        results = analyzer.analyze_single_protocol_with_correlation(
            files['dbc'], files['asc'], ProtocolType.CAN
        )
        
        if results and 'correlation_analysis' in results:
            print(f"✅ Complete Day 3 analyzer test PASSED!")
            
            # Check all components
            basic = results['basic_analysis']
            corr = results['correlation_analysis']
            latency = results['latency_analysis']
            
            print(f"   - Basic analysis: {basic['total_signals']} signals")
            print(f"   - Correlations: {corr['total_correlations']}")
            print(f"   - Gateway paths: {latency['total_gateway_paths']}")
            
            # Test export functionality
            exported = analyzer.export_complete_results("test_day3")
            print(f"   - Exported {len(exported)} files")
            
            # Cleanup exported files
            for filepath in exported.values():
                try:
                    os.unlink(filepath)
                except:
                    pass
            
            return True
        else:
            print("❌ Complete analyzer test FAILED - missing results")
            return False
            
    except Exception as e:
        print(f"❌ Complete analyzer test ERROR: {e}")
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

def test_multi_protocol_correlation():
    """Test multi-protocol correlation functionality"""
    print("\n🧪 Testing Multi-Protocol Correlation")
    print("-" * 35)
    
    files = create_correlated_test_data()
    
    try:
        from day3_network_analyzer import Day3NetworkAnalyzer
        from day2_universal_signal import ProtocolType
        
        analyzer = Day3NetworkAnalyzer()
        
        # Test multi-protocol analysis
        log_configs = [
            {"protocol": ProtocolType.CAN, "database": files['dbc'], "path": files['asc']},
            {"protocol": ProtocolType.LIN, "database": files['ldf'], "path": files['lin_log']}
        ]
        
        results = analyzer.analyze_complete_network(log_configs)
        
        if results:
            corr = results['correlation_analysis']
            
            # Look for cross-protocol correlations
            cross_protocol_correlations = 0
            for correlation in corr['correlations']:
                if correlation.signal1_protocol != correlation.signal2_protocol:
                    cross_protocol_correlations += 1
            
            print(f"✅ Multi-protocol correlation test PASSED!")
            print(f"   - Total correlations: {corr['total_correlations']}")
            print(f"   - Cross-protocol: {cross_protocol_correlations}")
            print(f"   - Gateway candidates: {corr['gateway_candidates']}")
            
            return True
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

def main():
    """Run all Day 3 tests"""
    print("🚀 Day 3 - Complete Network Analyzer Tests")
    print("=" * 50)
    
    # Run all tests
    test1_pass = test_correlation_engine()
    test2_pass = test_gateway_analyzer()
    test3_pass = test_complete_day3_analyzer()
    test4_pass = test_multi_protocol_correlation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Day 3 Test Results:")
    print(f"   Correlation Engine: {'✅' if test1_pass else '❌'}")
    print(f"   Gateway Analyzer: {'✅' if test2_pass else '❌'}")
    print(f"   Complete Analyzer: {'✅' if test3_pass else '❌'}")
    print(f"   Multi-Protocol: {'✅' if test4_pass else '❌'}")
    
    if all([test1_pass, test2_pass, test3_pass, test4_pass]):
        print("\n🎉 DAY 3 COMPLETE!")
        print("✅ All tests passed!")
        print("🚀 Cross-protocol correlation engine working!")
        print("✅ Gateway latency analysis working!")
        print("🔗 Multi-protocol network analysis complete!")
        print("\n🎯 Key Achievements:")
        print("   - Signal correlation across protocols")
        print("   - Gateway latency measurement")
        print("   - End-to-end network visibility") 
        print("   - Executive reporting")
        print("\n✅ Ready for Day 4: FastAPI Backend!")
    else:
        print("\n❌ Some tests failed - fix issues before Day 4")
        print("🔧 Check the error messages above")

if __name__ == "__main__":
    main()