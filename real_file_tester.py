#!/usr/bin/env python3
"""
Real Automotive File Tester
Test the analyzer with actual automotive files and provide detailed diagnostics
"""

import sys
import os
from pathlib import Path
import time
from can_analyzer import SimpleCANAnalyzer

def analyze_dbc_file(dbc_path):
    """Analyze DBC file structure"""
    print(f"\n🔍 Analyzing DBC: {dbc_path}")
    
    try:
        import cantools
        db = cantools.database.load_file(dbc_path)
        
        print(f"  📊 Messages: {len(db.messages)}")
        print(f"  📡 Nodes: {len(db.nodes)}")
        
        # Show top 10 messages
        print(f"\n  🎯 Top 10 Messages:")
        for i, msg in enumerate(db.messages[:10]):
            signal_count = len(msg.signals)
            print(f"    {msg.name} (ID: 0x{msg.frame_id:X}): {signal_count} signals")
        
        # Count total signals
        total_signals = sum(len(msg.signals) for msg in db.messages)
        print(f"  📈 Total Signals: {total_signals}")
        
        return db
        
    except Exception as e:
        print(f"  ❌ DBC Error: {e}")
        return None

def analyze_log_file(log_path):
    """Analyze log file structure"""
    print(f"\n🔍 Analyzing Log: {log_path}")
    
    try:
        file_size = os.path.getsize(log_path) / (1024 * 1024)  # MB
        print(f"  📁 File Size: {file_size:.2f} MB")
        
        # Read first few lines to understand format
        with open(log_path, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 20:  # First 20 lines
                    break
                lines.append(line.strip())
        
        print(f"  📋 First 20 lines:")
        for i, line in enumerate(lines):
            print(f"    {i+1:2d}: {line}")
        
        # Try to detect format
        has_timestamps = any('.' in line.split()[0] if line.split() else False for line in lines if line and not line.startswith(('date', 'base', '//', 'Begin', 'End', 'internal')))
        has_hex_ids = any(any(c in '0123456789ABCDEF' for c in part) for line in lines for part in line.split()[2:3] if line.split() and len(line.split()) > 2)
        
        print(f"  🔍 Format Detection:")
        print(f"    Timestamps detected: {has_timestamps}")
        print(f"    Hex IDs detected: {has_hex_ids}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Log Error: {e}")
        return False

def test_real_files(dbc_path, log_path):
    """Test with real automotive files"""
    print("🚀 Testing with Real Automotive Files")
    print("=" * 50)
    
    # Validate files exist
    if not Path(dbc_path).exists():
        print(f"❌ DBC file not found: {dbc_path}")
        return False
    
    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return False
    
    # Analyze files first
    db = analyze_dbc_file(dbc_path)
    if not db:
        return False
    
    log_ok = analyze_log_file(log_path)
    if not log_ok:
        return False
    
    # Run the analyzer
    print(f"\n🔬 Running Analysis...")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        analyzer = SimpleCANAnalyzer()
        df = analyzer.analyze_file(dbc_path, log_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if df.empty:
            print("❌ No data extracted - possible format mismatch")
            return False
        
        # Success! Show detailed results
        print(f"\n✅ SUCCESS! Analysis Complete")
        print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"   📊 Signal samples: {len(df)}")
        print(f"   🎯 Unique signals: {df['signal'].nunique()}")
        print(f"   📨 Unique messages: {df['message'].nunique()}")
        print(f"   ⏰ Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}s")
        print(f"   📈 Duration: {df['timestamp'].max() - df['timestamp'].min():.2f}s")
        
        # Show signal statistics
        print(f"\n🎯 Most Active Signals:")
        signal_counts = df['signal'].value_counts().head(10)
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} samples")
        
        # Show message statistics
        print(f"\n📨 Most Active Messages:")
        msg_counts = df['message'].value_counts().head(10)
        for msg, count in msg_counts.items():
            print(f"   {msg}: {count} samples")
        
        # Show sample data
        print(f"\n📋 Sample Data (first 10 rows):")
        print(df.head(10).to_string(index=False))
        
        # Save results
        output_file = f"real_test_results_{int(time.time())}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    if len(sys.argv) != 3:
        print("🧪 Real Automotive File Tester")
        print("Usage: python real_file_tester.py <dbc_file> <log_file>")
        print("")
        print("Examples:")
        print("  python real_file_tester.py vehicle.dbc trace.asc")
        print("  python real_file_tester.py my_car.dbc log_data.asc")
        print("")
        print("Common automotive file formats:")
        print("  DBC: .dbc (CAN database)")
        print("  ASC: .asc (Vector ASCII log)")
        print("  BLF: .blf (Vector Binary log) - not yet supported")
        print("  TRC: .trc (PEAK trace) - not yet supported")
        return
    
    dbc_file = sys.argv[1]
    log_file = sys.argv[2]
    
    success = test_real_files(dbc_file, log_file)
    
    if success:
        print("\n🎉 REAL FILE TEST PASSED!")
        print("✅ Your analyzer works with real automotive data!")
        print("🚀 Ready for Day 2!")
    else:
        print("\n🔧 Issues found - let's debug...")
        print("Common issues:")
        print("  1. Log format not ASC (try converting with Vector tools)")
        print("  2. DBC doesn't match log CAN IDs")
        print("  3. File encoding issues (try UTF-8)")
        print("  4. Timestamps not in expected format")

if __name__ == "__main__":
    main()