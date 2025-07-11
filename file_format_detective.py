#!/usr/bin/env python3
"""
Automotive File Format Detective
Identify and analyze unknown automotive file formats
"""

import sys
import os
from pathlib import Path
import re

def detect_log_format(file_path):
    """Detect the format of a log file"""
    print(f"🔍 Detecting format for: {file_path}")
    
    try:
        # Try to read as text first
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [f.readline().strip() for _ in range(50)]  # First 50 lines
    except UnicodeDecodeError:
        try:
            # Try latin-1 encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = [f.readline().strip() for _ in range(50)]
        except:
            print("❌ Cannot read file as text - possibly binary format")
            return "BINARY"
    
    # Remove empty lines
    lines = [line for line in lines if line.strip()]
    
    if not lines:
        print("❌ File appears to be empty")
        return "EMPTY"
    
    # Check for common formats
    format_indicators = {
        'ASC': ['date', 'base hex', 'timestamps absolute', 'Rx', 'Tx'],
        'TRC': ['TRC', 'PEAK', 'PCAN'],
        'BLF': ['BLF', 'Vector'],
        'CSV': [',', 'timestamp', 'id', 'data'],
        'CANDUMP': ['can0', 'can1', 'vcan']
    }
    
    scores = {fmt: 0 for fmt in format_indicators.keys()}
    
    # Score each format
    for line in lines:
        line_lower = line.lower()
        for fmt, indicators in format_indicators.items():
            for indicator in indicators:
                if indicator.lower() in line_lower:
                    scores[fmt] += 1
    
    # Find best match
    best_format = max(scores, key=scores.get)
    best_score = scores[best_format]
    
    print(f"📊 Format Detection Scores:")
    for fmt, score in scores.items():
        print(f"   {fmt}: {score}")
    
    if best_score > 0:
        print(f"🎯 Detected Format: {best_format}")
    else:
        print(f"❓ Unknown format")
        best_format = "UNKNOWN"
    
    # Show sample lines
    print(f"\n📋 Sample Lines:")
    for i, line in enumerate(lines[:10]):
        print(f"   {i+1:2d}: {line}")
    
    return best_format

def analyze_asc_structure(file_path):
    """Analyze ASC file structure in detail"""
    print(f"\n🔬 Detailed ASC Analysis")
    
    try:
        with open(file_path, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 100:  # First 100 lines
                    break
                lines.append(line.strip())
        
        # Find actual data lines
        data_lines = []
        for line in lines:
            if not line:
                continue
            if line.startswith(('date', 'base', '//', 'Begin', 'End', 'internal')):
                continue
            
            parts = line.split()
            if len(parts) >= 6:  # Minimum for CAN message
                try:
                    # Try to parse as timestamp
                    float(parts[0])
                    data_lines.append(line)
                except ValueError:
                    continue
        
        print(f"📊 Found {len(data_lines)} potential data lines")
        
        if data_lines:
            print(f"\n🎯 Sample Data Lines:")
            for i, line in enumerate(data_lines[:5]):
                parts = line.split()
                print(f"   {i+1}: {line}")
                if len(parts) >= 6:
                    print(f"      → Timestamp: {parts[0]}")
                    print(f"      → Channel: {parts[1]}")
                    print(f"      → CAN ID: {parts[2]}")
                    print(f"      → Direction: {parts[3] if len(parts) > 3 else 'N/A'}")
                    print(f"      → Type: {parts[4] if len(parts) > 4 else 'N/A'}")
                    print(f"      → DLC: {parts[5] if len(parts) > 5 else 'N/A'}")
                    print(f"      → Data: {' '.join(parts[6:]) if len(parts) > 6 else 'N/A'}")
        
        return len(data_lines) > 0
        
    except Exception as e:
        print(f"❌ ASC analysis failed: {e}")
        return False

def check_dbc_compatibility(dbc_path, log_path):
    """Check if DBC and log files are compatible"""
    print(f"\n🔗 Checking DBC-Log Compatibility")
    
    try:
        # Load DBC
        import cantools
        db = cantools.database.load_file(dbc_path)
        dbc_ids = set(msg.frame_id for msg in db.messages)
        print(f"📊 DBC has {len(dbc_ids)} message IDs")
        
        # Extract CAN IDs from log
        log_ids = set()
        with open(log_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        can_id = int(parts[2], 16)
                        log_ids.add(can_id)
                    except ValueError:
                        continue
        
        print(f"📊 Log has {len(log_ids)} unique CAN IDs")
        
        # Find matches
        matches = dbc_ids.intersection(log_ids)
        print(f"🎯 Matching IDs: {len(matches)}")
        
        if matches:
            print(f"✅ Compatible! Found {len(matches)} matching CAN IDs")
            # Show some matches
            for i, can_id in enumerate(sorted(matches)[:10]):
                msg = db.get_message_by_frame_id(can_id)
                print(f"   0x{can_id:X}: {msg.name} ({len(msg.signals)} signals)")
        else:
            print(f"❌ No matching CAN IDs found!")
            print(f"   DBC IDs (first 10): {sorted(list(dbc_ids))[:10]}")
            print(f"   Log IDs (first 10): {sorted(list(log_ids))[:10]}")
        
        return len(matches) > 0
        
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return False

def main():
    """Main detective function"""
    if len(sys.argv) < 2:
        print("🕵️ Automotive File Format Detective")
        print("Usage: python file_format_detective.py <file1> [file2]")
        print("")
        print("Examples:")
        print("  python file_format_detective.py mystery_log.txt")
        print("  python file_format_detective.py vehicle.dbc trace.asc")
        return
    
    file1 = sys.argv[1]
    
    if not Path(file1).exists():
        print(f"❌ File not found: {file1}")
        return
    
    # Detect format of first file
    format1 = detect_log_format(file1)
    
    # If it's ASC, do detailed analysis
    if format1 == "ASC":
        analyze_asc_structure(file1)
    
    # If two files provided, check compatibility
    if len(sys.argv) >= 3:
        file2 = sys.argv[2]
        if Path(file2).exists():
            format2 = detect_log_format(file2)
            
            # Check if we have DBC + Log combination
            if (file1.endswith('.dbc') and format2 == "ASC") or (file2.endswith('.dbc') and format1 == "ASC"):
                dbc_file = file1 if file1.endswith('.dbc') else file2
                log_file = file2 if file1.endswith('.dbc') else file1
                check_dbc_compatibility(dbc_file, log_file)
        else:
            print(f"❌ Second file not found: {file2}")

if __name__ == "__main__":
    main()