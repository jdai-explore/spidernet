#!/usr/bin/env python3
"""
CAN Log Parser: Parse DBC file and read CAN log, extract signals
"""

import cantools
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Any

class SimpleCANAnalyzer:
     
    def __init__(self):
        self.database = None
        self.signals = []
        
    def load_dbc(self, dbc_path: str) -> bool:
        """Load DBC database file"""
        try:
            self.database = cantools.database.load_file(dbc_path)
            print(f"✅ Loaded DBC: {len(self.database.messages)} messages")
            return True
        except Exception as e:
            print(f"❌ DBC load failed: {e}")
            return False
    
    def parse_asc_log(self, log_path: str) -> List[Dict]:
        """Parse ASC log file - most common format"""
        messages = []
        
        try:
            with open(log_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    
                    # Skip headers, empty lines, and comments
                    if not line or line.startswith('date') or line.startswith('base') or line.startswith('//') or line.startswith('Begin') or line.startswith('End') or line.startswith('internal'):
                        continue
                    
                    # Parse ASC format: timestamp channel id rx/tx d dlc data
                    parts = line.split()
                    if len(parts) < 8:  # Need at least: time chan id rx d dlc data...
                        continue
                    
                    try:
                        # ASC format: timestamp channel CAN_ID Rx d dlc data_bytes
                        timestamp = float(parts[0])
                        can_id = int(parts[2], 16)  # CAN ID in hex
                        # Skip parts[3] (Rx/Tx) and parts[4] ('d' indicator)
                        dlc = int(parts[5])
                        
                        # Get data bytes - they start at parts[6]
                        data_parts = parts[6:6+dlc]
                        if len(data_parts) < dlc:
                            continue  # Not enough data bytes
                        
                        # Convert hex strings to bytes
                        data = bytes([int(b, 16) for b in data_parts])
                        
                        messages.append({
                            'timestamp': timestamp,
                            'can_id': can_id,
                            'dlc': dlc,
                            'data': data,
                            'line_num': line_num
                        })
                        
                    except (ValueError, IndexError) as e:
                        print(f"Debug: Skipping line {line_num}: {line} - Error: {e}")
                        continue  # Skip malformed lines
            
            print(f"✅ Parsed {len(messages)} CAN messages")
            return messages
            
        except Exception as e:
            print(f"❌ Log parsing failed: {e}")
            return []
    
    def extract_signals(self, messages: List[Dict]) -> List[Dict]:
        """Extract signal values from CAN messages"""
        if not self.database:
            print("❌ No DBC loaded")
            return []
        
        signals = []
        
        for msg in messages:
            try:
                # Find message definition in DBC
                db_message = self.database.get_message_by_frame_id(msg['can_id'])
                
                # Decode message data
                decoded = db_message.decode(msg['data'])
                
                # Extract each signal
                for signal_name, value in decoded.items():
                    signals.append({
                        'timestamp': msg['timestamp'],
                        'message': db_message.name,
                        'signal': signal_name,
                        'value': value,
                        'can_id': msg['can_id']
                    })
                    
            except (KeyError, cantools.database.DecodeError):
                # Message not in DBC or decode error - skip
                continue
        
        print(f"✅ Extracted {len(signals)} signal values")
        return signals
    
    def analyze_file(self, dbc_path: str, log_path: str) -> pd.DataFrame:
        """Complete analysis pipeline"""
        print(f"🔍 Analyzing: {log_path}")
        
        # Load DBC
        if not self.load_dbc(dbc_path):
            return pd.DataFrame()
        
        # Parse log
        messages = self.parse_asc_log(log_path)
        if not messages:
            return pd.DataFrame()
        
        # Extract signals
        signals = self.extract_signals(messages)
        if not signals:
            return pd.DataFrame()
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(signals)
        
        # Basic statistics
        print(f"\n📊 Analysis Results:")
        print(f"Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}s")
        print(f"Unique messages: {df['message'].nunique()}")
        print(f"Unique signals: {df['signal'].nunique()}")
        
        # Signal summary
        print(f"\n🎯 Top 10 Most Active Signals:")
        signal_counts = df['signal'].value_counts().head(10)
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count} samples")
        
        return df

def main():
    """Day 1 demo - simple command line interface"""
    if len(sys.argv) != 3:
        print("Usage: python can_analyzer.py <dbc_file> <asc_log_file>")
        print("Example: python can_analyzer.py vehicle.dbc trace.asc")
        return
    
    dbc_file = sys.argv[1]
    log_file = sys.argv[2]
    
    # Validate files exist
    if not Path(dbc_file).exists():
        print(f"❌ DBC file not found: {dbc_file}")
        return
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    # Run analysis
    analyzer = SimpleCANAnalyzer()
    df = analyzer.analyze_file(dbc_file, log_file)
    
    if not df.empty:
        print(f"\n✅ SUCCESS! Analyzed {len(df)} signal samples")
        print(f"💾 Data ready for further analysis")
        
        # Save results for verification
        output_file = "day1_results.csv"
        df.to_csv(output_file, index=False)
        print(f"📁 Results saved to: {output_file}")
    else:
        print("❌ Analysis failed - check your files")

if __name__ == "__main__":
    main()