#!/usr/bin/env python3
"""
Day 2: Enhanced Multi-Protocol Analyzer
Combines Day 1 foundation with signal quality and multi-protocol support
"""

import sys
import time
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

# Import our Day 2 components
from day2_universal_signal import (
    MultiProtocolAnalyzer, ProtocolType, ProtocolFactory
)
from day2_signal_quality import SignalQualityAnalyzer

class EnhancedNetworkAnalyzer:
    """Day 2 - Enhanced automotive network analyzer"""
    
    def __init__(self):
        self.multi_protocol = MultiProtocolAnalyzer()
        self.quality_analyzer = SignalQualityAnalyzer()
        self.analysis_results = {}
    
    def add_protocol_database(self, protocol: ProtocolType, db_path: str) -> bool:
        """Add protocol database (DBC, LDF, etc.)"""
        return self.multi_protocol.add_protocol(protocol, db_path)
    
    def analyze_mixed_logs(self, log_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple log files with different protocols
        log_configs: [{"path": "file.asc", "protocol": ProtocolType.CAN}, ...]
        """
        print("🚀 Day 2 - Enhanced Multi-Protocol Analysis")
        print("=" * 50)
        
        start_time = time.time()
        all_signals = []
        
        # Process each log file
        for i, config in enumerate(log_configs):
            log_path = config["path"]
            protocol = config.get("protocol", ProtocolType.CAN)
            
            print(f"\n📁 Processing {log_path} ({protocol.value})")
            
            if not Path(log_path).exists():
                print(f"❌ File not found: {log_path}")
                continue
            
            # Analyze log
            signals = self.multi_protocol.analyze_log(log_path, protocol)
            all_signals.extend(signals)
        
        if not all_signals:
            print("❌ No signals extracted from any log file")
            return {}
        
        # Convert to DataFrame for analysis
        df = self.multi_protocol.get_all_signals_df()
        
        # Perform quality analysis
        print(f"\n🔍 Analyzing signal quality...")
        quality_results = self.quality_analyzer.analyze_signal_quality(df)
        
        # Generate analysis results
        end_time = time.time()
        processing_time = end_time - start_time
        
        results = {
            'processing_time': processing_time,
            'total_signals': len(df),
            'protocols': df['protocol'].unique().tolist(),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration': df['timestamp'].max() - df['timestamp'].min()
            },
            'signal_statistics': self._generate_signal_statistics(df),
            'quality_analysis': quality_results,
            'protocol_summary': self.multi_protocol.get_protocol_summary(),
            'data_frame': df
        }
        
        self.analysis_results = results
        self._print_analysis_summary(results)
        
        return results
    
    def analyze_single_protocol(self, db_path: str, log_path: str, 
                              protocol: ProtocolType = ProtocolType.CAN) -> Dict[str, Any]:
        """Analyze single protocol - backward compatible with Day 1"""
        print(f"🔍 Day 2 - Single Protocol Analysis ({protocol.value})")
        print("=" * 40)
        
        # Add database
        if not self.add_protocol_database(protocol, db_path):
            return {}
        
        # Analyze log
        return self.analyze_mixed_logs([{"path": log_path, "protocol": protocol}])
    
    def _generate_signal_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive signal statistics"""
        stats = {
            'total_samples': len(df),
            'unique_signals': df['signal'].nunique(),
            'unique_messages': df['message'].nunique(),
            'protocols_used': df['protocol'].nunique()
        }
        
        # Per-protocol breakdown
        protocol_stats = {}
        for protocol in df['protocol'].unique():
            proto_df = df[df['protocol'] == protocol]
            protocol_stats[protocol] = {
                'samples': len(proto_df),
                'signals': proto_df['signal'].nunique(),
                'messages': proto_df['message'].nunique()
            }
        
        stats['per_protocol'] = protocol_stats
        
        # Most active signals
        signal_counts = df['signal'].value_counts().head(10)
        stats['most_active_signals'] = signal_counts.to_dict()
        
        # Most active messages
        message_counts = df['message'].value_counts().head(10)
        stats['most_active_messages'] = message_counts.to_dict()
        
        return stats
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Print comprehensive analysis summary"""
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"   ⏱️  Processing time: {results['processing_time']:.2f} seconds")
        print(f"   📊 Total signal samples: {results['total_signals']:,}")
        print(f"   🎯 Unique signals: {results['signal_statistics']['unique_signals']}")
        print(f"   📨 Unique messages: {results['signal_statistics']['unique_messages']}")
        print(f"   🌐 Protocols: {', '.join(results['protocols'])}")
        print(f"   ⏰ Time range: {results['time_range']['start']:.2f} - {results['time_range']['end']:.2f}s")
        print(f"   📈 Duration: {results['time_range']['duration']:.2f}s")
        
        # Protocol breakdown
        print(f"\n🌐 Per-Protocol Breakdown:")
        for protocol, stats in results['signal_statistics']['per_protocol'].items():
            print(f"   {protocol}: {stats['samples']:,} samples, {stats['signals']} signals")
        
        # Quality summary
        if results['quality_analysis']:
            quality_scores = [q.quality_score for q in results['quality_analysis'].values()]
            avg_quality = sum(quality_scores) / len(quality_scores)
            poor_signals = sum(1 for s in quality_scores if s < 50)
            
            print(f"\n📊 Signal Quality Summary:")
            print(f"   Average quality: {avg_quality:.1f}/100")
            print(f"   Signals needing attention: {poor_signals}")
        
        # Top signals
        print(f"\n🎯 Most Active Signals:")
        for signal, count in list(results['signal_statistics']['most_active_signals'].items())[:5]:
            print(f"   {signal}: {count:,} samples")
    
    def generate_enhanced_report(self) -> str:
        """Generate comprehensive Day 2 report"""
        if not self.analysis_results:
            return "No analysis results available"
        
        results = self.analysis_results
        report = []
        
        report.append("🚀 DAY 2 - ENHANCED NETWORK ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Executive Summary
        report.append(f"\n📈 EXECUTIVE SUMMARY")
        report.append(f"   Analysis completed in {results['processing_time']:.2f} seconds")
        report.append(f"   Processed {results['total_signals']:,} signal samples")
        report.append(f"   Analyzed {len(results['protocols'])} protocol(s): {', '.join(results['protocols'])}")
        report.append(f"   Time span: {results['time_range']['duration']:.2f} seconds")
        
        # Protocol Details
        report.append(f"\n🌐 PROTOCOL ANALYSIS")
        for protocol, stats in results['signal_statistics']['per_protocol'].items():
            report.append(f"   {protocol}:")
            report.append(f"     - {stats['samples']:,} signal samples")
            report.append(f"     - {stats['signals']} unique signals")
            report.append(f"     - {stats['messages']} unique messages")
        
        # Signal Quality Report
        if results['quality_analysis']:
            quality_report = self.quality_analyzer.generate_quality_report(results['quality_analysis'])
            report.append(f"\n{quality_report}")
        
        # Top Performers
        report.append(f"\n🏆 TOP SIGNAL ACTIVITY")
        for signal, count in list(results['signal_statistics']['most_active_signals'].items())[:10]:
            report.append(f"   {signal}: {count:,} samples")
        
        return "\n".join(report)
    
    def export_results(self, output_prefix: str = "day2_analysis") -> Dict[str, str]:
        """Export analysis results to files"""
        if not self.analysis_results:
            return {}
        
        exported_files = {}
        timestamp = int(time.time())
        
        # Export main data
        df = self.analysis_results['data_frame']
        csv_file = f"{output_prefix}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        exported_files['data'] = csv_file
        
        # Export quality analysis
        if self.analysis_results['quality_analysis']:
            quality_data = []
            for signal_name, quality in self.analysis_results['quality_analysis'].items():
                quality_data.append({
                    'signal': quality.signal_name,
                    'message': quality.message_name,
                    'quality_score': quality.quality_score,
                    'samples': quality.total_samples,
                    'coverage': quality.time_coverage,
                    'update_rate': quality.update_rate,
                    'stuck_percentage': quality.stuck_percentage,
                    'issues': ', '.join([i.value for i in quality.issues])
                })
            
            quality_df = pd.DataFrame(quality_data)
            quality_file = f"{output_prefix}_quality_{timestamp}.csv"
            quality_df.to_csv(quality_file, index=False)
            exported_files['quality'] = quality_file
        
        # Export comprehensive report
        report = self.generate_enhanced_report()
        report_file = f"{output_prefix}_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        exported_files['report'] = report_file
        
        return exported_files

def main():
    """Day 2 command line interface"""
    if len(sys.argv) < 3:
        print("🚀 Day 2 - Enhanced Multi-Protocol Network Analyzer")
        print("=" * 55)
        print("\nUsage Options:")
        print("1. Single Protocol (CAN):")
        print("   python day2_analyzer.py <dbc_file> <asc_file>")
        print("\n2. Multi-Protocol:")
        print("   python day2_analyzer.py multi <config_file>")
        print("\nExamples:")
        print("   python day2_analyzer.py vehicle.dbc trace.asc")
        print("   python day2_analyzer.py multi analysis_config.txt")
        print("\nConfig file format (one per line):")
        print("   CAN,database.dbc,log1.asc")
        print("   LIN,lin_config.ldf,lin_log.txt")
        return
    
    analyzer = EnhancedNetworkAnalyzer()
    
    if sys.argv[1].lower() == "multi":
        # Multi-protocol mode
        config_file = sys.argv[2]
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            return
        
        # Parse config file
        log_configs = []
        protocol_dbs = {}
        
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 3:
                    protocol_str = parts[0].strip().upper()
                    db_path = parts[1].strip()
                    log_path = parts[2].strip()
                    
                    try:
                        protocol = ProtocolType[protocol_str]
                        
                        # Add database if not already added
                        if protocol not in protocol_dbs:
                            analyzer.add_protocol_database(protocol, db_path)
                            protocol_dbs[protocol] = db_path
                        
                        log_configs.append({"path": log_path, "protocol": protocol})
                        
                    except KeyError:
                        print(f"⚠️  Unknown protocol: {protocol_str}")
        
        if log_configs:
            results = analyzer.analyze_mixed_logs(log_configs)
        else:
            print("❌ No valid log configurations found")
            return
    
    else:
        # Single protocol mode (backward compatible)
        dbc_file = sys.argv[1]
        log_file = sys.argv[2]
        
        if not Path(dbc_file).exists():
            print(f"❌ DBC file not found: {dbc_file}")
            return
        
        if not Path(log_file).exists():
            print(f"❌ Log file not found: {log_file}")
            return
        
        results = analyzer.analyze_single_protocol(dbc_file, log_file, ProtocolType.CAN)
    
    if results:
        # Export results
        exported = analyzer.export_results()
        
        print(f"\n💾 Results exported:")
        for file_type, filename in exported.items():
            print(f"   {file_type}: {filename}")
        
        print(f"\n🎉 Day 2 Analysis Complete!")
        print(f"✅ Enhanced multi-protocol analyzer working!")
        print(f"🚀 Ready for Day 3: Cross-Protocol Correlation!")
    else:
        print(f"\n❌ Analysis failed - check your files and try again")

if __name__ == "__main__":
    main()