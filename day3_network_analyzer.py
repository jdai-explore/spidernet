#!/usr/bin/env python3
"""
day3_network_analyzer.py
Complete Day 3 Multi-Protocol Network Analyzer
Combines Day 1-2 foundation with cross-protocol correlation and gateway analysis
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Import Day 2 components
from day2_enhanced_analyzer import EnhancedNetworkAnalyzer
from day2_universal_signal import ProtocolType

# Import Day 3 components
from day3_correlation_engine import CrossProtocolCorrelator
from day3_gateway_analyzer import GatewayLatencyAnalyzer

class Day3NetworkAnalyzer:
    """Day 3 - Complete multi-protocol network analyzer with correlation analysis"""
    
    def __init__(self):
        # Initialize components
        self.day2_analyzer = EnhancedNetworkAnalyzer()
        self.correlator = CrossProtocolCorrelator()
        self.gateway_analyzer = GatewayLatencyAnalyzer()
        
        # Results storage
        self.analysis_results = {}
        self.correlations = []
        self.latency_analyses = []
        
    def analyze_complete_network(self, log_configs: List[Dict[str, Any]], 
                               analysis_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete network analysis with correlation and latency analysis
        
        Args:
            log_configs: List of {"path": "file", "protocol": ProtocolType, "database": "db_file"}
            analysis_config: Optional configuration for analysis parameters
        """
        print("🚀 Day 3 - Complete Network Analysis")
        print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Load protocol databases
        self._load_protocol_databases(log_configs)
        
        # Step 2: Basic multi-protocol analysis (Day 2)
        print(f"\n📊 Step 1: Multi-Protocol Signal Analysis")
        basic_results = self.day2_analyzer.analyze_mixed_logs(log_configs)
        
        if not basic_results or basic_results['total_signals'] == 0:
            print("❌ No signals found - cannot proceed with correlation analysis")
            return {}
        
        df = basic_results['data_frame']
        
        # Step 3: Cross-protocol correlation analysis
        print(f"\n🔗 Step 2: Cross-Protocol Correlation Analysis")
        self.correlations = self.correlator.find_correlations(df)
        
        # Step 4: Gateway latency analysis
        print(f"\n⏱️  Step 3: Gateway Latency Analysis")
        self.latency_analyses = self.gateway_analyzer.analyze_gateway_latencies(self.correlations, df)
        
        # Step 5: Compile complete results
        end_time = time.time()
        
        complete_results = {
            'analysis_metadata': {
                'analysis_time': end_time - start_time,
                'analysis_timestamp': time.time(),
                'day3_version': '1.0',
                'protocols_analyzed': basic_results['protocols']
            },
            
            # Day 2 results
            'basic_analysis': basic_results,
            
            # Day 3 correlation results  
            'correlation_analysis': {
                'total_correlations': len(self.correlations),
                'gateway_candidates': len([c for c in self.correlations if c.gateway_candidate]),
                'correlations': self.correlations,
                'correlation_report': self.correlator.generate_correlation_report()
            },
            
            # Day 3 latency results
            'latency_analysis': {
                'total_gateway_paths': len(self.latency_analyses),
                'latency_analyses': self.latency_analyses,
                'overall_performance': self.gateway_analyzer.calculate_overall_gateway_performance(),
                'latency_report': self.gateway_analyzer.generate_latency_report()
            }
        }
        
        self.analysis_results = complete_results
        self._print_complete_analysis_summary(complete_results)
        
        return complete_results
    
    def _load_protocol_databases(self, log_configs: List[Dict[str, Any]]):
        """Load all required protocol databases"""
        print(f"📚 Loading Protocol Databases")
        
        loaded_protocols = set()
        
        for config in log_configs:
            protocol = config.get('protocol', ProtocolType.CAN)
            database = config.get('database')
            
            if not database:
                print(f"⚠️  No database specified for {protocol.value}")
                continue
            
            if protocol in loaded_protocols:
                continue  # Already loaded
            
            success = self.day2_analyzer.add_protocol_database(protocol, database)
            if success:
                loaded_protocols.add(protocol)
            else:
                print(f"❌ Failed to load {protocol.value} database: {database}")
    
    def _print_complete_analysis_summary(self, results: Dict[str, Any]):
        """Print comprehensive Day 3 analysis summary"""
        print(f"\n✅ COMPLETE NETWORK ANALYSIS FINISHED!")
        
        # Basic metrics
        basic = results['basic_analysis']
        print(f"   ⏱️  Total time: {results['analysis_metadata']['analysis_time']:.2f} seconds")
        print(f"   📊 Signals processed: {basic['total_signals']:,}")
        print(f"   🌐 Protocols: {', '.join(basic['protocols'])}")
        
        # Correlation metrics
        corr = results['correlation_analysis']
        print(f"   🔗 Correlations found: {corr['total_correlations']}")
        print(f"   🚪 Gateway candidates: {corr['gateway_candidates']}")
        
        # Latency metrics
        latency = results['latency_analysis']
        if latency['total_gateway_paths'] > 0:
            overall_perf = latency['overall_performance']
            print(f"   ⏱️  Gateway paths: {latency['total_gateway_paths']}")
            print(f"   📈 Avg latency: {overall_perf.get('overall_mean_latency', 0):.1f}ms")
            print(f"   🎯 Performance: {overall_perf.get('average_performance_score', 0):.1f}/100")
        else:
            print(f"   ⏱️  No gateway latencies measured")
    
    def analyze_single_protocol_with_correlation(self, database_path: str, log_path: str, 
                                               protocol: ProtocolType = ProtocolType.CAN) -> Dict[str, Any]:
        """Analyze single protocol with correlation analysis (for testing)"""
        log_config = [{
            "path": log_path,
            "protocol": protocol,
            "database": database_path
        }]
        
        return self.analyze_complete_network(log_config)
    
    def export_complete_results(self, output_prefix: str = "day3_complete_analysis") -> Dict[str, str]:
        """Export all Day 3 analysis results"""
        if not self.analysis_results:
            return {}
        
        exported_files = {}
        timestamp = int(time.time())
        
        # Export basic Day 2 results
        day2_exports = self.day2_analyzer.export_results(f"{output_prefix}_basic")
        exported_files.update(day2_exports)
        
        # Export correlation results
        if self.correlations:
            corr_data = []
            for corr in self.correlations:
                corr_data.append({
                    'signal1': corr.signal1_name,
                    'protocol1': corr.signal1_protocol,
                    'message1': corr.signal1_message,
                    'signal2': corr.signal2_name,
                    'protocol2': corr.signal2_protocol,
                    'message2': corr.signal2_message,
                    'correlation_type': corr.correlation_type.value,
                    'correlation_coefficient': corr.correlation_coefficient,
                    'confidence_score': corr.confidence_score,
                    'delay_ms': corr.delay_ms,
                    'gateway_candidate': corr.gateway_candidate,
                    'description': corr.description
                })
            
            corr_df = pd.DataFrame(corr_data)
            corr_file = f"{output_prefix}_correlations_{timestamp}.csv"
            corr_df.to_csv(corr_file, index=False)
            exported_files['correlations'] = corr_file
        
        # Export latency results
        if self.latency_analyses:
            latency_data = []
            for analysis in self.latency_analyses:
                latency_data.append({
                    'source_signal': analysis.gateway_path.source_signal,
                    'source_protocol': analysis.gateway_path.source_protocol,
                    'dest_signal': analysis.gateway_path.destination_signal,
                    'dest_protocol': analysis.gateway_path.destination_protocol,
                    'gateway_type': analysis.gateway_path.gateway_type.value,
                    'measurements': analysis.total_measurements,
                    'mean_latency_ms': analysis.mean_latency,
                    'median_latency_ms': analysis.median_latency,
                    'p95_latency_ms': analysis.p95_latency,
                    'p99_latency_ms': analysis.p99_latency,
                    'jitter_ms': analysis.jitter,
                    'performance_score': analysis.performance_score,
                    'meets_requirements': analysis.meets_requirements,
                    'recommendations': '; '.join(analysis.recommendations)
                })
            
            latency_df = pd.DataFrame(latency_data)
            latency_file = f"{output_prefix}_latencies_{timestamp}.csv"
            latency_df.to_csv(latency_file, index=False)
            exported_files['latencies'] = latency_file
        
        # Export comprehensive report
        report = self.generate_executive_report()
        report_file = f"{output_prefix}_executive_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        exported_files['executive_report'] = report_file
        
        # Export configuration for reproducibility
        config = {
            'analysis_metadata': self.analysis_results.get('analysis_metadata', {}),
            'protocols_used': self.analysis_results['basic_analysis']['protocols'],
            'total_signals': self.analysis_results['basic_analysis']['total_signals'],
            'correlations_found': len(self.correlations),
            'gateway_paths': len(self.latency_analyses)
        }
        
        config_file = f"{output_prefix}_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        exported_files['config'] = config_file
        
        return exported_files
    
    def generate_executive_report(self) -> str:
        """Generate executive summary report for Day 3 analysis"""
        if not self.analysis_results:
            return "No analysis results available"
        
        report = []
        report.append("🚀 DAY 3 - EXECUTIVE ANALYSIS REPORT")
        report.append("COMPLETE MULTI-PROTOCOL NETWORK ANALYSIS")
        report.append("=" * 60)
        
        # Executive Summary
        basic = self.analysis_results['basic_analysis']
        corr = self.analysis_results['correlation_analysis']
        latency = self.analysis_results['latency_analysis']
        meta = self.analysis_results['analysis_metadata']
        
        report.append(f"\n📊 EXECUTIVE SUMMARY")
        report.append(f"   Analysis completed in {meta['analysis_time']:.1f} seconds")
        report.append(f"   Protocols analyzed: {', '.join(meta['protocols_analyzed'])}")
        report.append(f"   Total signal samples: {basic['total_signals']:,}")
        report.append(f"   Cross-protocol correlations: {corr['total_correlations']}")
        report.append(f"   Gateway paths identified: {corr['gateway_candidates']}")
        report.append(f"   End-to-end latencies measured: {latency['total_gateway_paths']}")
        
        # Key Findings
        report.append(f"\n🔍 KEY FINDINGS")
        
        if corr['total_correlations'] > 0:
            report.append(f"   ✅ Found {corr['total_correlations']} signal correlations across protocols")
            report.append(f"   ✅ Identified {corr['gateway_candidates']} potential gateway signal pairs")
        else:
            report.append(f"   ⚠️  No cross-protocol correlations detected")
        
        if latency['total_gateway_paths'] > 0:
            overall_perf = latency['overall_performance']
            avg_latency = overall_perf.get('overall_mean_latency', 0)
            avg_performance = overall_perf.get('average_performance_score', 0)
            
            report.append(f"   📊 Average gateway latency: {avg_latency:.1f}ms")
            report.append(f"   📈 Average performance score: {avg_performance:.1f}/100")
            
            if avg_performance >= 80:
                report.append(f"   ✅ Gateway performance is excellent")
            elif avg_performance >= 60:
                report.append(f"   ⚠️  Gateway performance needs monitoring")
            else:
                report.append(f"   ❌ Gateway performance requires immediate attention")
        
        # Add detailed reports
        if corr['total_correlations'] > 0:
            report.append(f"\n{corr['correlation_report']}")
        
        if latency['total_gateway_paths'] > 0:
            report.append(f"\n{latency['latency_report']}")
        
        # Business Impact
        report.append(f"\n💼 BUSINESS IMPACT")
        report.append(f"   🎯 Network visibility: Complete multi-protocol coverage achieved")
        report.append(f"   🔗 Integration analysis: Cross-protocol dependencies identified") 
        report.append(f"   ⏱️  Performance monitoring: Real-time latency measurement capability")
        report.append(f"   🚪 Gateway optimization: Specific improvement opportunities identified")
        
        # Recommendations
        report.append(f"\n🎯 RECOMMENDATIONS")
        
        if corr['gateway_candidates'] > 0:
            report.append(f"   1. Monitor {corr['gateway_candidates']} gateway signal pairs for reliability")
        
        if latency['total_gateway_paths'] > 0:
            worst_performers = self.gateway_analyzer.get_worst_performing_gateways(1)
            if worst_performers and worst_performers[0].performance_score < 70:
                report.append(f"   2. Investigate gateway performance issues in {worst_performers[0].gateway_path.gateway_type.value}")
        
        report.append(f"   3. Implement continuous monitoring for identified correlations")
        report.append(f"   4. Use this analysis as baseline for network optimization")
        
        return "\n".join(report)

def main():
    """Day 3 command line interface"""
    if len(sys.argv) < 2:
        print("🚀 Day 3 - Complete Multi-Protocol Network Analyzer")
        print("=" * 55)
        print("\nUsage Options:")
        print("1. Single Protocol with Correlation:")
        print("   python day3_network_analyzer.py single <database> <logfile> [protocol]")
        print("\n2. Multi-Protocol Complete Analysis:")
        print("   python day3_network_analyzer.py multi <config_file>")
        print("\nExamples:")
        print("   python day3_network_analyzer.py single vehicle.dbc trace.asc CAN")
        print("   python day3_network_analyzer.py multi network_config.txt")
        print("\nConfig file format (one per line):")
        print("   protocol,database_file,log_file")
        print("   CAN,vehicle.dbc,can_trace.asc")
        print("   LIN,lin_setup.ldf,lin_trace.txt")
        print("   ETHERNET,someip_config.arxml,ethernet_trace.pcap")
        return
    
    analyzer = Day3NetworkAnalyzer()
    mode = sys.argv[1].lower()
    
    if mode == "single":
        # Single protocol mode
        if len(sys.argv) < 4:
            print("❌ Single mode requires: database_file log_file [protocol]")
            return
        
        database_file = sys.argv[2]
        log_file = sys.argv[3]
        protocol_str = sys.argv[4] if len(sys.argv) > 4 else "CAN"
        
        try:
            protocol = ProtocolType[protocol_str.upper()]
        except KeyError:
            print(f"❌ Unknown protocol: {protocol_str}")
            return
        
        # Validate files
        if not Path(database_file).exists():
            print(f"❌ Database file not found: {database_file}")
            return
        
        if not Path(log_file).exists():
            print(f"❌ Log file not found: {log_file}")
            return
        
        # Run analysis
        results = analyzer.analyze_single_protocol_with_correlation(database_file, log_file, protocol)
        
    elif mode == "multi":
        # Multi-protocol mode
        if len(sys.argv) < 3:
            print("❌ Multi mode requires: config_file")
            return
        
        config_file = sys.argv[2]
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            return
        
        # Parse config file
        log_configs = []
        
        with open(config_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) < 3:
                    print(f"⚠️  Line {line_num}: Invalid format (need protocol,database,logfile)")
                    continue
                
                try:
                    protocol = ProtocolType[parts[0].strip().upper()]
                    database = parts[1].strip()
                    logfile = parts[2].strip()
                    
                    log_configs.append({
                        "protocol": protocol,
                        "database": database,
                        "path": logfile
                    })
                    
                except KeyError:
                    print(f"⚠️  Line {line_num}: Unknown protocol {parts[0]}")
                    continue
        
        if not log_configs:
            print("❌ No valid configurations found")
            return
        
        # Run analysis
        results = analyzer.analyze_complete_network(log_configs)
        
    else:
        print(f"❌ Unknown mode: {mode}")
        return
    
    # Export results if analysis successful
    if results:
        exported = analyzer.export_complete_results()
        
        print(f"\n💾 Complete Analysis Results Exported:")
        for file_type, filename in exported.items():
            print(f"   {file_type}: {filename}")
        
        print(f"\n🎉 DAY 3 COMPLETE!")
        print(f"✅ Cross-protocol correlation analysis working!")
        print(f"✅ Gateway latency measurement working!")
        print(f"✅ Complete network visibility achieved!")
        print(f"\n🚀 Ready for Day 4: FastAPI Backend!")
    else:
        print(f"\n❌ Analysis failed - check your files and configuration")

if __name__ == "__main__":
    main()