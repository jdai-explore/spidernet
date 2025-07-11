#!/usr/bin/env python3
"""
day3_gateway_analyzer.py
Gateway Latency Analysis System
Analyze end-to-end latencies and gateway performance in automotive networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import defaultdict

class GatewayType(Enum):
    CAN_TO_LIN = "CAN_TO_LIN"
    CAN_TO_ETHERNET = "CAN_TO_ETHERNET"
    LIN_TO_CAN = "LIN_TO_CAN"
    LIN_TO_ETHERNET = "LIN_TO_ETHERNET"
    ETHERNET_TO_CAN = "ETHERNET_TO_CAN"
    ETHERNET_TO_LIN = "ETHERNET_TO_LIN"
    MULTI_HOP = "MULTI_HOP"

@dataclass
class GatewayPath:
    """Represents a signal path through gateway(s)"""
    source_signal: str
    source_protocol: str
    source_message: str
    
    destination_signal: str
    destination_protocol: str
    destination_message: str
    
    gateway_type: GatewayType
    hop_count: int = 1
    
    # Gateway nodes (if known)
    gateway_nodes: List[str] = None
    
    def __post_init__(self):
        if self.gateway_nodes is None:
            self.gateway_nodes = []

@dataclass
class LatencyMeasurement:
    """Single latency measurement between correlated signals"""
    timestamp: float
    source_timestamp: float
    destination_timestamp: float
    latency_ms: float
    
    source_value: Any
    destination_value: Any
    
    # Quality indicators
    measurement_quality: float  # 0-100
    confidence: float          # 0-100

@dataclass
class LatencyAnalysis:
    """Complete latency analysis for a gateway path"""
    gateway_path: GatewayPath
    
    # Statistical metrics
    total_measurements: int
    valid_measurements: int
    
    # Latency statistics (all in milliseconds)
    min_latency: float
    max_latency: float
    mean_latency: float
    median_latency: float
    std_latency: float
    p95_latency: float
    p99_latency: float
    
    # Performance assessment
    performance_score: float   # 0-100 (higher = better)
    meets_requirements: bool   # Based on automotive timing requirements
    
    # Quality metrics
    jitter: float             # Latency variation
    packet_loss: float        # Percentage of missed correlations
    
    # Raw measurements for detailed analysis
    measurements: List[LatencyMeasurement] = None
    
    # Analysis summary
    summary: str = ""
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.measurements is None:
            self.measurements = []
        if self.recommendations is None:
            self.recommendations = []

class GatewayLatencyAnalyzer:
    """Analyze gateway latencies and end-to-end performance"""
    
    def __init__(self):
        # Automotive timing requirements (ms)
        self.timing_requirements = {
            'safety_critical': 10.0,    # Safety systems
            'powertrain': 20.0,         # Engine/transmission
            'chassis': 50.0,            # Steering/braking
            'comfort': 100.0,           # HVAC, lighting
            'infotainment': 200.0       # Entertainment, navigation
        }
        
        self.latency_analyses = []
        
    def analyze_gateway_latencies(self, correlations: List, df: pd.DataFrame) -> List[LatencyAnalysis]:
        """Analyze latencies for all gateway correlations"""
        print("⏱️  Analyzing gateway latencies...")
        
        self.latency_analyses = []
        
        # Filter for gateway candidates
        gateway_correlations = [corr for corr in correlations if corr.gateway_candidate]
        
        if not gateway_correlations:
            print("   No gateway correlations found")
            return []
        
        print(f"   Analyzing {len(gateway_correlations)} gateway paths")
        
        for correlation in gateway_correlations:
            analysis = self._analyze_single_gateway_path(correlation, df)
            if analysis:
                self.latency_analyses.append(analysis)
        
        print(f"✅ Completed latency analysis for {len(self.latency_analyses)} paths")
        return self.latency_analyses
    
    def _analyze_single_gateway_path(self, correlation, df: pd.DataFrame) -> Optional[LatencyAnalysis]:
        """Analyze latency for a single gateway path"""
        
        # Create gateway path
        gateway_path = self._create_gateway_path(correlation)
        
        # Extract signal data
        source_data = df[
            (df['signal'] == correlation.signal1_name) & 
            (df['protocol'] == correlation.signal1_protocol)
        ].copy()
        
        dest_data = df[
            (df['signal'] == correlation.signal2_name) & 
            (df['protocol'] == correlation.signal2_protocol)
        ].copy()
        
        if len(source_data) < 5 or len(dest_data) < 5:
            return None
        
        # Find latency measurements
        measurements = self._calculate_latency_measurements(source_data, dest_data)
        
        if len(measurements) < 3:
            return None
        
        # Calculate statistics
        latencies = [m.latency_ms for m in measurements]
        
        analysis = LatencyAnalysis(
            gateway_path=gateway_path,
            total_measurements=len(measurements),
            valid_measurements=len([m for m in measurements if m.measurement_quality > 70]),
            min_latency=min(latencies),
            max_latency=max(latencies),
            mean_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            p95_latency=np.percentile(latencies, 95),
            p99_latency=np.percentile(latencies, 99),
            measurements=measurements
        )
        
        # Calculate performance metrics
        self._calculate_performance_metrics(analysis)
        
        # Generate recommendations
        self._generate_recommendations(analysis)
        
        return analysis
    
    def _create_gateway_path(self, correlation) -> GatewayPath:
        """Create gateway path from correlation"""
        
        # Determine gateway type
        proto1 = correlation.signal1_protocol
        proto2 = correlation.signal2_protocol
        
        gateway_type_map = {
            ('CAN', 'LIN'): GatewayType.CAN_TO_LIN,
            ('LIN', 'CAN'): GatewayType.LIN_TO_CAN,
            ('CAN', 'ETHERNET'): GatewayType.CAN_TO_ETHERNET,
            ('ETHERNET', 'CAN'): GatewayType.ETHERNET_TO_CAN,
            ('LIN', 'ETHERNET'): GatewayType.LIN_TO_ETHERNET,
            ('ETHERNET', 'LIN'): GatewayType.ETHERNET_TO_LIN
        }
        
        gateway_type = gateway_type_map.get((proto1, proto2), GatewayType.MULTI_HOP)
        
        return GatewayPath(
            source_signal=correlation.signal1_name,
            source_protocol=proto1,
            source_message=correlation.signal1_message,
            destination_signal=correlation.signal2_name,
            destination_protocol=proto2,
            destination_message=correlation.signal2_message,
            gateway_type=gateway_type
        )
    
    def _calculate_latency_measurements(self, source_data: pd.DataFrame, 
                                      dest_data: pd.DataFrame) -> List[LatencyMeasurement]:
        """Calculate individual latency measurements"""
        measurements = []
        
        # Sort by timestamp
        source_data = source_data.sort_values('timestamp')
        dest_data = dest_data.sort_values('timestamp')
        
        # For each source signal, find corresponding destination signal
        for _, source_row in source_data.iterrows():
            source_time = source_row['timestamp']
            source_value = source_row['value']
            
            # Find destination signals within reasonable time window (1 second)
            time_window = 1.0  # seconds
            
            dest_candidates = dest_data[
                (dest_data['timestamp'] >= source_time) &
                (dest_data['timestamp'] <= source_time + time_window)
            ]
            
            if len(dest_candidates) == 0:
                continue
            
            # Find best matching destination signal
            # For identical correlation, look for same value
            # For other correlations, use closest in time
            
            if len(dest_candidates) == 1:
                dest_row = dest_candidates.iloc[0]
            else:
                # Multiple candidates - choose best match
                dest_row = self._find_best_destination_match(source_row, dest_candidates)
            
            if dest_row is not None:
                latency_ms = (dest_row['timestamp'] - source_time) * 1000
                
                # Quality assessment
                quality = self._assess_measurement_quality(source_row, dest_row, latency_ms)
                
                measurement = LatencyMeasurement(
                    timestamp=source_time,
                    source_timestamp=source_time,
                    destination_timestamp=dest_row['timestamp'],
                    latency_ms=latency_ms,
                    source_value=source_value,
                    destination_value=dest_row['value'],
                    measurement_quality=quality,
                    confidence=quality
                )
                
                measurements.append(measurement)
        
        return measurements
    
    def _find_best_destination_match(self, source_row, dest_candidates: pd.DataFrame):
        """Find best matching destination signal for latency measurement"""
        
        # Strategy 1: Exact value match (for gateway relays)
        try:
            source_val = float(source_row['value'])
            exact_matches = dest_candidates[dest_candidates['value'] == source_val]
            
            if len(exact_matches) > 0:
                # Return earliest exact match
                return exact_matches.iloc[0]
        except:
            pass
        
        # Strategy 2: Closest in time
        time_diffs = abs(dest_candidates['timestamp'] - source_row['timestamp'])
        closest_idx = time_diffs.idxmin()
        return dest_candidates.loc[closest_idx]
    
    def _assess_measurement_quality(self, source_row, dest_row, latency_ms: float) -> float:
        """Assess quality of a latency measurement"""
        quality = 100.0
        
        # Penalize very high latencies (likely false matches)
        if latency_ms > 500:  # 500ms
            quality -= 50
        elif latency_ms > 200:  # 200ms
            quality -= 20
        
        # Penalize negative latencies (timing errors)
        if latency_ms < 0:
            quality = 0
        
        # Bonus for value similarity (gateway relays)
        try:
            source_val = float(source_row['value'])
            dest_val = float(dest_row['value'])
            
            if abs(source_val - dest_val) < 0.01:  # Very similar values
                quality += 10
            elif abs(source_val - dest_val) < abs(source_val) * 0.1:  # Within 10%
                quality += 5
        except:
            pass
        
        return min(100.0, max(0.0, quality))
    
    def _calculate_performance_metrics(self, analysis: LatencyAnalysis):
        """Calculate performance metrics for latency analysis"""
        
        # Calculate jitter (latency variation)
        analysis.jitter = analysis.std_latency
        
        # Calculate packet loss (estimated from missing correlations)
        total_possible = analysis.total_measurements
        valid_measurements = analysis.valid_measurements
        analysis.packet_loss = ((total_possible - valid_measurements) / total_possible) * 100 if total_possible > 0 else 0
        
        # Calculate performance score
        score = 100.0
        
        # Penalize high latency
        if analysis.mean_latency > 100:  # > 100ms
            score -= 30
        elif analysis.mean_latency > 50:  # > 50ms
            score -= 15
        elif analysis.mean_latency > 20:  # > 20ms
            score -= 5
        
        # Penalize high jitter
        if analysis.jitter > 50:  # > 50ms jitter
            score -= 20
        elif analysis.jitter > 20:  # > 20ms jitter
            score -= 10
        
        # Penalize packet loss
        score -= analysis.packet_loss * 0.5  # 0.5 points per percent lost
        
        analysis.performance_score = max(0.0, score)
        
        # Determine if meets requirements (assume comfort level by default)
        analysis.meets_requirements = analysis.p95_latency <= self.timing_requirements['comfort']
        
        # Generate summary
        analysis.summary = f"Gateway latency: {analysis.mean_latency:.1f}ms ± {analysis.jitter:.1f}ms, " \
                          f"Performance: {analysis.performance_score:.1f}/100"
    
    def _generate_recommendations(self, analysis: LatencyAnalysis):
        """Generate performance recommendations"""
        analysis.recommendations = []
        
        if analysis.mean_latency > 100:
            analysis.recommendations.append("High latency detected - investigate gateway processing time")
        
        if analysis.jitter > 30:
            analysis.recommendations.append("High jitter detected - check for network congestion")
        
        if analysis.packet_loss > 5:
            analysis.recommendations.append("Packet loss detected - verify signal correlation accuracy")
        
        if analysis.p99_latency > analysis.mean_latency * 3:
            analysis.recommendations.append("Latency spikes detected - investigate worst-case scenarios")
        
        if not analysis.meets_requirements:
            analysis.recommendations.append("Does not meet automotive timing requirements")
        
        if not analysis.recommendations:
            analysis.recommendations.append("Gateway performance is acceptable")
    
    def get_worst_performing_gateways(self, count: int = 5) -> List[LatencyAnalysis]:
        """Get worst performing gateway paths"""
        return sorted(self.latency_analyses, key=lambda x: x.performance_score)[:count]
    
    def get_best_performing_gateways(self, count: int = 5) -> List[LatencyAnalysis]:
        """Get best performing gateway paths"""
        return sorted(self.latency_analyses, key=lambda x: x.performance_score, reverse=True)[:count]
    
    def get_gateways_by_type(self, gateway_type: GatewayType) -> List[LatencyAnalysis]:
        """Get gateway analyses by type"""
        return [analysis for analysis in self.latency_analyses 
                if analysis.gateway_path.gateway_type == gateway_type]
    
    def calculate_overall_gateway_performance(self) -> Dict[str, Any]:
        """Calculate overall gateway system performance"""
        if not self.latency_analyses:
            return {}
        
        all_latencies = []
        all_performance_scores = []
        
        for analysis in self.latency_analyses:
            all_latencies.extend([m.latency_ms for m in analysis.measurements])
            all_performance_scores.append(analysis.performance_score)
        
        return {
            'total_gateway_paths': len(self.latency_analyses),
            'total_measurements': sum(len(a.measurements) for a in self.latency_analyses),
            'overall_mean_latency': statistics.mean(all_latencies) if all_latencies else 0,
            'overall_p95_latency': np.percentile(all_latencies, 95) if all_latencies else 0,
            'overall_p99_latency': np.percentile(all_latencies, 99) if all_latencies else 0,
            'average_performance_score': statistics.mean(all_performance_scores),
            'paths_meeting_requirements': sum(1 for a in self.latency_analyses if a.meets_requirements),
            'gateway_types': list(set(a.gateway_path.gateway_type.value for a in self.latency_analyses))
        }
    
    def generate_latency_report(self) -> str:
        """Generate comprehensive latency analysis report"""
        if not self.latency_analyses:
            return "No gateway latency data available"
        
        report = []
        report.append("⏱️  GATEWAY LATENCY ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Overall performance
        overall = self.calculate_overall_gateway_performance()
        
        report.append(f"\n📊 Overall Performance:")
        report.append(f"   Gateway paths analyzed: {overall['total_gateway_paths']}")
        report.append(f"   Total measurements: {overall['total_measurements']:,}")
        report.append(f"   Average latency: {overall['overall_mean_latency']:.1f}ms")
        report.append(f"   95th percentile: {overall['overall_p95_latency']:.1f}ms")
        report.append(f"   99th percentile: {overall['overall_p99_latency']:.1f}ms")
        report.append(f"   Average performance: {overall['average_performance_score']:.1f}/100")
        report.append(f"   Paths meeting requirements: {overall['paths_meeting_requirements']}/{overall['total_gateway_paths']}")
        
        # By gateway type
        report.append(f"\n🌐 By Gateway Type:")
        gateway_types = {}
        for analysis in self.latency_analyses:
            gt = analysis.gateway_path.gateway_type.value
            if gt not in gateway_types:
                gateway_types[gt] = []
            gateway_types[gt].append(analysis)
        
        for gtype, analyses in gateway_types.items():
            avg_latency = statistics.mean([a.mean_latency for a in analyses])
            avg_performance = statistics.mean([a.performance_score for a in analyses])
            report.append(f"   {gtype}: {len(analyses)} paths, {avg_latency:.1f}ms avg, {avg_performance:.1f}/100")
        
        # Best performers
        best = self.get_best_performing_gateways(3)
        if best:
            report.append(f"\n🏆 Best Performing Paths:")
            for i, analysis in enumerate(best):
                path = analysis.gateway_path
                report.append(f"   {i+1}. {path.source_signal} → {path.destination_signal}")
                report.append(f"      {path.source_protocol} to {path.destination_protocol}: "
                             f"{analysis.mean_latency:.1f}ms, Score: {analysis.performance_score:.1f}")
        
        # Worst performers
        worst = self.get_worst_performing_gateways(3)
        if worst:
            report.append(f"\n⚠️  Paths Needing Attention:")
            for i, analysis in enumerate(worst):
                path = analysis.gateway_path
                report.append(f"   {i+1}. {path.source_signal} → {path.destination_signal}")
                report.append(f"      {path.source_protocol} to {path.destination_protocol}: "
                             f"{analysis.mean_latency:.1f}ms, Score: {analysis.performance_score:.1f}")
                if analysis.recommendations:
                    report.append(f"      Recommendations: {'; '.join(analysis.recommendations[:2])}")
        
        # Timing requirements check
        report.append(f"\n🎯 Timing Requirements Assessment:")
        for req_name, req_time in self.timing_requirements.items():
            meeting_count = sum(1 for a in self.latency_analyses if a.p95_latency <= req_time)
            report.append(f"   {req_name.title()} (<{req_time}ms): {meeting_count}/{len(self.latency_analyses)} paths")
        
        return "\n".join(report)