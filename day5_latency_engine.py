#!/usr/bin/env python3
"""
day5_latency_engine.py
Day 5: Enhanced Latency Analysis Engine
Advanced statistical analysis, performance benchmarking, and multi-hop gateway analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
import time
from collections import defaultdict, deque
import warnings

# Import Day 3 components for extension
from day3_gateway_analyzer import GatewayLatencyAnalyzer, LatencyAnalysis, GatewayPath
from day3_correlation_engine import SignalCorrelation

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class LatencyCategory(Enum):
    SAFETY_CRITICAL = "safety_critical"      # <10ms
    POWERTRAIN = "powertrain"                # <20ms  
    CHASSIS = "chassis"                      # <50ms
    COMFORT = "comfort"                      # <100ms
    INFOTAINMENT = "infotainment"           # <200ms
    DIAGNOSTIC = "diagnostic"                # <500ms

class PerformanceGrade(Enum):
    EXCELLENT = "excellent"    # >90 score
    GOOD = "good"             # 70-90 score
    FAIR = "fair"             # 50-70 score
    POOR = "poor"             # 30-50 score
    CRITICAL = "critical"     # <30 score

class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable" 
    DEGRADING = "degrading"
    VOLATILE = "volatile"

@dataclass
class LatencyDistribution:
    """Statistical distribution analysis of latencies"""
    mean: float
    median: float
    mode: float
    std_dev: float
    variance: float
    skewness: float
    kurtosis: float
    
    # Percentiles
    p50: float  # median
    p90: float
    p95: float
    p99: float
    p99_9: float
    
    # Range
    min_latency: float
    max_latency: float
    range_span: float
    
    # Quality metrics
    coefficient_of_variation: float
    outlier_count: int
    outlier_percentage: float
    
    def get_quality_assessment(self) -> str:
        """Get human-readable quality assessment"""
        if self.coefficient_of_variation < 0.1:
            return "Very consistent latencies"
        elif self.coefficient_of_variation < 0.3:
            return "Reasonably consistent latencies"
        elif self.coefficient_of_variation < 0.5:
            return "Moderate latency variation"
        else:
            return "High latency variation - investigate"

@dataclass 
class PerformanceBenchmark:
    """Performance benchmark against industry standards"""
    category: LatencyCategory
    measured_latency: float
    benchmark_limit: float
    performance_margin: float  # positive = good, negative = violation
    performance_ratio: float   # measured/benchmark
    grade: PerformanceGrade
    meets_requirement: bool
    
    # Context
    message_description: str
    criticality_level: str
    improvement_needed: float  # ms to reach benchmark

@dataclass
class TrendAnalysis:
    """Time-series trend analysis of latency performance"""
    direction: TrendDirection
    slope: float  # ms per second (trend rate)
    correlation_coefficient: float
    trend_strength: str  # weak, moderate, strong
    
    # Statistical tests
    is_statistically_significant: bool
    confidence_level: float
    
    # Predictions
    predicted_latency_1min: float
    predicted_latency_5min: float
    trend_stability: float  # 0-100%

@dataclass
class MultiHopAnalysis:
    """Analysis of multi-hop gateway paths"""
    hop_count: int
    individual_hops: List[float]  # latency of each hop
    cumulative_latency: float
    hop_distribution: LatencyDistribution
    
    # Performance analysis
    bottleneck_hop: int  # index of slowest hop
    bottleneck_latency: float
    efficiency_score: float  # 0-100%
    
    # Optimization potential
    optimization_potential: float  # potential improvement in ms
    recommended_actions: List[str]

@dataclass
class EnhancedLatencyResults:
    """Complete enhanced latency analysis results"""
    basic_analysis: LatencyAnalysis  # From Day 3
    
    # Enhanced metrics
    distribution: LatencyDistribution
    benchmark: PerformanceBenchmark
    trend: TrendAnalysis
    multi_hop: Optional[MultiHopAnalysis]
    
    # Advanced scoring
    overall_score: float  # 0-100
    reliability_score: float  # 0-100
    efficiency_score: float  # 0-100
    
    # Recommendations
    priority_issues: List[str]
    optimization_recommendations: List[str]
    monitoring_recommendations: List[str]

class EnhancedLatencyEngine:
    """Day 5: Enhanced latency analysis with advanced statistics and benchmarking"""
    
    def __init__(self):
        # Automotive timing requirements (ms) - enhanced from Day 3
        self.timing_requirements = {
            LatencyCategory.SAFETY_CRITICAL: 10.0,   # Safety systems
            LatencyCategory.POWERTRAIN: 20.0,        # Engine/transmission
            LatencyCategory.CHASSIS: 50.0,           # Steering/braking  
            LatencyCategory.COMFORT: 100.0,          # HVAC, lighting
            LatencyCategory.INFOTAINMENT: 200.0,     # Entertainment
            LatencyCategory.DIAGNOSTIC: 500.0        # Diagnostics
        }
        
        # Performance scoring weights
        self.scoring_weights = {
            'latency_performance': 0.4,    # How well meets timing requirements
            'consistency': 0.3,            # Low variation
            'reliability': 0.2,            # Few outliers
            'trend': 0.1                   # Improving or stable trend
        }
        
        self.enhanced_results = []
        
    def analyze_enhanced_latencies(self, basic_analyses: List[LatencyAnalysis], 
                                 correlations: List[SignalCorrelation],
                                 df: pd.DataFrame) -> List[EnhancedLatencyResults]:
        """
        Perform enhanced latency analysis on basic results from Day 3
        """
        print("🔬 Enhanced Latency Analysis Engine - Day 5")
        print("-" * 50)
        
        self.enhanced_results = []
        
        if not basic_analyses:
            print("   No basic latency analyses to enhance")
            return []
        
        print(f"   Enhancing {len(basic_analyses)} latency analyses...")
        
        for i, basic_analysis in enumerate(basic_analyses):
            print(f"   Analyzing path {i+1}/{len(basic_analyses)}: "
                  f"{basic_analysis.gateway_path.source_signal} → "
                  f"{basic_analysis.gateway_path.destination_signal}")
            
            enhanced_result = self._analyze_single_enhanced_latency(
                basic_analysis, correlations, df
            )
            
            if enhanced_result:
                self.enhanced_results.append(enhanced_result)
        
        print(f"✅ Enhanced analysis complete: {len(self.enhanced_results)} paths analyzed")
        return self.enhanced_results
    
    def _analyze_single_enhanced_latency(self, basic_analysis: LatencyAnalysis,
                                       correlations: List[SignalCorrelation],
                                       df: pd.DataFrame) -> Optional[EnhancedLatencyResults]:
        """Perform enhanced analysis on a single latency path"""
        
        if not basic_analysis.measurements:
            return None
        
        latencies = [m.latency_ms for m in basic_analysis.measurements]
        timestamps = [m.timestamp for m in basic_analysis.measurements]
        
        # 1. Statistical Distribution Analysis
        distribution = self._calculate_latency_distribution(latencies)
        
        # 2. Performance Benchmarking
        benchmark = self._benchmark_performance(basic_analysis, latencies)
        
        # 3. Trend Analysis
        trend = self._analyze_trends(timestamps, latencies)
        
        # 4. Multi-hop Analysis (if applicable)
        multi_hop = self._analyze_multi_hop_path(basic_analysis, df)
        
        # 5. Calculate Enhanced Scores
        overall_score = self._calculate_overall_score(distribution, benchmark, trend)
        reliability_score = self._calculate_reliability_score(distribution, basic_analysis)
        efficiency_score = self._calculate_efficiency_score(basic_analysis, multi_hop)
        
        # 6. Generate Recommendations
        priority_issues = self._identify_priority_issues(distribution, benchmark, trend)
        optimization_recs = self._generate_optimization_recommendations(
            basic_analysis, distribution, benchmark, multi_hop
        )
        monitoring_recs = self._generate_monitoring_recommendations(trend, distribution)
        
        return EnhancedLatencyResults(
            basic_analysis=basic_analysis,
            distribution=distribution,
            benchmark=benchmark,
            trend=trend,
            multi_hop=multi_hop,
            overall_score=overall_score,
            reliability_score=reliability_score,
            efficiency_score=efficiency_score,
            priority_issues=priority_issues,
            optimization_recommendations=optimization_recs,
            monitoring_recommendations=monitoring_recs
        )
    
    def _calculate_latency_distribution(self, latencies: List[float]) -> LatencyDistribution:
        """Calculate comprehensive statistical distribution of latencies"""
        
        if len(latencies) < 3:
            # Minimal distribution for small datasets
            return LatencyDistribution(
                mean=np.mean(latencies),
                median=np.median(latencies),
                mode=latencies[0],
                std_dev=np.std(latencies) if len(latencies) > 1 else 0,
                variance=np.var(latencies) if len(latencies) > 1 else 0,
                skewness=0, kurtosis=0, p50=np.median(latencies),
                p90=np.percentile(latencies, 90), p95=np.percentile(latencies, 95),
                p99=np.percentile(latencies, 99), p99_9=np.percentile(latencies, 99.9),
                min_latency=min(latencies), max_latency=max(latencies),
                range_span=max(latencies) - min(latencies),
                coefficient_of_variation=0, outlier_count=0, outlier_percentage=0
            )
        
        arr = np.array(latencies)
        
        # Basic statistics
        mean_val = np.mean(arr)
        median_val = np.median(arr)
        std_val = np.std(arr)
        var_val = np.var(arr)
        
        # Mode calculation (most frequent value, binned)
        try:
            hist, bin_edges = np.histogram(arr, bins=min(20, len(arr)//2))
            mode_idx = np.argmax(hist)
            mode_val = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        except:
            mode_val = median_val
        
        # Advanced statistics
        try:
            from scipy import stats
            skewness = stats.skew(arr)
            kurtosis = stats.kurtosis(arr)
        except ImportError:
            # Fallback without scipy
            skewness = 0.0
            kurtosis = 0.0
        
        # Percentiles
        percentiles = {
            'p50': np.percentile(arr, 50),
            'p90': np.percentile(arr, 90),
            'p95': np.percentile(arr, 95),
            'p99': np.percentile(arr, 99),
            'p99_9': np.percentile(arr, 99.9)
        }
        
        # Range
        min_lat = np.min(arr)
        max_lat = np.max(arr)
        range_span = max_lat - min_lat
        
        # Quality metrics
        coeff_var = std_val / mean_val if mean_val > 0 else 0
        
        # Outlier detection using IQR method
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(arr)) * 100
        else:
            outlier_count = 0
            outlier_percentage = 0
        
        return LatencyDistribution(
            mean=mean_val,
            median=median_val,
            mode=mode_val,
            std_dev=std_val,
            variance=var_val,
            skewness=skewness,
            kurtosis=kurtosis,
            p50=percentiles['p50'],
            p90=percentiles['p90'],
            p95=percentiles['p95'],
            p99=percentiles['p99'],
            p99_9=percentiles['p99_9'],
            min_latency=min_lat,
            max_latency=max_lat,
            range_span=range_span,
            coefficient_of_variation=coeff_var,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage
        )
    
    def _benchmark_performance(self, basic_analysis: LatencyAnalysis,
                             latencies: List[float]) -> PerformanceBenchmark:
        """Benchmark performance against automotive timing requirements"""
        
        # Categorize the signal based on name/type
        signal_name = basic_analysis.gateway_path.source_signal.lower()
        
        # Automotive signal categorization
        if any(keyword in signal_name for keyword in ['brake', 'airbag', 'abs', 'esp', 'safety']):
            category = LatencyCategory.SAFETY_CRITICAL
        elif any(keyword in signal_name for keyword in ['rpm', 'throttle', 'engine', 'transmission', 'gear']):
            category = LatencyCategory.POWERTRAIN
        elif any(keyword in signal_name for keyword in ['steer', 'wheel', 'speed', 'suspension']):
            category = LatencyCategory.CHASSIS
        elif any(keyword in signal_name for keyword in ['hvac', 'light', 'window', 'door', 'seat']):
            category = LatencyCategory.COMFORT
        elif any(keyword in signal_name for keyword in ['radio', 'navigation', 'display', 'media']):
            category = LatencyCategory.INFOTAINMENT
        else:
            category = LatencyCategory.DIAGNOSTIC  # Default for unknown signals
        
        # Get benchmark limit
        benchmark_limit = self.timing_requirements[category]
        
        # Use P95 latency for benchmarking (industry standard)
        measured_latency = np.percentile(latencies, 95)
        
        # Calculate performance metrics
        performance_margin = benchmark_limit - measured_latency
        performance_ratio = measured_latency / benchmark_limit
        meets_requirement = measured_latency <= benchmark_limit
        
        # Grade assignment
        if performance_ratio <= 0.5:
            grade = PerformanceGrade.EXCELLENT
        elif performance_ratio <= 0.8:
            grade = PerformanceGrade.GOOD
        elif performance_ratio <= 1.0:
            grade = PerformanceGrade.FAIR
        elif performance_ratio <= 1.5:
            grade = PerformanceGrade.POOR
        else:
            grade = PerformanceGrade.CRITICAL
        
        # Improvement needed
        improvement_needed = max(0, measured_latency - benchmark_limit)
        
        return PerformanceBenchmark(
            category=category,
            measured_latency=measured_latency,
            benchmark_limit=benchmark_limit,
            performance_margin=performance_margin,
            performance_ratio=performance_ratio,
            grade=grade,
            meets_requirement=meets_requirement,
            message_description=f"{basic_analysis.gateway_path.source_signal} → {basic_analysis.gateway_path.destination_signal}",
            criticality_level=category.value,
            improvement_needed=improvement_needed
        )
    
    def _analyze_trends(self, timestamps: List[float], latencies: List[float]) -> TrendAnalysis:
        """Analyze time-series trends in latency performance"""
        
        if len(timestamps) < 5:
            # Not enough data for trend analysis
            return TrendAnalysis(
                direction=TrendDirection.STABLE,
                slope=0.0,
                correlation_coefficient=0.0,
                trend_strength="insufficient_data",
                is_statistically_significant=False,
                confidence_level=0.0,
                predicted_latency_1min=np.mean(latencies),
                predicted_latency_5min=np.mean(latencies),
                trend_stability=100.0
            )
        
        # Convert timestamps to relative time (seconds from start)
        time_array = np.array(timestamps) - timestamps[0]
        latency_array = np.array(latencies)
        
        # Linear regression for trend
        try:
            slope, intercept = np.polyfit(time_array, latency_array, 1)
            correlation_coeff = np.corrcoef(time_array, latency_array)[0, 1]
        except:
            slope = 0.0
            intercept = np.mean(latencies)
            correlation_coeff = 0.0
        
        # Determine trend direction
        if abs(slope) < 0.1:  # Less than 0.1 ms per second
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.DEGRADING  # Latency increasing
        else:
            direction = TrendDirection.IMPROVING  # Latency decreasing
        
        # Check for volatility
        if np.std(latencies) / np.mean(latencies) > 0.5:  # High coefficient of variation
            direction = TrendDirection.VOLATILE
        
        # Trend strength
        abs_correlation = abs(correlation_coeff)
        if abs_correlation > 0.7:
            trend_strength = "strong"
        elif abs_correlation > 0.3:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"
        
        # Statistical significance (simplified)
        is_significant = abs_correlation > 0.5 and len(timestamps) > 10
        confidence_level = min(95.0, abs_correlation * 100)
        
        # Predictions (simple linear extrapolation)
        current_time = time_array[-1]
        predicted_1min = slope * (current_time + 60) + intercept
        predicted_5min = slope * (current_time + 300) + intercept
        
        # Trend stability (inverse of volatility)
        stability = max(0, 100 - (np.std(latencies) / np.mean(latencies) * 100))
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            correlation_coefficient=correlation_coeff,
            trend_strength=trend_strength,
            is_statistically_significant=is_significant,
            confidence_level=confidence_level,
            predicted_latency_1min=max(0, predicted_1min),
            predicted_latency_5min=max(0, predicted_5min),
            trend_stability=min(100, stability)
        )
    
    def _analyze_multi_hop_path(self, basic_analysis: LatencyAnalysis,
                              df: pd.DataFrame) -> Optional[MultiHopAnalysis]:
        """Analyze multi-hop gateway paths for optimization opportunities"""
        
        # For now, simulate multi-hop analysis
        # In a real implementation, this would trace signal paths through multiple gateways
        
        gateway_path = basic_analysis.gateway_path
        
        # Check if this could be a multi-hop path
        if gateway_path.gateway_type.value in ['MULTI_HOP'] or 'gateway' in gateway_path.source_signal.lower():
            
            # Simulate hop analysis
            total_latency = basic_analysis.mean_latency
            
            # Estimate individual hops (simplified simulation)
            if total_latency > 50:  # Likely multi-hop if > 50ms
                hop_count = 3
                # Simulate individual hop latencies
                hop1 = total_latency * 0.3  # First gateway
                hop2 = total_latency * 0.5  # Network transmission
                hop3 = total_latency * 0.2  # Final gateway
                individual_hops = [hop1, hop2, hop3]
            elif total_latency > 20:  # Possible 2-hop
                hop_count = 2
                hop1 = total_latency * 0.4
                hop2 = total_latency * 0.6
                individual_hops = [hop1, hop2]
            else:
                return None  # Single hop
            
            # Find bottleneck
            bottleneck_hop = individual_hops.index(max(individual_hops))
            bottleneck_latency = max(individual_hops)
            
            # Calculate efficiency score
            ideal_latency = total_latency * 0.5  # 50% efficiency target
            efficiency_score = min(100, (ideal_latency / total_latency) * 100)
            
            # Optimization potential
            optimization_potential = bottleneck_latency * 0.3  # 30% improvement possible
            
            # Recommendations
            recommendations = []
            if bottleneck_hop == 0:
                recommendations.append("Optimize source gateway processing")
            elif bottleneck_hop == len(individual_hops) - 1:
                recommendations.append("Optimize destination gateway processing")
            else:
                recommendations.append("Optimize network transmission path")
            
            if efficiency_score < 50:
                recommendations.append("Consider direct routing bypass")
            
            # Create distribution for hops
            hop_distribution = self._calculate_latency_distribution(individual_hops)
            
            return MultiHopAnalysis(
                hop_count=hop_count,
                individual_hops=individual_hops,
                cumulative_latency=total_latency,
                hop_distribution=hop_distribution,
                bottleneck_hop=bottleneck_hop,
                bottleneck_latency=bottleneck_latency,
                efficiency_score=efficiency_score,
                optimization_potential=optimization_potential,
                recommended_actions=recommendations
            )
        
        return None
    
    def _calculate_overall_score(self, distribution: LatencyDistribution,
                               benchmark: PerformanceBenchmark,
                               trend: TrendAnalysis) -> float:
        """Calculate overall latency performance score (0-100)"""
        
        # Component scores
        latency_score = self._get_latency_performance_score(benchmark)
        consistency_score = self._get_consistency_score(distribution)
        reliability_score = self._get_reliability_component_score(distribution)
        trend_score = self._get_trend_score(trend)
        
        # Weighted average
        overall = (
            latency_score * self.scoring_weights['latency_performance'] +
            consistency_score * self.scoring_weights['consistency'] +
            reliability_score * self.scoring_weights['reliability'] +
            trend_score * self.scoring_weights['trend']
        )
        
        return min(100.0, max(0.0, overall))
    
    def _get_latency_performance_score(self, benchmark: PerformanceBenchmark) -> float:
        """Score based on how well latency meets timing requirements"""
        if benchmark.performance_ratio <= 0.5:
            return 100.0
        elif benchmark.performance_ratio <= 0.8:
            return 90.0
        elif benchmark.performance_ratio <= 1.0:
            return 70.0
        elif benchmark.performance_ratio <= 1.5:
            return 40.0
        else:
            return max(0.0, 40.0 - (benchmark.performance_ratio - 1.5) * 20)
    
    def _get_consistency_score(self, distribution: LatencyDistribution) -> float:
        """Score based on latency consistency (low variation)"""
        cv = distribution.coefficient_of_variation
        
        if cv <= 0.1:
            return 100.0
        elif cv <= 0.2:
            return 90.0
        elif cv <= 0.3:
            return 75.0
        elif cv <= 0.5:
            return 50.0
        else:
            return max(0.0, 50.0 - (cv - 0.5) * 100)
    
    def _get_reliability_component_score(self, distribution: LatencyDistribution) -> float:
        """Score based on reliability (few outliers)"""
        outlier_pct = distribution.outlier_percentage
        
        if outlier_pct <= 1.0:
            return 100.0
        elif outlier_pct <= 5.0:
            return 80.0
        elif outlier_pct <= 10.0:
            return 60.0
        else:
            return max(0.0, 60.0 - (outlier_pct - 10.0) * 3)
    
    def _get_trend_score(self, trend: TrendAnalysis) -> float:
        """Score based on trend direction and stability"""
        base_score = 70.0  # Neutral for stable trends
        
        if trend.direction == TrendDirection.IMPROVING:
            base_score = 90.0
        elif trend.direction == TrendDirection.DEGRADING:
            base_score = 40.0
        elif trend.direction == TrendDirection.VOLATILE:
            base_score = 30.0
        
        # Adjust for trend stability
        stability_factor = trend.trend_stability / 100.0
        return base_score * stability_factor
    
    def _calculate_reliability_score(self, distribution: LatencyDistribution,
                                   basic_analysis: LatencyAnalysis) -> float:
        """Calculate reliability score based on measurement quality and consistency"""
        
        # Base reliability from successful measurements
        measurement_reliability = (basic_analysis.valid_measurements / 
                                 basic_analysis.total_measurements * 100 
                                 if basic_analysis.total_measurements > 0 else 0)
        
        # Consistency reliability (inverse of coefficient of variation)
        consistency_reliability = max(0, 100 - distribution.coefficient_of_variation * 200)
        
        # Outlier reliability (inverse of outlier percentage)
        outlier_reliability = max(0, 100 - distribution.outlier_percentage * 2)
        
        # Combined reliability score
        return (measurement_reliability * 0.4 + 
                consistency_reliability * 0.4 + 
                outlier_reliability * 0.2)
    
    def _calculate_efficiency_score(self, basic_analysis: LatencyAnalysis,
                                  multi_hop: Optional[MultiHopAnalysis]) -> float:
        """Calculate efficiency score based on latency performance"""
        
        if multi_hop:
            return multi_hop.efficiency_score
        
        # Single-hop efficiency based on basic performance score
        return basic_analysis.performance_score
    
    def _identify_priority_issues(self, distribution: LatencyDistribution,
                                benchmark: PerformanceBenchmark,
                                trend: TrendAnalysis) -> List[str]:
        """Identify high-priority issues requiring immediate attention"""
        
        issues = []
        
        # Critical timing violations
        if not benchmark.meets_requirement:
            severity = "CRITICAL" if benchmark.performance_ratio > 2.0 else "HIGH"
            issues.append(f"{severity}: Timing requirement violation "
                         f"({benchmark.measured_latency:.1f}ms > {benchmark.benchmark_limit:.1f}ms)")
        
        # High variability
        if distribution.coefficient_of_variation > 0.5:
            issues.append("HIGH: Excessive latency variation - system instability risk")
        
        # Degrading trends
        if trend.direction == TrendDirection.DEGRADING and trend.is_statistically_significant:
            issues.append("MEDIUM: Degrading latency trend detected - monitor closely")
        
        # High outlier rate
        if distribution.outlier_percentage > 10:
            issues.append("MEDIUM: High outlier rate - investigate root causes")
        
        # Volatile performance
        if trend.direction == TrendDirection.VOLATILE:
            issues.append("MEDIUM: Volatile latency pattern - system stability concern")
        
        return issues
    
    def _generate_optimization_recommendations(self, basic_analysis: LatencyAnalysis,
                                             distribution: LatencyDistribution,
                                             benchmark: PerformanceBenchmark,
                                             multi_hop: Optional[MultiHopAnalysis]) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Performance optimizations
        if benchmark.improvement_needed > 0:
            recommendations.append(f"Reduce latency by {benchmark.improvement_needed:.1f}ms "
                                 f"to meet {benchmark.category.value} requirements")
        
        # Consistency optimizations
        if distribution.coefficient_of_variation > 0.3:
            recommendations.append("Implement latency buffering to reduce variation")
            recommendations.append("Review gateway processing load balancing")
        
        # Multi-hop optimizations
        if multi_hop and multi_hop.optimization_potential > 5:
            recommendations.extend(multi_hop.recommended_actions)
            recommendations.append(f"Potential {multi_hop.optimization_potential:.1f}ms improvement available")
        
        # Protocol-specific optimizations
        gateway_type = basic_analysis.gateway_path.gateway_type.value
        if "CAN" in gateway_type and distribution.mean > 20:
            recommendations.append("Consider CAN bus optimization: check bus load and message priorities")
        elif "LIN" in gateway_type and distribution.mean > 50:
            recommendations.append("Consider LIN scheduling optimization")
        elif "ETHERNET" in gateway_type and distribution.mean > 10:
            recommendations.append("Consider Ethernet QoS configuration")
        
        return recommendations
    
    def _generate_monitoring_recommendations(self, trend: TrendAnalysis,
                                           distribution: LatencyDistribution) -> List[str]:
        """Generate monitoring and alerting recommendations"""
        
        recommendations = []
        
        # Trend-based monitoring
        if trend.direction == TrendDirection.DEGRADING:
            recommendations.append("Set up trend monitoring with 5% degradation alert threshold")
        
        if trend.trend_stability < 70:
            recommendations.append("Implement real-time latency monitoring due to instability")
        
        # Distribution-based monitoring
        if distribution.coefficient_of_variation > 0.3:
            recommendations.append("Monitor latency variance with upper control limit alerts")
        
        if distribution.outlier_percentage > 5:
            recommendations.append("Set up outlier detection with automatic root cause logging")
        
        # Performance-based monitoring
        if distribution.p99 > distribution.mean * 3:
            recommendations.append("Monitor P99 latency spikes - investigate worst-case scenarios")
        
        # Proactive monitoring
        recommendations.append("Implement dashboard with real-time latency metrics")
        recommendations.append("Set up automated performance regression detection")
        
        return recommendations
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system latency performance summary"""
        
        if not self.enhanced_results:
            return {"error": "No enhanced analysis results available"}
        
        # Aggregate metrics across all paths
        overall_scores = [r.overall_score for r in self.enhanced_results]
        reliability_scores = [r.reliability_score for r in self.enhanced_results]
        efficiency_scores = [r.efficiency_score for r in self.enhanced_results]
        
        # Benchmark compliance
        compliant_paths = sum(1 for r in self.enhanced_results if r.benchmark.meets_requirement)
        total_paths = len(self.enhanced_results)
        compliance_rate = (compliant_paths / total_paths * 100) if total_paths > 0 else 0
        
        # Grade distribution
        grade_counts = {}
        for r in self.enhanced_results:
            grade = r.benchmark.grade.value
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        # Critical issues
        critical_issues = 0
        high_issues = 0
        for r in self.enhanced_results:
            for issue in r.priority_issues:
                if issue.startswith("CRITICAL"):
                    critical_issues += 1
                elif issue.startswith("HIGH"):
                    high_issues += 1
        
        # Trending analysis
        improving_paths = sum(1 for r in self.enhanced_results 
                            if r.trend.direction == TrendDirection.IMPROVING)
        degrading_paths = sum(1 for r in self.enhanced_results 
                            if r.trend.direction == TrendDirection.DEGRADING)
        
        return {
            "summary": {
                "total_paths_analyzed": total_paths,
                "average_overall_score": np.mean(overall_scores),
                "average_reliability_score": np.mean(reliability_scores),
                "average_efficiency_score": np.mean(efficiency_scores),
                "compliance_rate_percent": compliance_rate
            },
            "performance_grades": grade_counts,
            "issues": {
                "critical_count": critical_issues,
                "high_priority_count": high_issues,
                "paths_needing_attention": sum(1 for r in self.enhanced_results if r.overall_score < 70)
            },
            "trends": {
                "improving_paths": improving_paths,
                "degrading_paths": degrading_paths,
                "stable_paths": total_paths - improving_paths - degrading_paths
            },
            "recommendations": {
                "immediate_action_required": critical_issues > 0,
                "monitoring_setup_needed": high_issues > 0,
                "optimization_potential": sum(1 for r in self.enhanced_results if r.overall_score < 85)
            }
        }
    
    def generate_enhanced_report(self) -> str:
        """Generate comprehensive enhanced latency analysis report"""
        
        if not self.enhanced_results:
            return "No enhanced latency analysis results available"
        
        report = []
        report.append("🔬 DAY 5 - ENHANCED LATENCY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Executive Summary
        summary = self.get_system_performance_summary()
        
        report.append(f"\n📊 EXECUTIVE SUMMARY")
        report.append(f"   Paths Analyzed: {summary['summary']['total_paths_analyzed']}")
        report.append(f"   Average Performance Score: {summary['summary']['average_overall_score']:.1f}/100")
        report.append(f"   Timing Compliance Rate: {summary['summary']['compliance_rate_percent']:.1f}%")
        report.append(f"   System Reliability: {summary['summary']['average_reliability_score']:.1f}/100")
        report.append(f"   System Efficiency: {summary['summary']['average_efficiency_score']:.1f}/100")
        
        # Critical Issues
        if summary['issues']['critical_count'] > 0:
            report.append(f"\n🚨 CRITICAL ISSUES")
            report.append(f"   Critical Issues: {summary['issues']['critical_count']}")
            report.append(f"   High Priority Issues: {summary['issues']['high_priority_count']}")
            report.append(f"   Paths Needing Attention: {summary['issues']['paths_needing_attention']}")
        
        # Performance Grade Distribution
        report.append(f"\n🏆 PERFORMANCE GRADES")
        for grade, count in summary['performance_grades'].items():
            percentage = (count / summary['summary']['total_paths_analyzed']) * 100
            report.append(f"   {grade.upper()}: {count} paths ({percentage:.1f}%)")
        
        # Trend Analysis
        report.append(f"\n📈 TREND ANALYSIS")
        report.append(f"   Improving: {summary['trends']['improving_paths']} paths")
        report.append(f"   Stable: {summary['trends']['stable_paths']} paths")
        report.append(f"   Degrading: {summary['trends']['degrading_paths']} paths")
        
        # Detailed Path Analysis (Top 5 and Bottom 5)
        sorted_results = sorted(self.enhanced_results, key=lambda x: x.overall_score, reverse=True)
        
        report.append(f"\n🥇 TOP PERFORMING PATHS")
        for i, result in enumerate(sorted_results[:5]):
            path = result.basic_analysis.gateway_path
            report.append(f"   {i+1}. {path.source_signal} → {path.destination_signal}")
            report.append(f"      Score: {result.overall_score:.1f}/100, "
                         f"Latency: {result.distribution.p95:.1f}ms, "
                         f"Grade: {result.benchmark.grade.value}")
        
        report.append(f"\n⚠️  PATHS NEEDING ATTENTION")
        for i, result in enumerate(sorted_results[-5:]):
            path = result.basic_analysis.gateway_path
            report.append(f"   {i+1}. {path.source_signal} → {path.destination_signal}")
            report.append(f"      Score: {result.overall_score:.1f}/100, "
                         f"Latency: {result.distribution.p95:.1f}ms, "
                         f"Grade: {result.benchmark.grade.value}")
            
            # Show top priority issue
            if result.priority_issues:
                report.append(f"      Issue: {result.priority_issues[0]}")
        
        # Statistical Insights
        all_latencies = []
        for result in self.enhanced_results:
            all_latencies.extend([m.latency_ms for m in result.basic_analysis.measurements])
        
        if all_latencies:
            report.append(f"\n📊 STATISTICAL INSIGHTS")
            report.append(f"   Total Measurements: {len(all_latencies):,}")
            report.append(f"   System-wide P50 Latency: {np.percentile(all_latencies, 50):.1f}ms")
            report.append(f"   System-wide P95 Latency: {np.percentile(all_latencies, 95):.1f}ms")
            report.append(f"   System-wide P99 Latency: {np.percentile(all_latencies, 99):.1f}ms")
            report.append(f"   Latency Range: {min(all_latencies):.1f} - {max(all_latencies):.1f}ms")
        
        # Multi-hop Analysis
        multi_hop_results = [r for r in self.enhanced_results if r.multi_hop]
        if multi_hop_results:
            report.append(f"\n🔗 MULTI-HOP GATEWAY ANALYSIS")
            report.append(f"   Multi-hop Paths: {len(multi_hop_results)}")
            
            avg_efficiency = np.mean([r.multi_hop.efficiency_score for r in multi_hop_results])
            total_optimization = sum(r.multi_hop.optimization_potential for r in multi_hop_results)
            
            report.append(f"   Average Efficiency: {avg_efficiency:.1f}%")
            report.append(f"   Total Optimization Potential: {total_optimization:.1f}ms")
        
        # Recommendations Summary
        report.append(f"\n🎯 KEY RECOMMENDATIONS")
        
        # Collect all unique recommendations
        all_optimizations = set()
        all_monitoring = set()
        
        for result in self.enhanced_results:
            all_optimizations.update(result.optimization_recommendations)
            all_monitoring.update(result.monitoring_recommendations)
        
        if all_optimizations:
            report.append(f"   📈 Optimization Actions:")
            for i, rec in enumerate(list(all_optimizations)[:5], 1):
                report.append(f"      {i}. {rec}")
        
        if all_monitoring:
            report.append(f"   📊 Monitoring Setup:")
            for i, rec in enumerate(list(all_monitoring)[:3], 1):
                report.append(f"      {i}. {rec}")
        
        # Performance Targets
        report.append(f"\n🎯 PERFORMANCE TARGETS")
        report.append(f"   Target: >90% paths with scores >80")
        report.append(f"   Current: {sum(1 for r in self.enhanced_results if r.overall_score > 80) / len(self.enhanced_results) * 100:.1f}%")
        report.append(f"   Target: 100% timing compliance")
        report.append(f"   Current: {summary['summary']['compliance_rate_percent']:.1f}%")
        
        return "\n".join(report)
    
    def export_enhanced_results(self, output_prefix: str = "day5_enhanced_latency") -> Dict[str, str]:
        """Export enhanced latency analysis results"""
        
        if not self.enhanced_results:
            return {}
        
        exported_files = {}
        timestamp = int(time.time())
        
        # Export detailed results to CSV
        detailed_data = []
        for result in self.enhanced_results:
            path = result.basic_analysis.gateway_path
            detailed_data.append({
                'source_signal': path.source_signal,
                'destination_signal': path.destination_signal,
                'gateway_type': path.gateway_type.value,
                'overall_score': result.overall_score,
                'reliability_score': result.reliability_score,
                'efficiency_score': result.efficiency_score,
                'mean_latency_ms': result.distribution.mean,
                'p95_latency_ms': result.distribution.p95,
                'p99_latency_ms': result.distribution.p99,
                'std_dev_ms': result.distribution.std_dev,
                'coefficient_of_variation': result.distribution.coefficient_of_variation,
                'outlier_percentage': result.distribution.outlier_percentage,
                'benchmark_category': result.benchmark.category.value,
                'benchmark_limit_ms': result.benchmark.benchmark_limit,
                'performance_grade': result.benchmark.grade.value,
                'meets_requirement': result.benchmark.meets_requirement,
                'trend_direction': result.trend.direction.value,
                'trend_slope': result.trend.slope,
                'trend_stability': result.trend.trend_stability,
                'multi_hop': result.multi_hop is not None,
                'priority_issues_count': len(result.priority_issues),
                'optimization_recs_count': len(result.optimization_recommendations)
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = f"{output_prefix}_detailed_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
        exported_files['detailed_analysis'] = detailed_file
        
        # Export system summary
        summary = self.get_system_performance_summary()
        summary_file = f"{output_prefix}_summary_{timestamp}.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        exported_files['system_summary'] = summary_file
        
        # Export comprehensive report
        report = self.generate_enhanced_report()
        report_file = f"{output_prefix}_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        exported_files['comprehensive_report'] = report_file
        
        # Export recommendations
        recommendations_data = []
        for result in self.enhanced_results:
            path = result.basic_analysis.gateway_path
            
            for issue in result.priority_issues:
                recommendations_data.append({
                    'signal_path': f"{path.source_signal} → {path.destination_signal}",
                    'type': 'Priority Issue',
                    'description': issue,
                    'score_impact': result.overall_score
                })
            
            for rec in result.optimization_recommendations:
                recommendations_data.append({
                    'signal_path': f"{path.source_signal} → {path.destination_signal}",
                    'type': 'Optimization',
                    'description': rec,
                    'score_impact': result.overall_score
                })
        
        if recommendations_data:
            recommendations_df = pd.DataFrame(recommendations_data)
            recommendations_file = f"{output_prefix}_recommendations_{timestamp}.csv"
            recommendations_df.to_csv(recommendations_file, index=False)
            exported_files['recommendations'] = recommendations_file
        
        return exported_files

# Integration function for Day 4 FastAPI backend
def integrate_enhanced_latency_analysis(basic_analyses: List[LatencyAnalysis],
                                      correlations: List[SignalCorrelation],
                                      df: pd.DataFrame) -> Dict[str, Any]:
    """
    Integration function for Day 4 FastAPI backend
    Performs enhanced latency analysis and returns results
    """
    
    engine = EnhancedLatencyEngine()
    enhanced_results = engine.analyze_enhanced_latencies(basic_analyses, correlations, df)
    
    if not enhanced_results:
        return {"error": "No enhanced latency analysis results"}
    
    # Generate summary for API response
    summary = engine.get_system_performance_summary()
    
    # Export files
    exported_files = engine.export_enhanced_results()
    
    return {
        "enhanced_results_count": len(enhanced_results),
        "system_summary": summary,
        "exported_files": exported_files,
        "overall_performance_score": summary['summary']['average_overall_score'],
        "compliance_rate": summary['summary']['compliance_rate_percent'],
        "critical_issues": summary['issues']['critical_count'],
        "report_preview": engine.generate_enhanced_report()[:500] + "...",
        "day": 5,
        "feature": "Enhanced Latency Analysis Engine"
    }