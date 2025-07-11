#!/usr/bin/env python3
"""
Day 2: Signal Quality Assessment System
Detect signal health issues: missing, stuck, noisy, out-of-range
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class SignalIssueType(Enum):
    MISSING = "missing"
    STUCK = "stuck"  
    NOISY = "noisy"
    OUT_OF_RANGE = "out_of_range"
    LOW_FREQUENCY = "low_frequency"
    HIGH_FREQUENCY = "high_frequency"

@dataclass
class SignalQuality:
    """Signal quality assessment results"""
    signal_name: str
    message_name: str
    total_samples: int
    time_coverage: float  # Percentage of time signal was present
    update_rate: float   # Hz
    stuck_percentage: float
    noise_level: float
    range_violations: int
    quality_score: float  # 0-100
    issues: List[SignalIssueType]
    
class SignalQualityAnalyzer:
    """Analyze signal quality and detect automotive network issues"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_coverage': 80.0,      # % - Minimum time coverage
            'max_stuck': 15.0,         # % - Maximum stuck values
            'max_noise': 0.1,          # - Maximum noise ratio
            'min_update_rate': 0.1,    # Hz - Minimum update frequency
            'max_update_rate': 1000.0  # Hz - Maximum reasonable frequency
        }
    
    def analyze_signal_quality(self, df: pd.DataFrame) -> Dict[str, SignalQuality]:
        """Analyze quality for all signals in the dataset"""
        print("🔍 Analyzing signal quality...")
        
        if df.empty:
            return {}
        
        quality_results = {}
        
        # Group by signal for analysis
        for signal_name in df['signal'].unique():
            signal_data = df[df['signal'] == signal_name].copy()
            quality = self._analyze_single_signal(signal_data)
            quality_results[signal_name] = quality
        
        # Summary statistics
        total_signals = len(quality_results)
        good_signals = sum(1 for q in quality_results.values() if q.quality_score >= 80)
        poor_signals = sum(1 for q in quality_results.values() if q.quality_score < 50)
        
        print(f"📊 Quality Analysis Complete:")
        print(f"   Total signals: {total_signals}")
        print(f"   Good quality (≥80): {good_signals}")
        print(f"   Poor quality (<50): {poor_signals}")
        
        return quality_results
    
    def _analyze_single_signal(self, signal_data: pd.DataFrame) -> SignalQuality:
        """Analyze quality of a single signal"""
        signal_name = signal_data['signal'].iloc[0]
        message_name = signal_data['message'].iloc[0]
        values = signal_data['value'].values
        timestamps = signal_data['timestamp'].values
        
        # Basic metrics
        total_samples = len(values)
        time_span = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 0
        
        # Time coverage analysis
        time_coverage = self._calculate_time_coverage(timestamps, time_span)
        
        # Update rate analysis
        update_rate = self._calculate_update_rate(timestamps)
        
        # Stuck value analysis
        stuck_percentage = self._calculate_stuck_percentage(values)
        
        # Noise analysis
        noise_level = self._calculate_noise_level(values)
        
        # Range violation analysis (basic - needs signal metadata for proper ranges)
        range_violations = self._detect_range_violations(values)
        
        # Detect issues
        issues = self._detect_issues(time_coverage, update_rate, stuck_percentage, 
                                   noise_level, range_violations)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(time_coverage, update_rate, 
                                                    stuck_percentage, noise_level, 
                                                    range_violations)
        
        return SignalQuality(
            signal_name=signal_name,
            message_name=message_name,
            total_samples=total_samples,
            time_coverage=time_coverage,
            update_rate=update_rate,
            stuck_percentage=stuck_percentage,
            noise_level=noise_level,
            range_violations=range_violations,
            quality_score=quality_score,
            issues=issues
        )
    
    def _calculate_time_coverage(self, timestamps: np.ndarray, time_span: float) -> float:
        """Calculate what percentage of time the signal was present"""
        if time_span <= 0 or len(timestamps) < 2:
            return 100.0 if len(timestamps) > 0 else 0.0
        
        # Simple approach: assume signal should be present throughout time span
        # More sophisticated would use expected update rate
        expected_samples = time_span * 10  # Assume 10Hz baseline
        actual_samples = len(timestamps)
        coverage = min(100.0, (actual_samples / expected_samples) * 100)
        
        return coverage
    
    def _calculate_update_rate(self, timestamps: np.ndarray) -> float:
        """Calculate signal update rate in Hz"""
        if len(timestamps) < 2:
            return 0.0
        
        time_diffs = np.diff(timestamps)
        avg_period = np.mean(time_diffs)
        
        return 1.0 / avg_period if avg_period > 0 else 0.0
    
    def _calculate_stuck_percentage(self, values: np.ndarray) -> float:
        """Calculate percentage of consecutive identical values"""
        if len(values) < 2:
            return 0.0
        
        # Find consecutive identical values
        stuck_count = 0
        current_run = 1
        
        for i in range(1, len(values)):
            if values[i] == values[i-1]:
                current_run += 1
            else:
                if current_run >= 3:  # 3+ consecutive = stuck
                    stuck_count += current_run
                current_run = 1
        
        # Check final run
        if current_run >= 3:
            stuck_count += current_run
        
        return (stuck_count / len(values)) * 100
    
    def _calculate_noise_level(self, values: np.ndarray) -> float:
        """Calculate signal noise level"""
        if len(values) < 3:
            return 0.0
        
        # Use coefficient of variation as noise measure
        try:
            if np.std(values) == 0:
                return 0.0
            return np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0.0
        except:
            return 0.0
    
    def _detect_range_violations(self, values: np.ndarray) -> int:
        """Detect basic range violations (needs proper signal metadata)"""
        # Basic outlier detection using IQR method
        if len(values) < 4:
            return 0
        
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return 0
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        violations = np.sum((values < lower_bound) | (values > upper_bound))
        return int(violations)
    
    def _detect_issues(self, time_coverage: float, update_rate: float, 
                      stuck_percentage: float, noise_level: float, 
                      range_violations: int) -> List[SignalIssueType]:
        """Detect specific signal issues"""
        issues = []
        
        if time_coverage < self.quality_thresholds['min_coverage']:
            issues.append(SignalIssueType.MISSING)
        
        if stuck_percentage > self.quality_thresholds['max_stuck']:
            issues.append(SignalIssueType.STUCK)
        
        if noise_level > self.quality_thresholds['max_noise']:
            issues.append(SignalIssueType.NOISY)
        
        if range_violations > 0:
            issues.append(SignalIssueType.OUT_OF_RANGE)
        
        if update_rate < self.quality_thresholds['min_update_rate']:
            issues.append(SignalIssueType.LOW_FREQUENCY)
        
        if update_rate > self.quality_thresholds['max_update_rate']:
            issues.append(SignalIssueType.HIGH_FREQUENCY)
        
        return issues
    
    def _calculate_quality_score(self, time_coverage: float, update_rate: float,
                               stuck_percentage: float, noise_level: float,
                               range_violations: int) -> float:
        """Calculate overall quality score 0-100"""
        score = 100.0
        
        # Penalize missing data
        score -= max(0, (100 - time_coverage) * 0.5)
        
        # Penalize stuck values
        score -= stuck_percentage * 2
        
        # Penalize excessive noise
        score -= min(20, noise_level * 100)
        
        # Penalize range violations
        score -= min(10, range_violations)
        
        # Penalize poor update rates
        if update_rate < 0.1:
            score -= 20
        elif update_rate > 1000:
            score -= 10
        
        return max(0.0, score)
    
    def generate_quality_report(self, quality_results: Dict[str, SignalQuality]) -> str:
        """Generate human-readable quality report"""
        if not quality_results:
            return "No signals to analyze"
        
        report = []
        report.append("📊 SIGNAL QUALITY REPORT")
        report.append("=" * 40)
        
        # Summary statistics
        scores = [q.quality_score for q in quality_results.values()]
        avg_score = np.mean(scores)
        
        report.append(f"\n📈 Overall Statistics:")
        report.append(f"   Total signals: {len(quality_results)}")
        report.append(f"   Average quality: {avg_score:.1f}/100")
        report.append(f"   Excellent (90+): {sum(1 for s in scores if s >= 90)}")
        report.append(f"   Good (70-89): {sum(1 for s in scores if 70 <= s < 90)}")
        report.append(f"   Fair (50-69): {sum(1 for s in scores if 50 <= s < 70)}")
        report.append(f"   Poor (<50): {sum(1 for s in scores if s < 50)}")
        
        # Top issues
        all_issues = []
        for q in quality_results.values():
            all_issues.extend(q.issues)
        
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            report.append(f"\n🚨 Most Common Issues:")
            for issue, count in issue_counts.most_common(5):
                report.append(f"   {issue.value}: {count} signals")
        
        # Worst performers
        worst_signals = sorted(quality_results.values(), 
                             key=lambda x: x.quality_score)[:5]
        
        if worst_signals:
            report.append(f"\n⚠️  Signals Needing Attention:")
            for q in worst_signals:
                issues_str = ", ".join([i.value for i in q.issues])
                report.append(f"   {q.signal_name}: {q.quality_score:.1f}/100 ({issues_str})")
        
        return "\n".join(report)