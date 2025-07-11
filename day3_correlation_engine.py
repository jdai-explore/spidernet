#!/usr/bin/env python3
"""
day3_correlation_engine.py
Cross-Protocol Signal Correlation Engine
Find signals that move together across different protocols (CAN, LIN, Ethernet)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import itertools
from scipy import stats
from collections import defaultdict
import warnings

# Suppress scipy constant input warnings for cleaner output
warnings.filterwarnings('ignore', message='An input array is constant*')

class CorrelationType(Enum):
    IDENTICAL = "identical"           # Exact same values (gateway relay)
    LINEAR = "linear"                # Linear relationship
    DELAYED = "delayed"              # Same signal with time delay
    INVERSE = "inverse"              # Inverted relationship
    CONDITIONAL = "conditional"      # Relationship under certain conditions
    DERIVED = "derived"              # One signal derived from another

@dataclass
class SignalCorrelation:
    """Represents correlation between two signals"""
    signal1_name: str
    signal1_protocol: str
    signal1_message: str
    
    signal2_name: str
    signal2_protocol: str  
    signal2_message: str
    
    correlation_type: CorrelationType
    correlation_coefficient: float   # -1 to 1
    confidence_score: float         # 0 to 100
    delay_ms: float                # Time delay in milliseconds
    
    # Statistical metrics
    sample_count: int
    time_overlap: float            # Percentage of time both signals active
    
    # Analysis details
    description: str
    gateway_candidate: bool        # Likely gateway signal pair
    
    # Raw data for further analysis
    signal1_timestamps: List[float] = None
    signal1_values: List[float] = None
    signal2_timestamps: List[float] = None
    signal2_values: List[float] = None

class CrossProtocolCorrelator:
    """Engine for finding correlations between signals across protocols"""
    
    def __init__(self):
        self.correlation_thresholds = {
            'min_correlation': 0.7,        # Minimum correlation coefficient
            'min_confidence': 80.0,        # Minimum confidence score
            'max_delay_ms': 500.0,         # Maximum acceptable delay
            'min_samples': 10,             # Minimum overlapping samples
            'min_time_overlap': 50.0       # Minimum time overlap percentage
        }
        
        self.correlations = []
        
    def find_correlations(self, df: pd.DataFrame) -> List[SignalCorrelation]:
        """Find all signal correlations in the dataset"""
        print("🔍 Searching for cross-protocol correlations...")
        
        if df.empty:
            return []
        
        self.correlations = []
        
        # Group signals by protocol
        protocols = df['protocol'].unique()
        
        print(f"   Analyzing {len(protocols)} protocols: {', '.join(protocols)}")
        
        # Find correlations within and across protocols
        for proto1, proto2 in itertools.combinations_with_replacement(protocols, 2):
            if proto1 == proto2:
                continue  # Skip same-protocol for now (focus on cross-protocol)
            
            self._find_protocol_pair_correlations(df, proto1, proto2)
        
        # Sort by confidence score
        self.correlations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        print(f"✅ Found {len(self.correlations)} significant correlations")
        return self.correlations
    
    def _find_protocol_pair_correlations(self, df: pd.DataFrame, proto1: str, proto2: str):
        """Find correlations between two specific protocols"""
        print(f"   Analyzing {proto1} ↔ {proto2}")
        
        # Get signals from each protocol
        proto1_signals = df[df['protocol'] == proto1]['signal'].unique()
        proto2_signals = df[df['protocol'] == proto2]['signal'].unique()
        
        correlations_found = 0
        
        # Test all signal pairs
        for sig1 in proto1_signals:
            for sig2 in proto2_signals:
                correlation = self._analyze_signal_pair(df, sig1, proto1, sig2, proto2)
                
                if correlation and correlation.confidence_score >= self.correlation_thresholds['min_confidence']:
                    self.correlations.append(correlation)
                    correlations_found += 1
        
        print(f"     → {correlations_found} correlations found")
    
    def _analyze_signal_pair(self, df: pd.DataFrame, sig1: str, proto1: str, 
                           sig2: str, proto2: str) -> Optional[SignalCorrelation]:
        """Analyze correlation between two specific signals"""
        
        # Get signal data
        sig1_data = df[(df['signal'] == sig1) & (df['protocol'] == proto1)].copy()
        sig2_data = df[(df['signal'] == sig2) & (df['protocol'] == proto2)].copy()
        
        if len(sig1_data) < self.correlation_thresholds['min_samples'] or \
           len(sig2_data) < self.correlation_thresholds['min_samples']:
            return None
        
        # Check for numeric values only
        try:
            sig1_data['value'] = pd.to_numeric(sig1_data['value'], errors='coerce')
            sig2_data['value'] = pd.to_numeric(sig2_data['value'], errors='coerce')
            
            sig1_data = sig1_data.dropna(subset=['value'])
            sig2_data = sig2_data.dropna(subset=['value'])
            
            if len(sig1_data) < 5 or len(sig2_data) < 5:
                return None
                
        except:
            return None
        
        # Find time overlap and align signals
        aligned_data = self._align_signals_in_time(sig1_data, sig2_data)
        
        if not aligned_data or len(aligned_data) < self.correlation_thresholds['min_samples']:
            return None
        
        # Calculate correlation metrics
        values1 = [d['value1'] for d in aligned_data]
        values2 = [d['value2'] for d in aligned_data]
        timestamps = [d['timestamp'] for d in aligned_data]
        
        # Test different correlation types
        best_correlation = self._find_best_correlation_type(
            values1, values2, timestamps, sig1, proto1, sig2, proto2, sig1_data, sig2_data
        )
        
        return best_correlation
    
    def _align_signals_in_time(self, sig1_data: pd.DataFrame, sig2_data: pd.DataFrame, 
                             max_delay_s: float = 0.5) -> List[Dict]:
        """Align two signals in time, accounting for possible delays"""
        
        # Simple nearest-neighbor alignment for now
        # More sophisticated would use interpolation
        
        aligned = []
        
        for _, row1 in sig1_data.iterrows():
            t1 = row1['timestamp']
            
            # Find closest signal2 sample within delay window
            time_diffs = np.abs(sig2_data['timestamp'] - t1)
            min_diff_idx = time_diffs.idxmin()
            min_diff = time_diffs.loc[min_diff_idx]
            
            if min_diff <= max_delay_s:
                row2 = sig2_data.loc[min_diff_idx]
                
                aligned.append({
                    'timestamp': t1,
                    'value1': row1['value'],
                    'value2': row2['value'],
                    'delay': row2['timestamp'] - t1
                })
        
        return aligned
    
    def _find_best_correlation_type(self, values1: List[float], values2: List[float], 
                                  timestamps: List[float], sig1: str, proto1: str,
                                  sig2: str, proto2: str, sig1_data: pd.DataFrame,
                                  sig2_data: pd.DataFrame) -> Optional[SignalCorrelation]:
        """Test different correlation types and return the best match"""
        
        if len(values1) != len(values2) or len(values1) < 3:
            return None
        
        best_correlation = None
        best_score = 0
        
        # Test 1: Identical correlation (gateway relay)
        identical_corr = self._test_identical_correlation(
            values1, values2, timestamps, sig1, proto1, sig2, proto2, sig1_data, sig2_data
        )
        if identical_corr and identical_corr.confidence_score > best_score:
            best_correlation = identical_corr
            best_score = identical_corr.confidence_score
        
        # Test 2: Linear correlation
        linear_corr = self._test_linear_correlation(
            values1, values2, timestamps, sig1, proto1, sig2, proto2, sig1_data, sig2_data
        )
        if linear_corr and linear_corr.confidence_score > best_score:
            best_correlation = linear_corr
            best_score = linear_corr.confidence_score
        
        # Test 3: Delayed correlation
        delayed_corr = self._test_delayed_correlation(
            values1, values2, timestamps, sig1, proto1, sig2, proto2, sig1_data, sig2_data
        )
        if delayed_corr and delayed_corr.confidence_score > best_score:
            best_correlation = delayed_corr
            best_score = delayed_corr.confidence_score
        
        # Test 4: Inverse correlation
        inverse_corr = self._test_inverse_correlation(
            values1, values2, timestamps, sig1, proto1, sig2, proto2, sig1_data, sig2_data
        )
        if inverse_corr and inverse_corr.confidence_score > best_score:
            best_correlation = inverse_corr
            best_score = inverse_corr.confidence_score
        
        return best_correlation
    
    def _test_identical_correlation(self, values1: List[float], values2: List[float],
                                  timestamps: List[float], sig1: str, proto1: str,
                                  sig2: str, proto2: str, sig1_data: pd.DataFrame,
                                  sig2_data: pd.DataFrame) -> Optional[SignalCorrelation]:
        """Test for identical values (perfect gateway relay)"""
        
        # Check if values are nearly identical
        differences = np.abs(np.array(values1) - np.array(values2))
        max_diff = max(np.max(np.array(values1)) * 0.01, 0.1)  # 1% tolerance or 0.1 minimum
        
        identical_count = np.sum(differences <= max_diff)
        identical_percentage = (identical_count / len(values1)) * 100
        
        if identical_percentage >= 70:  # Lower threshold to 70% for more detection
            # Calculate average delay
            delays = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
            avg_delay = np.mean(delays) * 1000  # Convert to ms
            
            # Mark as gateway candidate if cross-protocol and high similarity
            is_gateway = (proto1 != proto2) and identical_percentage >= 80
            
            return SignalCorrelation(
                signal1_name=sig1,
                signal1_protocol=proto1,
                signal1_message=sig1_data['message'].iloc[0],
                signal2_name=sig2,
                signal2_protocol=proto2,
                signal2_message=sig2_data['message'].iloc[0],
                correlation_type=CorrelationType.IDENTICAL,
                correlation_coefficient=1.0,
                confidence_score=identical_percentage,
                delay_ms=avg_delay,
                sample_count=len(values1),
                time_overlap=100.0,
                description=f"Identical values ({identical_percentage:.1f}% match) - {'gateway relay' if is_gateway else 'same signal'}",
                gateway_candidate=is_gateway
            )
        
        return None
    
    def _test_linear_correlation(self, values1: List[float], values2: List[float],
                               timestamps: List[float], sig1: str, proto1: str,
                               sig2: str, proto2: str, sig1_data: pd.DataFrame,
                               sig2_data: pd.DataFrame) -> Optional[SignalCorrelation]:
        """Test for linear correlation"""
        
        try:
            correlation_coeff, p_value = stats.pearsonr(values1, values2)
            
            if abs(correlation_coeff) >= self.correlation_thresholds['min_correlation']:
                # Calculate confidence based on correlation strength and p-value
                confidence = min(100, abs(correlation_coeff) * 100 * (1 - p_value))
                
                if confidence >= self.correlation_thresholds['min_confidence']:
                    # Mark as gateway candidate if cross-protocol and very high correlation
                    is_gateway = (proto1 != proto2) and abs(correlation_coeff) > 0.9
                    
                    return SignalCorrelation(
                        signal1_name=sig1,
                        signal1_protocol=proto1,
                        signal1_message=sig1_data['message'].iloc[0],
                        signal2_name=sig2,
                        signal2_protocol=proto2,
                        signal2_message=sig2_data['message'].iloc[0],
                        correlation_type=CorrelationType.LINEAR,
                        correlation_coefficient=correlation_coeff,
                        confidence_score=confidence,
                        delay_ms=0.0,
                        sample_count=len(values1),
                        time_overlap=100.0,
                        description=f"Linear correlation (r={correlation_coeff:.3f}, p={p_value:.3f})",
                        gateway_candidate=is_gateway
                    )
        except:
            pass
        
        return None
    
    def _test_delayed_correlation(self, values1: List[float], values2: List[float],
                                timestamps: List[float], sig1: str, proto1: str,
                                sig2: str, proto2: str, sig1_data: pd.DataFrame,
                                sig2_data: pd.DataFrame) -> Optional[SignalCorrelation]:
        """Test for delayed correlation (signal appears later in another protocol)"""
        
        # Test different delays
        max_delay_samples = min(10, len(values1) // 2)
        
        best_correlation = 0
        best_delay = 0
        
        for delay in range(1, max_delay_samples):
            if delay >= len(values1) or delay >= len(values2):
                break
                
            try:
                # Shift signal2 by delay samples
                corr_coeff, _ = stats.pearsonr(values1[:-delay], values2[delay:])
                
                if abs(corr_coeff) > abs(best_correlation):
                    best_correlation = corr_coeff
                    best_delay = delay
            except:
                continue
        
        if abs(best_correlation) >= self.correlation_thresholds['min_correlation']:
            # Estimate delay in milliseconds
            if best_delay < len(timestamps) - 1:
                time_delay = (timestamps[best_delay] - timestamps[0]) * 1000
            else:
                time_delay = best_delay * 10  # Assume 10ms per sample
            
            confidence = min(100, abs(best_correlation) * 90)  # Slightly lower confidence for delayed
            
            if confidence >= self.correlation_thresholds['min_confidence']:
                return SignalCorrelation(
                    signal1_name=sig1,
                    signal1_protocol=proto1,
                    signal1_message=sig1_data['message'].iloc[0],
                    signal2_name=sig2,
                    signal2_protocol=proto2,
                    signal2_message=sig2_data['message'].iloc[0],
                    correlation_type=CorrelationType.DELAYED,
                    correlation_coefficient=best_correlation,
                    confidence_score=confidence,
                    delay_ms=time_delay,
                    sample_count=len(values1) - best_delay,
                    time_overlap=100.0,
                    description=f"Delayed correlation (r={best_correlation:.3f}, delay={time_delay:.1f}ms)",
                    gateway_candidate=True
                )
        
        return None
    
    def _test_inverse_correlation(self, values1: List[float], values2: List[float],
                                timestamps: List[float], sig1: str, proto1: str,
                                sig2: str, proto2: str, sig1_data: pd.DataFrame,
                                sig2_data: pd.DataFrame) -> Optional[SignalCorrelation]:
        """Test for inverse correlation (signals move in opposite directions)"""
        
        try:
            # Test correlation with inverted signal2
            inverted_values2 = [-v for v in values2]
            correlation_coeff, p_value = stats.pearsonr(values1, inverted_values2)
            
            if correlation_coeff >= self.correlation_thresholds['min_correlation']:
                confidence = min(100, correlation_coeff * 100 * (1 - p_value))
                
                if confidence >= self.correlation_thresholds['min_confidence']:
                    return SignalCorrelation(
                        signal1_name=sig1,
                        signal1_protocol=proto1,
                        signal1_message=sig1_data['message'].iloc[0],
                        signal2_name=sig2,
                        signal2_protocol=proto2,
                        signal2_message=sig2_data['message'].iloc[0],
                        correlation_type=CorrelationType.INVERSE,
                        correlation_coefficient=-correlation_coeff,  # Store as negative
                        confidence_score=confidence,
                        delay_ms=0.0,
                        sample_count=len(values1),
                        time_overlap=100.0,
                        description=f"Inverse correlation (r={-correlation_coeff:.3f})",
                        gateway_candidate=False
                    )
        except:
            pass
        
        return None
    
    def get_gateway_candidates(self) -> List[SignalCorrelation]:
        """Get correlations that are likely gateway signal relays"""
        return [corr for corr in self.correlations if corr.gateway_candidate]
    
    def get_correlations_by_protocol_pair(self, proto1: str, proto2: str) -> List[SignalCorrelation]:
        """Get correlations between specific protocol pair"""
        return [
            corr for corr in self.correlations 
            if (corr.signal1_protocol == proto1 and corr.signal2_protocol == proto2) or
               (corr.signal1_protocol == proto2 and corr.signal2_protocol == proto1)
        ]
    
    def generate_correlation_report(self) -> str:
        """Generate human-readable correlation report"""
        if not self.correlations:
            return "No correlations found"
        
        report = []
        report.append("🔗 CROSS-PROTOCOL CORRELATION REPORT")
        report.append("=" * 50)
        
        # Summary
        gateway_candidates = len(self.get_gateway_candidates())
        avg_confidence = np.mean([c.confidence_score for c in self.correlations])
        
        report.append(f"\n📊 Summary:")
        report.append(f"   Total correlations: {len(self.correlations)}")
        report.append(f"   Gateway candidates: {gateway_candidates}")
        report.append(f"   Average confidence: {avg_confidence:.1f}%")
        
        # Group by protocol pairs
        protocol_pairs = defaultdict(list)
        for corr in self.correlations:
            key = f"{corr.signal1_protocol} ↔ {corr.signal2_protocol}"
            protocol_pairs[key].append(corr)
        
        report.append(f"\n🌐 By Protocol Pair:")
        for pair, correlations in protocol_pairs.items():
            report.append(f"   {pair}: {len(correlations)} correlations")
        
        # Top correlations
        report.append(f"\n🏆 Top Correlations:")
        for i, corr in enumerate(self.correlations[:10]):
            report.append(f"   {i+1}. {corr.signal1_name} ({corr.signal1_protocol}) ↔ "
                         f"{corr.signal2_name} ({corr.signal2_protocol})")
            report.append(f"      Type: {corr.correlation_type.value}, "
                         f"Confidence: {corr.confidence_score:.1f}%, "
                         f"Delay: {corr.delay_ms:.1f}ms")
        
        # Gateway candidates
        if gateway_candidates > 0:
            report.append(f"\n🚪 Gateway Signal Candidates:")
            for corr in self.get_gateway_candidates():
                report.append(f"   {corr.signal1_name} → {corr.signal2_name}")
                report.append(f"      {corr.signal1_protocol} to {corr.signal2_protocol}, "
                             f"Delay: {corr.delay_ms:.1f}ms")
        
        return "\n".join(report)