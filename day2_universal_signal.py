#!/usr/bin/env python3
"""
Day 2: Universal Signal Structure & Multi-Protocol Foundation
Support for CAN, LIN, and extensible protocol architecture
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

class ProtocolType(Enum):
    CAN = "CAN"
    LIN = "LIN"
    ETHERNET = "ETHERNET"
    J1939 = "J1939"
    UNKNOWN = "UNKNOWN"

@dataclass
class UniversalSignal:
    """Universal signal structure for all protocols"""
    # Core identification
    timestamp: float
    protocol: ProtocolType
    signal_name: str
    message_name: str
    value: Union[float, int, str, bool]
    
    # Protocol-specific details
    message_id: int
    channel: int = 1
    
    # Signal metadata
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    scaling: float = 1.0
    offset: float = 0.0
    
    # Quality indicators
    quality_flags: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = {}

class ProtocolParser:
    """Base class for protocol parsers"""
    
    def __init__(self, protocol: ProtocolType):
        self.protocol = protocol
        self.database = None
    
    def load_database(self, db_path: str) -> bool:
        """Load protocol database (DBC, LDF, etc.)"""
        raise NotImplementedError
    
    def parse_log(self, log_path: str) -> List[UniversalSignal]:
        """Parse log file and return universal signals"""
        raise NotImplementedError
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """Get information about loaded protocol database"""
        raise NotImplementedError

class CANParser(ProtocolParser):
    """CAN protocol parser using our Day 1 foundation"""
    
    def __init__(self):
        super().__init__(ProtocolType.CAN)
    
    def load_database(self, dbc_path: str) -> bool:
        """Load CAN DBC database"""
        try:
            import cantools
            self.database = cantools.database.load_file(dbc_path)
            return True
        except Exception as e:
            print(f"❌ CAN DBC load failed: {e}")
            return False
    
    def parse_log(self, log_path: str) -> List[UniversalSignal]:
        """Parse CAN ASC log file"""
        if not self.database:
            return []
        
        signals = []
        
        try:
            with open(log_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    
                    # Skip headers and comments
                    if not line or line.startswith(('date', 'base', '//', 'Begin', 'End', 'internal')):
                        continue
                    
                    # Parse CAN message
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    
                    try:
                        timestamp = float(parts[0])
                        channel = int(parts[1])
                        can_id = int(parts[2], 16)
                        dlc = int(parts[5])
                        data_parts = parts[6:6+dlc]
                        
                        if len(data_parts) < dlc:
                            continue
                        
                        data = bytes([int(b, 16) for b in data_parts])
                        
                        # Find message in database
                        try:
                            db_message = self.database.get_message_by_frame_id(can_id)
                            decoded = db_message.decode(data)
                            
                            # Create UniversalSignal for each signal
                            for signal_name, value in decoded.items():
                                db_signal = next(s for s in db_message.signals if s.name == signal_name)
                                
                                universal_signal = UniversalSignal(
                                    timestamp=timestamp,
                                    protocol=ProtocolType.CAN,
                                    signal_name=signal_name,
                                    message_name=db_message.name,
                                    value=value,
                                    message_id=can_id,
                                    channel=channel,
                                    unit=db_signal.unit or "",
                                    min_value=db_signal.minimum,
                                    max_value=db_signal.maximum,
                                    scaling=db_signal.scale,
                                    offset=db_signal.offset,
                                    quality_flags={"raw_data": data.hex()}
                                )
                                
                                signals.append(universal_signal)
                        
                        except (KeyError, Exception):
                            # Message not in DBC or decode error
                            continue
                    
                    except (ValueError, IndexError):
                        continue
            
            return signals
            
        except Exception as e:
            print(f"❌ CAN log parsing failed: {e}")
            return []
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """Get CAN database information"""
        if not self.database:
            return {}
        
        return {
            "protocol": "CAN",
            "messages": len(self.database.messages),
            "nodes": len(self.database.nodes),
            "total_signals": sum(len(msg.signals) for msg in self.database.messages)
        }

class LINParser(ProtocolParser):
    """LIN protocol parser - basic implementation"""
    
    def __init__(self):
        super().__init__(ProtocolType.LIN)
        self.schedule = {}
        self.signals = {}
    
    def load_database(self, ldf_path: str) -> bool:
        """Load LIN Description File (simplified)"""
        try:
            # Basic LDF parsing - real implementation would use proper LDF parser
            with open(ldf_path, 'r') as f:
                content = f.read()
            
            # Extract basic frame information (simplified)
            # In real implementation, use proper LDF parser library
            self.signals = {
                "LIN_Signal1": {"frame": "Frame1", "start_bit": 0, "length": 8},
                "LIN_Signal2": {"frame": "Frame2", "start_bit": 8, "length": 16}
            }
            
            print(f"✅ LIN LDF loaded: {len(self.signals)} signals")
            return True
            
        except Exception as e:
            print(f"❌ LIN LDF load failed: {e}")
            return False
    
    def parse_log(self, log_path: str) -> List[UniversalSignal]:
        """Parse LIN log file (basic implementation)"""
        signals = []
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Basic LIN log format: timestamp frame_id data
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            timestamp = float(parts[0])
                            frame_id = int(parts[1], 16)
                            data_hex = parts[2]
                            
                            # Create dummy signal for demo
                            universal_signal = UniversalSignal(
                                timestamp=timestamp,
                                protocol=ProtocolType.LIN,
                                signal_name=f"LIN_Signal_{frame_id}",
                                message_name=f"LIN_Frame_{frame_id}",
                                value=int(data_hex[:2], 16),  # First byte as value
                                message_id=frame_id,
                                channel=1,
                                unit="",
                                quality_flags={"raw_data": data_hex}
                            )
                            
                            signals.append(universal_signal)
                        
                        except ValueError:
                            continue
            
            return signals
            
        except Exception as e:
            print(f"❌ LIN log parsing failed: {e}")
            return []
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """Get LIN database information"""
        return {
            "protocol": "LIN",
            "signals": len(self.signals),
            "frames": len(set(s["frame"] for s in self.signals.values()))
        }

class ProtocolFactory:
    """Factory for creating protocol parsers"""
    
    @staticmethod
    def create_parser(protocol: ProtocolType) -> ProtocolParser:
        """Create appropriate parser for protocol"""
        if protocol == ProtocolType.CAN:
            return CANParser()
        elif protocol == ProtocolType.LIN:
            return LINParser()
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    
    @staticmethod
    def detect_protocol(file_path: str) -> ProtocolType:
        """Auto-detect protocol from file extension/content"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.dbc':
            return ProtocolType.CAN
        elif path.suffix.lower() == '.ldf':
            return ProtocolType.LIN
        elif path.suffix.lower() in ['.asc', '.blf']:
            # Could be CAN or LIN - need content analysis
            return ProtocolType.CAN  # Default to CAN for now
        
        return ProtocolType.UNKNOWN

class MultiProtocolAnalyzer:
    """Enhanced analyzer supporting multiple protocols"""
    
    def __init__(self):
        self.parsers = {}
        self.signals = []
    
    def add_protocol(self, protocol: ProtocolType, db_path: str) -> bool:
        """Add protocol support with database"""
        try:
            parser = ProtocolFactory.create_parser(protocol)
            if parser.load_database(db_path):
                self.parsers[protocol] = parser
                print(f"✅ {protocol.value} protocol added")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to add {protocol.value}: {e}")
            return False
    
    def analyze_log(self, log_path: str, protocol: ProtocolType = None) -> List[UniversalSignal]:
        """Analyze log file with specified or auto-detected protocol"""
        if protocol is None:
            protocol = ProtocolFactory.detect_protocol(log_path)
        
        if protocol not in self.parsers:
            print(f"❌ No parser available for {protocol.value}")
            return []
        
        signals = self.parsers[protocol].parse_log(log_path)
        self.signals.extend(signals)
        
        print(f"✅ Parsed {len(signals)} {protocol.value} signals")
        return signals
    
    def get_all_signals_df(self) -> pd.DataFrame:
        """Convert all signals to pandas DataFrame"""
        if not self.signals:
            return pd.DataFrame()
        
        rows = []
        for signal in self.signals:
            rows.append({
                'timestamp': signal.timestamp,
                'protocol': signal.protocol.value,
                'signal': signal.signal_name,
                'message': signal.message_name,
                'value': signal.value,
                'message_id': signal.message_id,
                'channel': signal.channel,
                'unit': signal.unit,
                'min_value': signal.min_value,
                'max_value': signal.max_value
            })
        
        return pd.DataFrame(rows)
    
    def get_protocol_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded protocols"""
        summary = {}
        for protocol, parser in self.parsers.items():
            summary[protocol.value] = parser.get_protocol_info()
        
        if self.signals:
            df = self.get_all_signals_df()
            summary['analysis'] = {
                'total_signals': len(df),
                'protocols_used': df['protocol'].nunique(),
                'time_range': f"{df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}s",
                'unique_signals': df['signal'].nunique()
            }
        
        return summary