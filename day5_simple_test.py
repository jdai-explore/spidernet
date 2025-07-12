#!/usr/bin/env python3
"""
day5_simple_test.py
Simple Day 5 test runner
"""

def test_day5():
    """Simple Day 5 test"""
    print("🔬 Simple Day 5 Test")
    print("-" * 20)
    
    try:
        from day5_integration import Day5IntegratedAnalyzer
        print("✅ Day 5 integration imported")
        
        analyzer = Day5IntegratedAnalyzer()
        print("✅ Analyzer created")
        
        configs = [{"protocol": "CAN", "database": "test.dbc", "path": "test.asc"}]
        results = analyzer.analyze_complete_network_enhanced(configs)
        
        if results:
            print("✅ Analysis completed")
            enhanced = results.get('enhanced_latency_analysis', {})
            print(f"📊 Enhanced results: {enhanced.get('enhanced_results_count', 0)}")
            return True
        else:
            print("❌ No results")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_day5()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    exit(0 if success else 1)
