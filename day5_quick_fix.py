#!/usr/bin/env python3
"""
day5_quick_fix.py
Quick fix for Day 5 import issues and validation
"""

import sys
import os
from pathlib import Path
import shutil

def fix_test_demo_imports():
    """Fix the import issues in day5_test_demo.py"""
    print("🔧 Fixing Day 5 Test Demo Imports")
    print("-" * 32)
    
    test_demo_file = Path("day5_test_demo.py")
    
    if not test_demo_file.exists():
        print("❌ day5_test_demo.py not found")
        return False
    
    # Read the current file
    with open(test_demo_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the import statements
    fixes = [
        ("from day5_integration_fix import Day5IntegratedAnalyzer", "from day5_integration import Day5IntegratedAnalyzer"),
        ("day5_integration_fix", "day5_integration"),
    ]
    
    original_content = content
    for old_import, new_import in fixes:
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"   ✅ Fixed: {old_import} → {new_import}")
    
    if content != original_content:
        # Backup original file
        backup_file = test_demo_file.with_suffix('.py.backup')
        shutil.copy2(test_demo_file, backup_file)
        print(f"   💾 Backup created: {backup_file}")
        
        # Write fixed content
        with open(test_demo_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("   ✅ day5_test_demo.py fixed successfully")
        return True
    else:
        print("   ℹ️  No import fixes needed")
        return True

def verify_day5_files():
    """Verify all Day 5 files are present"""
    print("\n📁 Verifying Day 5 Files")
    print("-" * 25)
    
    required_files = {
        'day5_latency_engine.py': 'Enhanced Latency Analysis Engine',
        'day5_integration.py': 'Integration Layer',
        'day5_fastapi_integration.py': 'FastAPI Integration',
        'day5_test_demo.py': 'Test and Demo Script'
    }
    
    all_present = True
    
    for filename, description in required_files.items():
        file_path = Path(filename)
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   ✅ {filename} ({size:,} bytes) - {description}")
        else:
            print(f"   ❌ {filename} - MISSING - {description}")
            all_present = False
    
    return all_present

def test_basic_imports():
    """Test basic Day 5 imports"""
    print("\n🔗 Testing Basic Imports")
    print("-" * 24)
    
    import_tests = [
        ('day5_latency_engine', 'Enhanced Latency Engine'),
        ('day5_integration', 'Integration Layer'),
        ('day5_fastapi_integration', 'FastAPI Integration')
    ]
    
    all_imports_ok = True
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name} - {description}")
        except ImportError as e:
            print(f"   ❌ {module_name} - {e}")
            all_imports_ok = False
        except Exception as e:
            print(f"   ⚠️  {module_name} - Warning: {e}")
    
    return all_imports_ok

def run_fixed_test():
    """Run the fixed Day 5 test"""
    print("\n🧪 Running Fixed Day 5 Test")
    print("-" * 27)
    
    try:
        # Try to import and run the main test function
        from day5_integration import Day5IntegratedAnalyzer
        
        print("   📊 Testing Day 5 integration...")
        analyzer = Day5IntegratedAnalyzer()
        
        # Test basic functionality
        print(f"   🔧 Day 3 ready: {getattr(analyzer, 'day3_ready', 'Unknown')}")
        print(f"   🔬 Day 5 ready: {getattr(analyzer, 'day5_ready', 'Unknown')}")
        
        # Test with simple config
        mock_configs = [{"protocol": "CAN", "database": "test.dbc", "path": "test.asc"}]
        
        print("   🔬 Running enhanced analysis...")
        import time
        start_time = time.time()
        
        results = analyzer.analyze_complete_network_enhanced(mock_configs)
        
        analysis_time = time.time() - start_time
        
        if results:
            print(f"   ✅ Analysis completed in {analysis_time:.2f}s")
            
            # Check results structure
            enhanced = results.get('enhanced_latency_analysis', {})
            metadata = results.get('analysis_metadata', {})
            
            print(f"   📊 Enhanced results: {enhanced.get('enhanced_results_count', 0)}")
            print(f"   📋 Metadata keys: {len(metadata)}")
            
            return True
        else:
            print("   ❌ No results returned")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_runner():
    """Create a simple test runner script"""
    print("\n📝 Creating Simple Test Runner")
    print("-" * 30)
    
    test_script = '''#!/usr/bin/env python3
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
    print(f"\\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    exit(0 if success else 1)
'''
    
    with open('day5_simple_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("   ✅ day5_simple_test.py created")
    return True

def main():
    """Main quick fix function"""
    print("🔧 DAY 5 QUICK FIX UTILITY")
    print("=" * 28)
    print("Fixing Day 5 import issues and validating setup")
    print()
    
    steps = [
        ("Verify Day 5 Files", verify_day5_files),
        ("Fix Test Demo Imports", fix_test_demo_imports),
        ("Test Basic Imports", test_basic_imports),
        ("Run Fixed Test", run_fixed_test),
        ("Create Simple Test Runner", create_simple_test_runner)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"   ❌ {step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n📊 Quick Fix Summary")
    print("-" * 20)
    
    passed = 0
    total = len(results)
    
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {step_name:<25} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n📈 Overall: {passed}/{total} steps passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("Day 5 should now work correctly.")
        print("\nNext steps:")
        print("1. Run: python day5_test_demo.py")
        print("2. Or run: python day5_simple_test.py")
        print("3. Check Day 5 functionality")
    elif success_rate >= 80:
        print("\n⚠️  Most fixes applied - minor issues remain")
        print("Review failed steps and apply manual fixes if needed")
    else:
        print("\n❌ Significant issues remain")
        print("Manual intervention required")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)