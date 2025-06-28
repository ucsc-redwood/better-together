#!/usr/bin/env python3
"""
Simple test script to verify the modular SMT components work correctly.
"""

import sys
import os

# Add the parent directory to the path so we can import the smt package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        from smt.baselines import get_baseline_for_config, get_num_stages_for_app
        from smt.data_loader import load_csv_and_compute_averages, define_data
        from smt.constraints import create_decision_variables, create_optimizer
        from smt.solver import solve_optimization_problem
        from smt.solution_analyzer import get_detailed_solution, dump_solutions_as_json
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_baselines():
    """Test baseline functionality."""
    try:
        from smt.baselines import get_baseline_for_config, get_num_stages_for_app
        
        # Test baseline retrieval
        baseline = get_baseline_for_config("jetson", "cifar-dense-cu", "cu")
        assert baseline is not None, "Baseline should not be None"
        assert "omp" in baseline, "Baseline should contain 'omp' key"
        assert "cu" in baseline, "Baseline should contain 'cu' key"
        assert "fastest" in baseline, "Baseline should contain 'fastest' key"
        
        # Test stage count retrieval
        stages = get_num_stages_for_app("cifar-dense")
        assert stages == 9, f"Expected 9 stages for cifar-dense, got {stages}"
        
        stages = get_num_stages_for_app("tree")
        assert stages == 7, f"Expected 7 stages for tree, got {stages}"
        
        print("✓ Baseline functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Baseline test failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    try:
        from smt.data_loader import define_data
        
        # Test data definition
        num_stages, core_types, stage_timings = define_data(app_name="cifar-dense")
        assert num_stages == 9, f"Expected 9 stages, got {num_stages}"
        assert len(core_types) == 4, f"Expected 4 core types, got {len(core_types)}"
        assert "Little" in core_types, "Core types should include 'Little'"
        assert "GPU" in core_types, "Core types should include 'GPU'"
        
        print("✓ Data loader functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_constraints():
    """Test constraints functionality."""
    try:
        from smt.constraints import create_decision_variables, create_optimizer
        from smt.data_loader import define_data
        
        # Test optimizer creation
        opt = create_optimizer()
        assert opt is not None, "Optimizer should not be None"
        
        # Test decision variables creation
        num_stages, core_types, _ = define_data(app_name="cifar-dense")
        x = create_decision_variables(num_stages, core_types)
        assert len(x) == num_stages * len(core_types), "Should create one variable per stage-core combination"
        
        print("✓ Constraints functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Constraints test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing SMT modular components...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_baselines,
        test_data_loader,
        test_constraints,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The modular structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 