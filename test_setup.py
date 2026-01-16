"""
Quick test script to verify the project setup

Run this after installation to ensure everything is working correctly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from pe_common import DATA_ROOT, MODEL_ROOT, DEVICE
        print("✓ pe_common constants imported successfully")
        print(f"  - DATA_ROOT: {DATA_ROOT}")
        print(f"  - MODEL_ROOT: {MODEL_ROOT}")
        print(f"  - DEVICE: {DEVICE}")
    except ImportError as e:
        print(f"✗ Failed to import pe_common constants: {e}")
        return False
    
    try:
        from pe_common.sequence_utils import align_wt_mut_sequences, remove_padding
        print("✓ pe_common.sequence_utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sequence_utils: {e}")
        return False
    
    try:
        from pe_common.features import calculate_mt_wallace, calculate_gc_content
        print("✓ pe_common.features imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import features: {e}")
        return False
    
    return True


def test_functionality():
    """Test basic functionality"""
    print("\nTesting functionality...")
    
    try:
        from pe_common.sequence_utils import remove_padding, align_wt_mut_sequences
        from pe_common.features import calculate_gc_content
        
        # Test remove_padding
        test_seq = "ATCG-N-X"
        cleaned = remove_padding(test_seq)
        assert cleaned == "ATCG", f"Expected 'ATCG', got '{cleaned}'"
        print("✓ remove_padding works correctly")
        
        # Test GC content
        gc = calculate_gc_content("ATCG")
        assert abs(gc - 0.5) < 0.01, f"Expected 0.5, got {gc}"
        print("✓ calculate_gc_content works correctly")
        
        # Test align sequences
        wt, mut = align_wt_mut_sequences("ATCG", "ATGCG", 2, 1, 1)
        print("✓ align_wt_mut_sequences works correctly")
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False
    
    return True


def test_service_structure():
    """Test that service structure exists"""
    print("\nTesting service structure...")
    
    project_root = Path(__file__).parent
    
    # Check pe-common
    pe_common = project_root / "packages" / "pe-common"
    if pe_common.exists():
        print("✓ packages/pe-common exists")
    else:
        print("✗ packages/pe-common not found")
        return False
    
    # Check PE Database service
    pe_db = project_root / "services" / "pe-db" / "app"
    if pe_db.exists():
        print("✓ services/pe-db/app exists")
    else:
        print("✗ services/pe-db/app not found")
        return False
    
    # Check main.py
    main_py = pe_db / "main.py"
    if main_py.exists():
        print("✓ services/pe-db/app/main.py exists")
    else:
        print("✗ services/pe-db/app/main.py not found")
        return False
    
    # Check data_prep
    data_prep = pe_db / "data_prep"
    if data_prep.exists():
        print("✓ services/pe-db/app/data_prep exists")
    else:
        print("✗ services/pe-db/app/data_prep not found")
        return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("PE-DB Project Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Functionality Tests", test_functionality),
        ("Structure Tests", test_service_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run PE Database: cd services/pe-db && uvicorn app.main:app --reload")
        print("  2. Access API docs: http://localhost:8000/docs")
        print("  3. Or use Docker: docker-compose up pe-db")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure pe-common is installed: pip install -e packages/pe-common")
        print("  2. Check your Python environment is activated")
        print("  3. Run setup script: ./setup-dev.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
