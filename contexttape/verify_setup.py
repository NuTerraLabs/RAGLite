#!/usr/bin/env python3
"""
ContextTape System Verification Script
=======================================

Comprehensive test to verify all system components work correctly.
Run this after installation to ensure everything is set up properly.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text):
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    version = sys.version_info
    print_info(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 9):
        print_success("Python version is compatible")
        return True
    else:
        print_error(f"Python 3.9+ required, found {version.major}.{version.minor}")
        return False


def check_imports():
    """Check if all required modules can be imported."""
    print_header("Checking Package Imports")
    
    modules = [
        ("numpy", "NumPy"),
        ("contexttape", "ContextTape"),
        ("contexttape.storage", "Storage module"),
        ("contexttape.embed", "Embedding module"),
        ("contexttape.search", "Search module"),
        ("contexttape.cli", "CLI module"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print_success(f"{display_name} import successful")
        except ImportError as e:
            print_error(f"{display_name} import failed: {e}")
            all_ok = False
    
    return all_ok


def check_version():
    """Check package version."""
    print_header("Checking Package Version")
    try:
        import contexttape
        version = contexttape.__version__
        print_info(f"ContextTape version: {version}")
        print_success("Version information available")
        return True
    except Exception as e:
        print_error(f"Could not get version: {e}")
        return False


def test_basic_storage():
    """Test basic storage operations."""
    print_header("Testing Basic Storage Operations")
    
    try:
        import numpy as np
        from contexttape import TSStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store
            store = TSStore(tmpdir)
            print_success("Store creation successful")
            
            # Append text
            text_id = store.append_text("Test document")
            print_success(f"Text append successful (ID: {text_id})")
            
            # Read text
            retrieved = store.read_text(text_id)
            assert retrieved == "Test document"
            print_success("Text retrieval successful")
            
            # Append with embedding
            emb = np.random.randn(1536).astype(np.float32)
            tid, vid = store.append_text_with_embedding("Doc with embedding", emb)
            print_success(f"Text+embedding append successful (text:{tid}, vec:{vid})")
            
            # Search
            query = np.random.randn(1536).astype(np.float32)
            results = store.search_by_vector(query, top_k=1)
            assert len(results) > 0
            print_success(f"Vector search successful (found {len(results)} results)")
            
        return True
    except Exception as e:
        print_error(f"Storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization():
    """Test int8 quantization."""
    print_header("Testing Int8 Quantization")
    
    try:
        import numpy as np
        from contexttape import TSStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TSStore(tmpdir)
            
            # Test quantized storage
            emb = np.random.randn(1536).astype(np.float32)
            tid, vid = store.append_text_with_embedding(
                "Quantized doc", emb, quantize=True
            )
            print_success("Quantized embedding stored")
            
            # Read back and verify
            retrieved = store.read_vector(vid)
            assert retrieved.dtype == np.float32
            assert len(retrieved) == 1536
            print_success("Quantized embedding retrieved and dequantized")
            
            # Check size savings
            import os
            seg_path = store._segment_path(vid)
            size = os.path.getsize(seg_path)
            expected_f32 = 32 + 1536 * 4  # header + float32 array
            print_info(f"Quantized segment size: {size} bytes")
            print_info(f"Float32 would be: {expected_f32} bytes")
            print_success(f"Space savings: ~{(1 - size/expected_f32)*100:.1f}%")
            
        return True
    except Exception as e:
        print_error(f"Quantization test failed: {e}")
        return False


def test_multi_store():
    """Test multi-store functionality."""
    print_header("Testing Multi-Store")
    
    try:
        import numpy as np
        from contexttape import TSStore, MultiStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple stores
            store1 = TSStore(os.path.join(tmpdir, "store1"))
            store2 = TSStore(os.path.join(tmpdir, "store2"))
            
            # Add data
            for store in [store1, store2]:
                emb = np.random.randn(1536).astype(np.float32)
                store.append_text_with_embedding("Document", emb)
            
            print_success("Multiple stores created and populated")
            
            # Search across both
            multi = MultiStore([store1, store2])
            query = np.random.randn(1536).astype(np.float32)
            results = multi.search(query, per_shard_k=2, final_k=2)
            
            assert len(results) > 0
            print_success(f"Multi-store search successful (found {len(results)} results)")
            
        return True
    except Exception as e:
        print_error(f"Multi-store test failed: {e}")
        return False


def test_batch_operations():
    """Test batch operations."""
    print_header("Testing Batch Operations")
    
    try:
        import numpy as np
        from contexttape import TSStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TSStore(tmpdir)
            
            # Batch append
            texts = [f"Doc {i}" for i in range(10)]
            embeddings = [np.random.randn(1536).astype(np.float32) for _ in range(10)]
            
            results = store.append_batch(texts, embeddings, quantize=True)
            assert len(results) == 10
            print_success(f"Batch append successful ({len(results)} documents)")
            
            # Verify all stored
            pairs = store.list_pairs()
            assert len(pairs) == 10
            print_success(f"All documents verified ({len(pairs)} pairs)")
            
        return True
    except Exception as e:
        print_error(f"Batch operations test failed: {e}")
        return False


def test_cli_available():
    """Test if CLI is accessible."""
    print_header("Testing CLI Availability")
    
    try:
        import subprocess
        result = subprocess.run(
            ["ct", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print_success("CLI 'ct' command available")
            print_info(f"Output: {result.stdout.strip()}")
            return True
        else:
            print_warning("CLI 'ct' command not found in PATH")
            print_info("You may need to reinstall or add to PATH")
            return False
    except FileNotFoundError:
        print_warning("CLI 'ct' command not found")
        return False
    except Exception as e:
        print_error(f"CLI test failed: {e}")
        return False


def test_openai_integration():
    """Test OpenAI integration if API key is available."""
    print_header("Testing OpenAI Integration")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_warning("OPENAI_API_KEY not set - skipping OpenAI tests")
        print_info("Set OPENAI_API_KEY to test embeddings")
        return None
    
    try:
        from contexttape import get_client, embed_text_1536
        
        client = get_client()
        print_success("OpenAI client created")
        
        # Test embedding
        emb = embed_text_1536(client, "Test text")
        assert len(emb) == 1536
        print_success(f"Text embedding successful (dim: {len(emb)})")
        
        return True
    except Exception as e:
        print_error(f"OpenAI integration failed: {e}")
        return False


def test_integrations():
    """Test framework integrations."""
    print_header("Testing Framework Integrations")
    
    try:
        from contexttape.integrations import (
            ContextTapeClient,
            check_integrations,
        )
        
        # Check which integrations are available
        available = check_integrations()
        
        print_info("Integration availability:")
        for name, is_available in available.items():
            if is_available:
                print_success(f"  {name}: available")
            else:
                print_info(f"  {name}: not available (optional)")
        
        # Test client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ContextTapeClient(tmpdir)
            print_success("ContextTapeClient instantiation successful")
        
        return True
    except Exception as e:
        print_error(f"Integrations test failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}ContextTape System Verification{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")
    
    tests = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Package Version", check_version),
        ("Basic Storage", test_basic_storage),
        ("Quantization", test_quantization),
        ("Multi-Store", test_multi_store),
        ("Batch Operations", test_batch_operations),
        ("CLI Availability", test_cli_available),
        ("OpenAI Integration", test_openai_integration),
        ("Framework Integrations", test_integrations),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print_error(f"Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    total = len(results)
    
    for name, result in results.items():
        if result is True:
            print_success(f"{name}: PASSED")
        elif result is False:
            print_error(f"{name}: FAILED")
        else:
            print_warning(f"{name}: SKIPPED")
    
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"Total tests: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"{YELLOW}Skipped: {skipped}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")
    
    if failed == 0:
        print(f"{GREEN}✓ All tests passed! ContextTape is ready to use.{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Please check the errors above.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
