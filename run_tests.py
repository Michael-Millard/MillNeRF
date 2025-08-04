#!/usr/bin/env python3
"""
Test runner script for NeRF implementation.
Run this from the project root directory.
"""

import subprocess
import sys
import os

def main():
    """Run all tests."""
    print("🧪 Running NeRF implementation tests...")
    print("=" * 50)
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run the test file
    try:
        # Use the virtual environment python if available
        python_cmd = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
        result = subprocess.run([python_cmd, "tests/test_nerf.py"], 
                              capture_output=False, 
                              text=True)
        
        print("=" * 50)
        if result.returncode == 0:
            print("✅ All tests completed successfully!")
        else:
            print("❌ Some tests failed. Check the output above.")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
