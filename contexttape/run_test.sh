#!/bin/bash
# Quick runner script to test ContextTape system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 ContextTape Quick Test Runner"
echo "================================="
echo ""

# Check if package is installed
if python -c "import contexttape" 2>/dev/null; then
    echo "✓ Package installed, using installed version"
    python test_system.py "$@"
else
    echo "✓ Using local source (development mode)"
    PYTHONPATH=src:$PYTHONPATH python test_system.py "$@"
fi
