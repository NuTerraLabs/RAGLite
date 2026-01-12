#!/usr/bin/env bash
# cleanup_stores.sh - Remove all temporary test/example stores
#
# This script removes all dynamically generated store directories
# created by running examples and tests. It does NOT remove source
# code, configuration, or documentation.
#
# Usage: bash cleanup_stores.sh

set -e

echo "🧹 ContextTape Store Cleanup"
echo "=============================="
echo ""
echo "This will remove all temporary stores created by examples and tests."
echo "Your source code and configuration will NOT be affected."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# List what will be removed
echo "📋 Stores to be removed:"
echo ""

STORES_FOUND=0

# Find all store directories
for pattern in "*_store" "*_ts" "tutorial_*" "multi_*" "hierarchy"; do
    if compgen -G "$pattern" > /dev/null 2>&1; then
        for dir in $pattern; do
            if [ -d "$dir" ]; then
                echo "  - $dir/ ($(du -sh "$dir" 2>/dev/null | cut -f1))"
                STORES_FOUND=$((STORES_FOUND + 1))
            fi
        done
    fi
done

if [ $STORES_FOUND -eq 0 ]; then
    echo "  ✓ No stores found. Directory is already clean!"
    echo ""
    exit 0
fi

echo ""
echo "Total: $STORES_FOUND store(s)"
echo ""

# Ask for confirmation
read -p "⚠️  Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled."
    exit 1
fi

echo ""
echo "🗑️  Removing stores..."
echo ""

REMOVED=0

# Remove all pattern-matched directories
for pattern in "*_store" "*_ts" "tutorial_*" "multi_*" "hierarchy"; do
    if compgen -G "$pattern" > /dev/null 2>&1; then
        for dir in $pattern; do
            if [ -d "$dir" ]; then
                echo "  Removing: $dir/"
                rm -rf "$dir"
                REMOVED=$((REMOVED + 1))
            fi
        done
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo "   Removed $REMOVED store(s)"
echo ""
echo "📦 Your source code and configuration are intact."
echo "   Run examples again to regenerate stores."
echo ""
