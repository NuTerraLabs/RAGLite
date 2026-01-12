#!/bin/bash
# Simple getting started script for ContextTape

set -e

echo "🚀 ContextTape Quick Start"
echo "=========================="
echo ""

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set"
    echo "   Run: export OPENAI_API_KEY=\"sk-...\""
    exit 1
fi

echo "✅ OPENAI_API_KEY is set"
echo ""

# Build Wikipedia knowledge base
echo "📚 Building Wikipedia knowledge base (3 pages)..."
echo "   This will create data/wiki/ directory"
echo ""

ct build-wiki --topics-file example_topics.txt --limit 3 --verbose

echo ""
echo "🎉 Done! Now try:"
echo "   ct search \"What is machine learning?\""
echo "   ct chat"
echo ""
