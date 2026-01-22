#!/bin/bash
# Simple Docker runner for benchmarks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

echo "════════════════════════════════════════════════════════════════"
echo "  Running Benchmarks in Docker"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Project: $SCRIPT_DIR"
echo "Results: $RESULTS_DIR"
echo ""

# Build image if needed
if ! docker images | grep -q homomorphic-auth; then
    echo "Building Docker image..."
    docker build -t homomorphic-auth . || exit 1
    echo ""
fi

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Run benchmarks
echo "Running: python run_benchmark.py"
echo ""
docker run --rm \
    -v "${RESULTS_DIR}:/app/results" \
    homomorphic-auth \
    python run_benchmark.py

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Complete! Results saved to: $RESULTS_DIR/benchmark/"
echo "════════════════════════════════════════════════════════════════"

