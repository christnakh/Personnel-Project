#!/bin/bash
# Docker setup for all 8 homomorphic authentication algorithms

echo "════════════════════════════════════════════════════════════════"
echo "  Docker Setup - All 8 Homomorphic Authentication Algorithms"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo ""
    echo "Please install Docker Desktop:"
    echo "  https://www.docker.com/products/docker-desktop"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo ""
echo "Building Docker image with Python 3.11..."
echo ""

if docker build -t homomorphic-auth .; then
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    echo ""
            echo -e "${CYAN}═══ Testing All Algorithms ═══${NC}"
            docker run --rm homomorphic-auth python run.py verify
    echo ""
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo ""
    echo "Usage:"
    echo ""
    echo "  # Run everything (files save to ./results/):"
    echo "  docker run --rm -v \"\$(pwd)/results:/app/results\" homomorphic-auth python run.py all --plots"
    echo ""
    echo "  # Individual commands:"
    echo "  docker run --rm -v \"\$(pwd)/results:/app/results\" homomorphic-auth python run.py test"
    echo "  docker run --rm -v \"\$(pwd)/results:/app/results\" homomorphic-auth python run.py verify"
    echo "  docker run --rm -v \"\$(pwd)/results:/app/results\" homomorphic-auth python run.py benchmark --plots"
    echo "  docker run --rm -v \"\$(pwd)/results:/app/results\" homomorphic-auth python run.py fl"
    echo ""
    echo "  # Alternative (if path has spaces, use absolute path):"
    echo "  docker run --rm -v \"\$PWD/results:/app/results\" homomorphic-auth python run.py all --plots"
    echo ""
    echo "  # Without Docker (7/10 algorithms work):"
    echo "  python3 run.py all --plots"
else
    echo ""
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
