#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/data"
CACHE_DIR="$PROJECT_DIR/cache"

MONACO_URL="https://download.geofabrik.de/europe/monaco-latest.osm.pbf"
MONACO_PBF="$DATA_DIR/monaco-latest.osm.pbf"
MONACO_GRAPH="$CACHE_DIR/monaco"

echo "=== GPUSSSP Integration Test ==="
echo ""

mkdir -p "$DATA_DIR" "$CACHE_DIR" "$MONACO_GRAPH" "${MONACO_GRAPH}_zorder"

echo "[1/4] Downloading Monaco OSM data..."
if [ ! -f "$MONACO_PBF" ]; then
    wget -q --show-progress -O "$MONACO_PBF" "$MONACO_URL"
else
    echo "  Already downloaded, skipping."
fi
echo ""

echo "[2/4] Converting OSM to graph format..."
"$BUILD_DIR/osm2graph" "$MONACO_PBF" "$MONACO_GRAPH"
echo ""

echo "[3/4] Reordering graph (z-order)..."
"$BUILD_DIR/graph_reorder" zorder "$MONACO_GRAPH" "${MONACO_GRAPH}_zorder"
echo ""

echo "[4/4] Running shortest path queries on all algorithms..."
cd "$BUILD_DIR" && ./gpusssp "${MONACO_GRAPH}_zorder" --skip "" -n 100
echo ""

echo "=== Integration test completed successfully ==="
