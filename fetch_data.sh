#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SOURCE_REPO="https://github.com/theophilegervet/learner-performance-prediction.git"
SOURCE_COMMIT="a7ae193aa6957003a764aed7c95b07666fd4f1da"
DEST="data/benchmarks"

if [ -d "$DEST/assistments09" ]; then
  echo "Benchmark data already present at $DEST; nothing to do."
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "Cloning the pinned preprocessing release..."
git clone "$SOURCE_REPO" "$tmp/src"
cd "$tmp/src"
git checkout --detach "$SOURCE_COMMIT"
cd - >/dev/null

mkdir -p "$DEST"
for d in assistments09 assistments12 assistments15 assistments17          algebra05 bridge_algebra06 spanish statics; do
  cp -r "$tmp/src/data/$d" "$DEST/"
  echo "  copied $d"
done

echo "Done. Original dataset-provider terms continue to apply."
