#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SOURCE_REPO="https://github.com/theophilegervet/learner-performance-prediction.git"
SOURCE_COMMIT="a7ae193aa6957003a764aed7c95b07666fd4f1da"

if [ -d data_npz/assistments09 ] || [ -f data/assistments09/preprocessed_data_train.csv ]; then
  echo "Benchmark data already present; nothing to do."
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "Cloning the pinned preprocessing release..."
git clone --depth 1 "$SOURCE_REPO" "$tmp/src"
cd "$tmp/src"
if [ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]; then
  git fetch --depth 1 origin "$SOURCE_COMMIT"
  git checkout --detach "$SOURCE_COMMIT"
fi
cd - >/dev/null

mkdir -p data
for d in assistments09 assistments12 assistments15 assistments17          algebra05 bridge_algebra06 spanish statics; do
  cp -r "$tmp/src/data/$d" data/
  echo "  copied $d"
done

python3 code/build_npz.py
echo "Done. The original dataset-provider terms continue to apply."
