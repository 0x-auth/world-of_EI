#!/usr/bin/env bash
# Pushes THREAD_COORDINATES.md into your world-of_EI repo.
# Assumes your repos live at ~/github-repos  (adjust REPO if different).
set -euo pipefail

REPO="$HOME/github-repos/world-of_EI"
[ -d "$REPO" ] || REPO="$HOME/github-repos/world-of_ei"
[ -d "$REPO" ] || { echo "repo not found under ~/github-repos — set REPO= manually"; exit 1; }

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$REPO/experiments/organism_v0"
mkdir -p "$DEST"
cp "$SRC_DIR/THREAD_COORDINATES.md" "$DEST/THREAD_COORDINATES.md"

cd "$REPO"
git add experiments/organism_v0/THREAD_COORDINATES.md
git commit -m "Add THREAD_COORDINATES.md: connect organism_v0 to thread mathematics"
git push

echo "pushed THREAD_COORDINATES.md to $REPO"
