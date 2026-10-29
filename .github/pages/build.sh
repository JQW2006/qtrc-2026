#!/usr/bin/env bash
# 把 report/ 组装成 mkdocs 的源目录，report/ 本身不做任何改动。
set -euo pipefail
cd "$(dirname "$0")/../.."
rm -rf _pages_src
cp -r report _pages_src
mv _pages_src/overview.md _pages_src/index.md
mkdir -p _pages_src/javascripts
cp .github/pages/mathjax.js _pages_src/javascripts/
