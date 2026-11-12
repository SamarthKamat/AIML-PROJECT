#!/usr/bin/env bash
# Download the two REAL datasets at pinned commits and verify integrity.
# DataCo original: Constante, Silva & Pereira (2019), Mendeley Data V5, doi:10.17632/8gx2fvg2k6.5 (CC BY 4.0).
#   If Mendeley is reachable you may instead download DataCoSupplyChainDataset.csv from there; the SHA-256 below is the mirror file.
# SupplyGraph: Wasi, Islam & Akib (2024), arXiv:2401.15299 (repository licence LGPL-2.1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
git clone --quiet https://github.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS.git "$TMP/dataco"
git -C "$TMP/dataco" checkout --quiet 7a466e7a4e9a8705c9b7b081dc081d2d16907fbc
mkdir -p "$ROOT/datasets/raw/dataco"
cp "$TMP/dataco/DataCoSupplyChainDataset.csv" "$TMP/dataco/DescriptionDataCoSupplyChain.csv" "$TMP/dataco/README.md" "$ROOT/datasets/raw/dataco/"
echo "994b3c8d24049cc46bf20e8161fedece9a9b95376cc0dc929921d76cf6b9bc0d  $ROOT/datasets/raw/dataco/DataCoSupplyChainDataset.csv" | sha256sum -c -
git clone --quiet https://github.com/CIOL-SUST/SupplyGraph.git "$TMP/sg"
git -C "$TMP/sg" checkout --quiet 0e6d010e256ee08e34ab6bef5a111ef1c758b32d
mkdir -p "$ROOT/datasets/raw/supplygraph"
cp -r "$TMP/sg/Raw Dataset" "$TMP/sg/LICENSE" "$TMP/sg/README.md" "$ROOT/datasets/raw/supplygraph/"
rm -rf "$TMP"
echo "Real datasets downloaded and verified."
