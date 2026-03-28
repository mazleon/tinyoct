#!/usr/bin/env bash
# =============================================================================
# TinyOCT — RunPod Server Setup Script
# =============================================================================
# One-shot setup for a fresh RunPod GPU pod.
# Run this immediately after SSHing into the pod:
#
#   bash setup_runpod.sh
#
# What it does:
#   1. Installs uv (fast Python package manager)
#   2. Syncs project dependencies (uv sync)
#   3. Downloads OCT2017 + OCTID from Google Drive
#   4. Auto-downloads OCTMNIST
#   5. Runs a smoke test to verify the full pipeline
#
# Prerequisites:
#   - Python 3.11+ available as 'python3'
#   - Project is already cloned / uploaded to this pod
#   - Google Drive folder is shared as "Anyone with the link"
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✅ OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[⚠️  WARN]${NC} $*"; }
err()  { echo -e "${RED}[❌ ERR]${NC} $*" >&2; }
info() { echo -e "       $*"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  TinyOCT — RunPod Server Setup"
echo "══════════════════════════════════════════════════════════"
echo "  Pod:  $(hostname)"
echo "  Dir:  $(pwd)"
echo "  Date: $(date)"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── 0. Check we are in the project root ──────────────────────────────────────
if [[ ! -f "pyproject.toml" ]]; then
    err "pyproject.toml not found. Run this script from the TinyOCT project root."
    err "  cd /path/to/tinyoct && bash setup_runpod.sh"
    exit 1
fi
ok "Project root confirmed: $(pwd)"

# ── 1. Install uv ─────────────────────────────────────────────────────────────
echo ""
info "Step 1 — Installing uv (Python package manager)..."
if command -v uv &>/dev/null; then
    ok "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH so uv is available
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed: $(uv --version)"
    else
        err "uv installation failed. Install manually: https://github.com/astral-sh/uv"
        exit 1
    fi
fi

# ── 2. Sync Python dependencies ───────────────────────────────────────────────
echo ""
info "Step 2 — Installing Python dependencies (uv sync)..."
uv sync
ok "Dependencies installed."

# ── 3. Download datasets from Google Drive ────────────────────────────────────
echo ""
info "Step 3 — Downloading datasets from Google Drive..."
info "  (OCT2017 ~2 GB + OCTID, this may take several minutes)"
echo ""
uv run scripts/download_gdrive.py

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
echo ""
info "Step 4 — Running smoke test (3 epochs) to verify pipeline..."
echo ""
uv run scripts/train.py --config configs/smoketest.yaml
ok "Smoke test passed!"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Setup complete! You are ready to run full training."
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "    # Full training on OCT2017:"
echo "    uv run scripts/train.py --config configs/experiment_oct2017.yaml"
echo ""
echo "    # All ablation runs R0–R5 (for paper Table 2):"
echo "    uv run scripts/run_ablations.py"
echo ""
echo "    # Evaluate best checkpoint:"
echo "    uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate"
echo ""
