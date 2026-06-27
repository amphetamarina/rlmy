#!/usr/bin/env bash
# rlmy installer — one command to get everything running.
# Usage: curl -LsSf https://raw.githubusercontent.com/diego-lima/rlmy/main/setup_install.sh | bash
#    or: bash setup_install.sh
set -euo pipefail

echo "🚀 Installing rlmy..."
echo ""


# ── 1. uv (Python package manager) ──────────────────────────────────────────
echo "=== Installing uv (Python package manager) ==="
if command -v uv &>/dev/null; then
    echo "  ✓ uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ""

# ── 2. rlmy (installed as a global CLI tool) ─────────────────────────────────
echo "=== Installing rlmy ==="
uv tool install rlmy
echo ""

# ── 3. MCP tools config (optional) ──────────────────────────────────────────
CONFIG_DIR="$HOME/.config/rlmy"
mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/mcp_servers.json" ]; then
    echo "=== MCP tools (optional) ==="
    echo '{"servers":{}}' > "$CONFIG_DIR/mcp_servers.json"
    echo "  Created $CONFIG_DIR/mcp_servers.json"
    echo "  Edit this file later to connect Slack, internal tools, etc."
    echo ""
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "✅ Installation complete!"
echo ""
echo "Next step:"
echo "  rlmy"
echo ""
echo "First run will ask which AI model to use."
echo "(You'll need API credentials for your chosen provider.)"
