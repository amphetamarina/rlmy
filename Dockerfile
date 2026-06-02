# rlmy — clean-room Docker image for testing.
# Modern base (Debian bookworm, glibc 2.36) so Deno + onnxruntime wheels resolve.
# Python pinned to 3.12 (best wheel coverage; avoids the cp313 gap).
FROM python:3.12-slim

# System deps: curl (deno + uv installers), unzip (deno), ca-certificates, git (uv builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# --- Deno (WASM sandbox) ---
RUN curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/root/.deno/bin:${PATH}"
# Verify Deno actually RUNS (not just exists) — catches glibc-style failures early
RUN deno --version

# --- uv (package manager) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# --- rlmy ---
# Default: install from PyPI (realistic end-user test).
# To test LOCAL source instead, comment the line below and use the COPY/-e block.
RUN uv tool install --python 3.12 rlmy

# (Alternative — test current local source instead of PyPI:)
# COPY . /app
# RUN uv tool install --python 3.12 /app

# uv tool puts the rlmy binary on PATH here
ENV PATH="/root/.local/bin:${PATH}"

# Sandbox lives in the centralized config dir by default (~/.config/rlmy/sandboxes/).
# Mount a volume there if you want trajectories to persist across container runs.

ENTRYPOINT ["rlmy"]
