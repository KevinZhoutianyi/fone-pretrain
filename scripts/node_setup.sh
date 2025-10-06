
# 1) Load cluster modules (best-effort)
module load anaconda3_gpu cuda >/dev/null 2>&1 || true

# 2) Activate conda env
if ! command -v conda >/dev/null 2>&1; then
  echo "[node_setup] conda not in PATH; skipping activation"
else
  eval "$(conda shell.bash hook)"
  conda activate fone || echo "[node_setup] Failed to activate 'fone' env; ensure it exists"
fi

# 3) Project env (WORK_HDD_DIR / PROJECT_DIR, etc.)
if [ -f "$REPO_ROOT/setup_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/setup_env.sh"
else
  echo "[node_setup] setup_env.sh not found; continuing"
fi

# 4) Load .env for HF token (if present)
if [ -f "/u/tzhou4/fone/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "/u/tzhou4/fone/.env"
  set +a
fi

# Normalize HF token env var
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# 5) Put HF caches on HDD to avoid HOME quota/latency
if [ -n "${WORK_HDD_DIR:-}" ]; then
  export HF_HOME="$WORK_HDD_DIR/.cache/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_HUB_CACHE="$HF_HOME/hub"
  mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
  echo "[node_setup] HF caches → $HF_HOME"
fi

# 6) Non-interactive login to Hugging Face (if token available)
if command -v huggingface-cli >/dev/null 2>&1 && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential --non-interactive || true
fi

# 7) PYTHONPATH so src/ imports work
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "[node_setup] Done. Summary:"
echo "  CONDA_PREFIX = ${CONDA_PREFIX:-<unset>}"
echo "  WORK_HDD_DIR = ${WORK_HDD_DIR:-<unset>}"
echo "  PROJECT_DIR  = ${PROJECT_DIR:-<unset>}"
echo "  HF_HOME      = ${HF_HOME:-<unset>}"
echo "  HF token set = $([ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] && echo yes || echo no)"


