#!/usr/bin/env bash
set -e

# ==============================================================================
# More than a Hack 2027 - Official 1-Command Reproduction Script
# ==============================================================================
# Guarantees 100% environment consistency across all runs.
#
# Execution modes:
#   1. Docker (Default if Docker is running):
#      Builds and executes inside an isolated container with pinned packages.
#   2. Local Virtual Environment (Automatic fallback or with --local):
#      Automatically creates an isolated .venv_reproduce, installs pinned
#      dependencies, and runs cleanly without touching system packages.
# ==============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

USE_DOCKER=false
MODE_FLAG=""
RUN_CALIBRATE=false
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    if [ "$arg" == "--docker" ]; then
        USE_DOCKER=true
    elif [ "$arg" == "--local" ]; then
        USE_DOCKER=false
    elif [ "$arg" == "--calibrate" ] || [ "$arg" == "--pareto" ]; then
        RUN_CALIBRATE=true
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

if [ "$USE_DOCKER" = false ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    # Default to Docker if available and --local was not passed
    if [ "$1" != "--local" ] && [ "$2" != "--local" ]; then
        USE_DOCKER=true
    fi
fi

echo "======================================================================"
echo "More than a Hack 2027: Reproducing Official Submission File"
echo "======================================================================"

# Check match data
if [ ! -d "data/tactical" ] || [ ! -d "data/broadcast" ]; then
    echo "ERROR: Missing match data."
    echo "Please unpack the official match assets into ./data/tactical and ./data/broadcast."
    echo "See data/README.md for details."
    exit 1
fi

# ------------------------------------------------------------------------------
# MODE A: DOCKER CONTAINERIZED EXECUTION
# ------------------------------------------------------------------------------
if [ "$USE_DOCKER" = true ]; then
    echo "[Environment] Docker detected. Running in isolated container..."
    IMAGE_NAME="barcelona-possession:latest"

    # Always build/update Docker image (uses layer caching, fast if unchanged)
    echo "Building / verifying Docker image ($IMAGE_NAME)..."
    docker build -t "$IMAGE_NAME" -f Dockerfile .

    echo "Executing reproduction inside Docker container..."
    docker run --rm \
        -v "$ROOT/data:/workspace/data" \
        -v "$ROOT/models:/workspace/models" \
        -v "$ROOT/evaluation/results:/workspace/evaluation/results" \
        "$IMAGE_NAME" "$@"

    echo ""
    echo "======================================================================"
    echo "SUCCESS: Docker execution completed and submission file verified!"
    echo "======================================================================"
    exit 0
fi

# ------------------------------------------------------------------------------
# MODE B: ISOLATED VIRTUAL ENVIRONMENT EXECUTION
# ------------------------------------------------------------------------------
echo "[Environment] Running in isolated Python virtual environment..."
VENV_DIR="$ROOT/.venv_reproduce"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "Installing pinned dependencies into virtual environment..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt -r requirements-starter.txt
fi

PYTHON_EXEC="$VENV_DIR/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

# 0. Optional Pareto calibration if requested
if [ "$RUN_CALIBRATE" = true ]; then
    echo ""
    echo "[Step 0/3] Calibrating decoder hyperparameters via Metrica Pareto optimization (Maximin Robustness)..."
    "$PYTHON_EXEC" starter/pareto_metrica.py --save-model
fi

# 1. Run data prep if cache is missing (fresh clone)
if [ ! -f "cache/tactical_players.npy" ] || [ ! -f "cache/broadcast_players.npy" ]; then
    echo ""
    echo "[Step 1/3] Parsing raw tracking assets into memory-mapped cache (starter/prep.py --all)..."
    "$PYTHON_EXEC" starter/prep.py --all
fi

# 2. Run algorithmic baseline
echo ""
echo "[Step 2/3] Running algorithmic baseline decoder..."
"$PYTHON_EXEC" starter/algorithmic_baseline.py "${PASSTHROUGH_ARGS[@]}"

# Determine CSV path
CSV_FILE="evaluation/results/Dracula in Barcelona.csv"
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [ "${PASSTHROUGH_ARGS[i]}" == "--name" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        CSV_FILE="evaluation/results/${PASSTHROUGH_ARGS[i+1]}.csv"
    fi
done

# 3. Validate against official scoring constraints
echo ""
echo "[Step 3/3] Validating submission file..."
"$PYTHON_EXEC" evaluation/validate_submission.py "$CSV_FILE"

echo ""
echo "======================================================================"
echo "SUCCESS: Submission file $CSV_FILE reproduced and verified!"
echo "======================================================================"
