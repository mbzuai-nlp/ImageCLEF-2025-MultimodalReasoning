SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30010}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
SGLANG_API_BASE="http://$SGLANG_HOST:$SGLANG_PORT/v1"
SGLANG_MODEL_NAME="placeholder"
SGLANG_SERVER_PID=""
SGLANG_VENV_DIR="${SGLANG_VENV_DIR:-${VENV_DIR:-}}"

cleanup_sglang_server() {
    if [ -n "${SGLANG_SERVER_PID:-}" ] && kill -0 "$SGLANG_SERVER_PID" 2>/dev/null; then
        echo "Stopping SGLang server (PID: $SGLANG_SERVER_PID)..."
        kill "$SGLANG_SERVER_PID" 2>/dev/null || true
        wait "$SGLANG_SERVER_PID" 2>/dev/null || true
    fi
    SGLANG_SERVER_PID=""
}

wait_for_sglang_server() {
    local log_file="$1"
    local retries=120

    for ((attempt=1; attempt<=retries; attempt++)); do
        if curl -fsS "http://$SGLANG_HOST:$SGLANG_PORT/v1/models" >/dev/null 2>&1; then
            echo "SGLang server is ready."
            return 0
        fi

        if [ -n "${SGLANG_SERVER_PID:-}" ] && ! kill -0 "$SGLANG_SERVER_PID" 2>/dev/null; then
            echo "SGLang server exited before becoming ready. Recent logs:"
            tail -n 50 "$log_file" || true
            return 1
        fi

        sleep 2
    done

    echo "Timed out waiting for the SGLang server. Recent logs:"
    tail -n 50 "$log_file" || true
    return 1
}

start_sglang_server() {
    local model_path="$1"
    local served_model_name="$2"
    local log_file="$3"
    shift 3
    local extra_args=("$@")

    cleanup_sglang_server

    echo "Starting SGLang server for $served_model_name ..."
    if [ -z "${SGLANG_VENV_DIR:-}" ]; then
        echo "Error: SGLANG_VENV_DIR is not set."
        return 1
    fi

    python -m sglang.launch_server \
        --model-path "$model_path" \
        --served-model-name "$served_model_name" \
        --host "$SGLANG_HOST" \
        --port "$SGLANG_PORT" \
        --tp-size "$SGLANG_TP_SIZE" \
        --trust-remote-code \
        --log-level "$SGLANG_LOG_LEVEL" \
        "${extra_args[@]}" \
        >"$log_file" 2>&1 &

    SGLANG_SERVER_PID=$!
    wait_for_sglang_server "$log_file"
}
