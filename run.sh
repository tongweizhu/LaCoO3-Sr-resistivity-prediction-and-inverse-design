#!/usr/bin/env bash

# Start the FastAPI predictor and the Rio frontend from one isolated venv.
# Run with `bash run.sh` or, after chmod +x, `./run.sh`.
set -Eeuo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$ROOT_DIR/zhu"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
VENV_DIR="$ROOT_DIR/.venv"
RUN_DIR="$ROOT_DIR/.run"
BACKEND_PORT="${BACKEND_PORT:-5050}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
BACKEND_PID=""
FRONTEND_PID=""

usage() {
    printf 'Usage: %s {start|install|stop|status}\n' "${0##*/}"
    printf '  start   Create/update .venv when needed, then run both services (default).\n'
    printf '  install Create/update .venv and install requirements only.\n'
    printf '  stop    Stop services previously started by this script.\n'
    printf '  status  Show the recorded service process status.\n'
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

select_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        PYTHON_COMMAND="$PYTHON_BIN"
    elif command -v python3.12 >/dev/null 2>&1; then
        PYTHON_COMMAND="$(command -v python3.12)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_COMMAND="$(command -v python3)"
    else
        die 'Python 3.12 was not found. Install it or set PYTHON_BIN to its executable path.'
    fi

    "$PYTHON_COMMAND" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || die "Python 3.12 is required; selected $($PYTHON_COMMAND --version 2>&1)."
}

requirements_hash() {
    "$PYTHON_COMMAND" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$REQUIREMENTS_FILE"
}

prepare_environment() {
    select_python
    [[ -f "$REQUIREMENTS_FILE" ]] || die "Missing $REQUIREMENTS_FILE"

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        printf 'Creating isolated virtual environment at %s\n' "$VENV_DIR"
        "$PYTHON_COMMAND" -m venv "$VENV_DIR" || die 'Unable to create .venv. Ensure the Python venv module is installed.'
    fi

    VENV_PYTHON="$VENV_DIR/bin/python"
    local expected_hash
    local installed_hash=""
    expected_hash="$(requirements_hash)"
    if [[ -f "$VENV_DIR/.requirements.sha256" ]]; then
        installed_hash="$(<"$VENV_DIR/.requirements.sha256")"
    fi

    if [[ "$expected_hash" != "$installed_hash" ]]; then
        printf 'Installing pinned dependencies into %s\n' "$VENV_DIR"
        "$VENV_PYTHON" -m pip install --upgrade pip
        "$VENV_PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"
        printf '%s\n' "$expected_hash" > "$VENV_DIR/.requirements.sha256"
    fi

    if ! "$VENV_PYTHON" -m pip check; then
        printf 'Repairing the environment from %s\n' "$REQUIREMENTS_FILE"
        "$VENV_PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"
        "$VENV_PYTHON" -m pip check
    fi
}

pid_file_for() {
    case "$1" in
        backend) printf '%s/backend.pid\n' "$RUN_DIR" ;;
        frontend) printf '%s/frontend.pid\n' "$RUN_DIR" ;;
        *) die "Unknown service: $1" ;;
    esac
}

recorded_pid() {
    local pid_file
    pid_file="$(pid_file_for "$1")"
    if [[ -f "$pid_file" ]]; then
        tr -d '[:space:]' < "$pid_file"
    fi
}

process_is_running() {
    [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

stop_recorded_service() {
    local service="$1"
    local pid_file
    local pid
    local expected_fragment
    local command_line
    pid_file="$(pid_file_for "$service")"
    pid="$(recorded_pid "$service")"

    case "$service" in
        backend) expected_fragment='uvicorn backend_app:app' ;;
        frontend) expected_fragment='rio run' ;;
        *) die "Unknown service: $service" ;;
    esac

    if process_is_running "$pid"; then
        command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$command_line" != *"$VENV_DIR"* || "$command_line" != *"$expected_fragment"* ]]; then
            printf 'Warning: refusing to stop PID %s; it is no longer the recorded %s process.\n' "$pid" "$service" >&2
            rm -f -- "$pid_file"
            return 0
        fi
        printf 'Stopping %s (PID %s)\n' "$service" "$pid"
        kill "$pid" 2>/dev/null || true
        for _ in {1..20}; do
            process_is_running "$pid" || break
            sleep 0.1
        done
        if process_is_running "$pid"; then
            printf 'Warning: %s PID %s did not stop; leaving it untouched.\n' "$service" "$pid" >&2
        fi
    fi
    rm -f -- "$pid_file"
}

cleanup() {
    local exit_status=$?
    trap - EXIT HUP INT TERM
    if [[ -n "$FRONTEND_PID" ]]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [[ -n "$BACKEND_PID" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    rm -f -- "$(pid_file_for backend)" "$(pid_file_for frontend)"
    exit "$exit_status"
}

port_is_available() {
    "$VENV_PYTHON" -c 'import socket, sys; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(0.2); result = sock.connect_ex(("127.0.0.1", int(sys.argv[1]))); sock.close(); raise SystemExit(0 if result else 1)' "$1"
}

wait_for_backend() {
    local url="http://127.0.0.1:${BACKEND_PORT}/healthz"
    for _ in {1..60}; do
        if "$VENV_PYTHON" -c 'import json, sys, urllib.request; response = urllib.request.urlopen(sys.argv[1], timeout=1); data = json.load(response); raise SystemExit(0 if data.get("status") == "ok" else 1)' "$url" >/dev/null 2>&1; then
            return 0
        fi
        process_is_running "$BACKEND_PID" || return 1
        sleep 0.25
    done
    return 1
}

show_log_tail() {
    local log_file="$1"
    [[ -f "$log_file" ]] || return 0
    printf '\nLast lines of %s:\n' "$log_file" >&2
    tail -n 40 "$log_file" >&2 || true
}

start_services() {
    prepare_environment
    [[ -d "$PROJECT_DIR" ]] || die "Missing Rio project directory: $PROJECT_DIR"
    mkdir -p "$RUN_DIR" "$RUN_DIR/matplotlib"

    if ! port_is_available "$BACKEND_PORT"; then
        die "Port $BACKEND_PORT is already in use. Stop the other API before starting this project."
    fi
    if ! port_is_available "$FRONTEND_PORT"; then
        die "Port $FRONTEND_PORT is already in use. Set FRONTEND_PORT to a free port."
    fi

    trap cleanup EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    printf 'Starting FastAPI at http://127.0.0.1:%s\n' "$BACKEND_PORT"
    (
        cd "$PROJECT_DIR"
        export MPLCONFIGDIR="$RUN_DIR/matplotlib"
        exec "$VENV_PYTHON" -m uvicorn backend_app:app --host 127.0.0.1 --port "$BACKEND_PORT"
    ) > "$RUN_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    printf '%s\n' "$BACKEND_PID" > "$(pid_file_for backend)"

    if ! wait_for_backend; then
        show_log_tail "$RUN_DIR/backend.log"
        die 'FastAPI did not become healthy; the Rio frontend was not started.'
    fi

    printf 'Starting Rio at http://127.0.0.1:%s\n' "$FRONTEND_PORT"
    (
        cd "$PROJECT_DIR"
        export MPLCONFIGDIR="$RUN_DIR/matplotlib"
        export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"
        # Release mode removes Rio's development sidebar, whose root
        # container intentionally uses `overflow: auto`. The workbench itself
        # is a fixed single-screen UI and must not inherit that scroll surface.
        exec "$VENV_PYTHON" -m rio run --release --port "$FRONTEND_PORT"
    ) > "$RUN_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    printf '%s\n' "$FRONTEND_PID" > "$(pid_file_for frontend)"

    sleep 0.5
    if ! process_is_running "$FRONTEND_PID"; then
        show_log_tail "$RUN_DIR/frontend.log"
        die 'Rio exited during startup.'
    fi

    printf 'Both services are running. Press Ctrl-C to stop them.\n'
    set +e
    wait -n "$BACKEND_PID" "$FRONTEND_PID"
    local service_status=$?
    set -e
    if ! process_is_running "$BACKEND_PID"; then
        show_log_tail "$RUN_DIR/backend.log"
    fi
    if ! process_is_running "$FRONTEND_PID"; then
        show_log_tail "$RUN_DIR/frontend.log"
    fi
    exit "$service_status"
}

show_status() {
    local service
    local pid
    for service in backend frontend; do
        pid="$(recorded_pid "$service")"
        if process_is_running "$pid"; then
            printf '%s: running (PID %s)\n' "$service" "$pid"
        else
            printf '%s: stopped\n' "$service"
        fi
    done
}

command="${1:-start}"
case "$command" in
    start)
        start_services
        ;;
    install)
        prepare_environment
        printf 'Environment is ready: %s\n' "$VENV_DIR"
        ;;
    stop)
        mkdir -p "$RUN_DIR"
        stop_recorded_service frontend
        stop_recorded_service backend
        ;;
    status)
        mkdir -p "$RUN_DIR"
        show_status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
