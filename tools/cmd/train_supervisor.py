#!/usr/bin/env python3
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _read_tail(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return text[-max_chars:]


def _is_oom(text: str) -> bool:
    hay = text.lower()
    markers = [
        "out of memory",
        "cuda out of memory",
        "mps backend out of memory",
        "resource exhausted",
        "cublas_status_alloc_failed",
        "std::bad_alloc",
    ]
    return any(m in hay for m in markers)


def _extract_batch_size(cmd: str):
    m = re.search(r"(\s-bs\s+)(\d+)", cmd)
    if not m:
        return None
    return int(m.group(2))


def _replace_batch_size(cmd: str, new_bs: int) -> str:
    return re.sub(r"(\s-bs\s+)(\d+)", r"\g<1>%d" % new_bs, cmd, count=1)


def _write_status(path: Path, payload: dict):
    payload = dict(payload)
    payload["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_log(log_path: Path, line: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{_now_iso()}] [supervisor] {line}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="Training shell command to execute")
    parser.add_argument("--cwd", required=True, help="Working directory")
    parser.add_argument("--exp", required=True, help="Experiment name")
    parser.add_argument("--log-file", required=True, help="Combined training/supervisor log file")
    parser.add_argument("--status-file", required=True, help="Training status JSON path")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=20)
    parser.add_argument("--oom-backoff", action="store_true", default=True)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--stop-request-file", default="")
    parser.add_argument("--stop-ack-file", default="")
    args = parser.parse_args()

    base_cmd = args.cmd
    cmd = base_cmd
    log_path = Path(args.log_file)
    status_path = Path(args.status_file)
    stop_request_file = Path(args.stop_request_file) if args.stop_request_file else None
    stop_ack_file = Path(args.stop_ack_file) if args.stop_ack_file else None

    supervisor_pid = os.getpid()
    start_at = _now_iso()

    status = {
        "exp": args.exp,
        "state": "running",
        "running": True,
        "started_at": start_at,
        "supervisor_pid": supervisor_pid,
        "child_pid": None,
        "attempt": 0,
        "max_retries": args.max_retries,
        "last_exit_code": None,
        "last_error_type": "",
        "message": "Training supervisor started",
    }
    _write_status(status_path, status)
    _append_log(log_path, f"Supervisor started pid={supervisor_pid}")

    stop_requested = {"value": False}

    def _handle_signal(signum, _frame):
        stop_requested["value"] = True
        _append_log(log_path, f"Received signal {signum}, will stop after current child exits")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    attempt = 0
    while attempt <= args.max_retries:
        attempt += 1
        status.update(
            {
                "state": "running",
                "running": True,
                "attempt": attempt,
                "message": f"Starting training attempt {attempt}/{args.max_retries + 1}",
            }
        )
        _write_status(status_path, status)
        _append_log(log_path, status["message"])

        with open(log_path, "a", encoding="utf-8") as lf:
            child = subprocess.Popen(
                cmd,
                shell=True,
                cwd=args.cwd,
                stdin=subprocess.DEVNULL,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["child_pid"] = child.pid
            _write_status(status_path, status)
            code = child.wait()

        status["last_exit_code"] = code
        status["child_pid"] = None

        if stop_requested["value"]:
            status.update(
                {
                    "state": "stopped",
                    "running": False,
                    "message": "Training supervisor stopped by signal",
                }
            )
            _write_status(status_path, status)
            _append_log(log_path, "Stopped by signal")
            return 130

        if code == 0:
            stop_ack = bool(stop_ack_file and stop_ack_file.exists())
            status.update(
                {
                    "state": "stopped" if stop_ack else "completed",
                    "running": False,
                    "last_error_type": "",
                    "message": (
                        "Training stopped gracefully at epoch boundary"
                        if stop_ack
                        else "Training completed successfully"
                    ),
                }
            )
            _write_status(status_path, status)
            _append_log(log_path, status["message"])
            return 0

        if stop_request_file and stop_request_file.exists():
            status.update(
                {
                    "state": "stopped",
                    "running": False,
                    "message": "Stop requested by user; training exited",
                }
            )
            _write_status(status_path, status)
            _append_log(log_path, status["message"])
            return 130

        tail = _read_tail(log_path)
        is_oom = _is_oom(tail)
        status["last_error_type"] = "oom" if is_oom else "process_failure"

        if attempt > args.max_retries:
            status.update(
                {
                    "state": "failed",
                    "running": False,
                    "message": f"Training failed after {attempt} attempt(s), exit code={code}",
                }
            )
            _write_status(status_path, status)
            _append_log(log_path, status["message"])
            return code

        if args.oom_backoff and is_oom:
            old_bs = _extract_batch_size(cmd)
            if old_bs is not None and old_bs > args.min_batch_size:
                new_bs = max(args.min_batch_size, old_bs // 2)
                if new_bs != old_bs:
                    cmd = _replace_batch_size(cmd, new_bs)
                    _append_log(
                        log_path,
                        f"OOM detected. Reducing batch size {old_bs} -> {new_bs} before retry.",
                    )

        status.update(
            {
                "message": f"Retrying in {args.retry_delay}s after exit code={code}",
            }
        )
        _write_status(status_path, status)
        _append_log(log_path, status["message"])
        time.sleep(max(args.retry_delay, 0))

    status.update(
        {
            "state": "failed",
            "running": False,
            "message": "Training failed in supervisor",
        }
    )
    _write_status(status_path, status)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
