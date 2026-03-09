#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.py"
RESEARCH_DIR = ROOT / "research"
RUNS_DIR = RESEARCH_DIR / "runs"
STATE_PATH = RESEARCH_DIR / "state.json"
RESULTS_PATH = RESEARCH_DIR / "diff_results.tsv"
BEST_PATH = RESEARCH_DIR / "current_best.md"
PID_PATH = RESEARCH_DIR / "runner.pid"
LOCK_PATH = RESEARCH_DIR / "runner.lock"

CONFIG_KEYS = [
    "USE_DIFF_ATTN",
    "DIFF_ATTN_LAST_LAYERS",
    "DIFF_ATTN_Q2_SOURCE",
    "DIFF_ATTN_LAMBDA_INIT",
    "DIFF_ATTN_WO_INIT",
    "DIFF_ATTN_WO_INIT_SCALE",
    "DIFF_ATTN_LAMBDA_MAX",
    "DIFF_ATTN_Q2_BLEND",
    "DIFF_ATTN_Q2_SCALE",
    "DIFF_ATTN_Y2_NORM",
    "DIFF_ATTN_Y2_CENTER_HEADS",
    "TOTAL_BATCH_SIZE",
]

DEFAULT_BEST_VAL_BPB = 2.078972
RUN_TIMEOUT_SECONDS = 15 * 60
MAX_COMPILE_SECONDS = 10.0
MAX_COMPILE_RETRIES = 1


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock():
    RESEARCH_DIR.mkdir(exist_ok=True)
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = int(LOCK_PATH.read_text().strip())
            except (OSError, ValueError):
                holder = None
            if holder and pid_is_alive(holder):
                raise RuntimeError(f"runner lock is already held by pid {holder}")
            LOCK_PATH.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{os.getpid()}\n")
        return


def format_literal(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        if value == 2**14:
            return "2**14"
        if value == 2**15:
            return "2**15"
        if value == 2**16:
            return "2**16"
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    raise TypeError(f"Unsupported literal: {value!r}")


def eval_literal(source):
    return eval(source, {"__builtins__": {}}, {})


def read_current_config():
    text = TRAIN_PATH.read_text()
    config = {}
    for key in CONFIG_KEYS:
        match = re.search(rf"^{key} = (.+)$", text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not find config key {key} in train.py")
        config[key] = eval_literal(match.group(1).strip())
    return config


def write_config(config):
    text = TRAIN_PATH.read_text()
    for key, value in config.items():
        text, count = re.subn(
            rf"^{key} = .+$",
            f"{key} = {format_literal(value)}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"Failed to update {key} in train.py")
    TRAIN_PATH.write_text(text)


def merge_config(config, fallback):
    merged = dict(fallback)
    merged.update(config)
    return {key: merged[key] for key in CONFIG_KEYS}


def config_fingerprint(config):
    parts = [f"{key}={config[key]}" for key in CONFIG_KEYS]
    return "|".join(parts)


def config_description(config):
    batch_power = int(round(config["TOTAL_BATCH_SIZE"]).bit_length() - 1)
    pieces = [
        f"layers={config['DIFF_ATTN_LAST_LAYERS']}",
        f"lam={config['DIFF_ATTN_LAMBDA_INIT']}",
        f"wo_scale={config['DIFF_ATTN_WO_INIT_SCALE']}",
        f"lam_max={config['DIFF_ATTN_LAMBDA_MAX']}",
        f"blend={config['DIFF_ATTN_Q2_BLEND']}",
        f"q2_scale={config['DIFF_ATTN_Q2_SCALE']}",
        f"y2norm={config['DIFF_ATTN_Y2_NORM']}",
        f"y2center={config['DIFF_ATTN_Y2_CENTER_HEADS']}",
        f"batch=2**{batch_power}",
    ]
    return ", ".join(pieces)


def make_slug(config):
    return (
        f"l{config['DIFF_ATTN_LAST_LAYERS']}"
        f"_lam{str(config['DIFF_ATTN_LAMBDA_INIT']).replace('.', 'p').replace('-', 'm')}"
        f"_wo{str(config['DIFF_ATTN_WO_INIT_SCALE']).replace('.', 'p')}"
        f"_lm{str(config['DIFF_ATTN_LAMBDA_MAX']).replace('.', 'p')}"
        f"_b{str(config['DIFF_ATTN_Q2_BLEND']).replace('.', 'p')}"
        f"_qs{str(config['DIFF_ATTN_Q2_SCALE']).replace('.', 'p')}"
        f"_n{int(config['DIFF_ATTN_Y2_NORM'])}"
        f"_c{int(config['DIFF_ATTN_Y2_CENTER_HEADS'])}"
        f"_tb{config['TOTAL_BATCH_SIZE']}"
    )


def ensure_research_files(best_config):
    RESEARCH_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "timestamp\tcommit\tval_bpb\tmemory_gb\tnum_steps\tstatus\tdescription\tlog_file\n"
        )
    if not STATE_PATH.exists():
        deadline = datetime.now() + timedelta(hours=10)
        state = {
            "started_at": now_iso(),
            "deadline_at": deadline.isoformat(timespec="seconds"),
            "run_count": 0,
            "best": {
                "val_bpb": DEFAULT_BEST_VAL_BPB,
                "memory_gb": 11.3,
                "num_steps": 144,
                "config": best_config,
                "description": config_description(best_config),
            },
            "seen": [],
            "last_result": None,
        }
        STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
    write_best_summary(load_state())


def load_state():
    return json.loads(STATE_PATH.read_text())


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def write_best_summary(state):
    best = state["best"]
    lines = [
        "# Current Best Diff-Attn Run",
        "",
        f"- `val_bpb = {best['val_bpb']:.6f}`",
        f"- `memory_gb = {best['memory_gb']:.1f}`",
        f"- `num_steps = {best['num_steps']}`",
        f"- `description = {best['description']}`",
        "",
        "## Config",
        "",
    ]
    for key in CONFIG_KEYS:
        lines.append(f"- `{key} = {best['config'][key]}`")
    BEST_PATH.write_text("\n".join(lines) + "\n")


def append_result(timestamp, commit_hash, metrics, status, description, log_file):
    row = [
        timestamp,
        commit_hash,
        f"{metrics['val_bpb']:.6f}" if metrics["val_bpb"] is not None else "0.000000",
        f"{metrics['memory_gb']:.1f}" if metrics["memory_gb"] is not None else "0.0",
        str(metrics["num_steps"]) if metrics["num_steps"] is not None else "0",
        status,
        description,
        log_file,
    ]
    with RESULTS_PATH.open("a") as handle:
        handle.write("\t".join(row) + "\n")


def git(args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def git_commit_all(message):
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    if not status.stdout.strip():
        return git(["rev-parse", "--short", "HEAD"]).stdout.strip()
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    return git(["rev-parse", "--short", "HEAD"]).stdout.strip()


def git_push():
    try:
        subprocess.run(["uv", "run", "python", "research/push_snapshot.py"], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] snapshot push failed: {exc}", flush=True)


def parse_metrics(log_text):
    metrics = {"val_bpb": None, "memory_gb": None, "num_steps": None, "compile_seconds": None}
    match = re.search(r"^Model compiled in ([0-9.]+)s$", log_text, re.MULTILINE)
    if match:
        metrics["compile_seconds"] = float(match.group(1))
    match = re.search(r"^val_bpb:\s+([0-9.]+)$", log_text, re.MULTILINE)
    if match:
        metrics["val_bpb"] = float(match.group(1))
    match = re.search(r"^peak_vram_mb:\s+([0-9.]+)$", log_text, re.MULTILINE)
    if match:
        metrics["memory_gb"] = float(match.group(1)) / 1024.0
    match = re.search(r"^num_steps:\s+([0-9]+)$", log_text, re.MULTILINE)
    if match:
        metrics["num_steps"] = int(match.group(1))
    return metrics


def run_experiment_once(config, attempt_idx):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = make_slug(config)
    suffix = "" if attempt_idx == 0 else f"_retry{attempt_idx}"
    log_path = RUNS_DIR / f"{timestamp}_{slug}{suffix}.log"
    write_config(config)
    cmd = ["uv", "run", "train.py"]
    started_at = now_iso()
    with log_path.open("w") as handle:
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT_SECONDS,
                check=False,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            handle.write("\nTIMEOUT\n")
            exit_code = 124
    log_text = log_path.read_text(errors="replace")
    metrics = parse_metrics(log_text)
    crashed = exit_code != 0 or metrics["val_bpb"] is None
    status = "crash" if crashed else "ok"
    return {
        "timestamp": timestamp,
        "started_at": started_at,
        "exit_code": exit_code,
        "log_path": str(log_path.relative_to(ROOT)),
        "metrics": metrics,
        "status": status,
        "description": config_description(config),
    }


def run_experiment(config):
    for attempt_idx in range(MAX_COMPILE_RETRIES + 1):
        result = run_experiment_once(config, attempt_idx)
        compile_seconds = result["metrics"]["compile_seconds"]
        if compile_seconds is None or compile_seconds <= MAX_COMPILE_SECONDS:
            return result
        if attempt_idx >= MAX_COMPILE_RETRIES:
            return result
        print(
            f"[runner] retrying {result['description']} because compile took {compile_seconds:.1f}s",
            flush=True,
        )
    raise RuntimeError("run_experiment retry loop fell through unexpectedly")


def curated_candidates(best):
    base = dict(best)
    candidates = [
        {**base, "DIFF_ATTN_LAMBDA_INIT": -2.25},
        {**base, "DIFF_ATTN_LAMBDA_INIT": -1.75},
        {**base, "DIFF_ATTN_WO_INIT_SCALE": 0.08},
        {**base, "DIFF_ATTN_WO_INIT_SCALE": 0.12},
        {**base, "DIFF_ATTN_LAMBDA_MAX": 0.75},
        {**base, "DIFF_ATTN_LAMBDA_MAX": 0.5},
        {**base, "TOTAL_BATCH_SIZE": 2**14},
        {**base, "DIFF_ATTN_Q2_BLEND": 0.9},
        {**base, "DIFF_ATTN_Q2_BLEND": 0.85},
        {**base, "DIFF_ATTN_Q2_SCALE": 0.85},
        {**base, "DIFF_ATTN_Q2_SCALE": 1.15},
        {**base, "DIFF_ATTN_Y2_NORM": True},
        {**base, "DIFF_ATTN_Y2_CENTER_HEADS": True},
        {**base, "DIFF_ATTN_Y2_CENTER_HEADS": True, "DIFF_ATTN_Y2_NORM": True},
        {**base, "DIFF_ATTN_Q2_SCALE": 0.85, "DIFF_ATTN_Y2_CENTER_HEADS": True},
        {**base, "DIFF_ATTN_Q2_BLEND": 0.9, "DIFF_ATTN_Y2_NORM": True},
        {**base, "DIFF_ATTN_LAST_LAYERS": 2},
        {**base, "DIFF_ATTN_LAST_LAYERS": 2, "DIFF_ATTN_Q2_BLEND": 0.9},
        {**base, "DIFF_ATTN_LAST_LAYERS": 2, "DIFF_ATTN_LAMBDA_MAX": 0.75},
        {**base, "DIFF_ATTN_LAST_LAYERS": 2, "DIFF_ATTN_Y2_NORM": True},
        {**base, "DIFF_ATTN_LAST_LAYERS": 1, "TOTAL_BATCH_SIZE": 2**14, "DIFF_ATTN_Q2_BLEND": 0.9},
        {**base, "DIFF_ATTN_LAST_LAYERS": 1, "TOTAL_BATCH_SIZE": 2**14, "DIFF_ATTN_Y2_NORM": True},
    ]
    return dedupe_candidates(candidates)


def mutate_candidate(best, rng):
    candidate = dict(best)
    knobs = rng.sample(
        [
            "DIFF_ATTN_LAST_LAYERS",
            "DIFF_ATTN_LAMBDA_INIT",
            "DIFF_ATTN_WO_INIT_SCALE",
            "DIFF_ATTN_LAMBDA_MAX",
            "DIFF_ATTN_Q2_BLEND",
            "DIFF_ATTN_Q2_SCALE",
            "DIFF_ATTN_Y2_NORM",
            "DIFF_ATTN_Y2_CENTER_HEADS",
            "TOTAL_BATCH_SIZE",
        ],
        k=rng.randint(1, 3),
    )
    for knob in knobs:
        if knob == "DIFF_ATTN_LAST_LAYERS":
            candidate[knob] = rng.choice([1, 2])
        elif knob == "DIFF_ATTN_LAMBDA_INIT":
            candidate[knob] = rng.choice([-2.5, -2.25, -2.0, -1.85, -1.75, -1.6])
        elif knob == "DIFF_ATTN_WO_INIT_SCALE":
            candidate[knob] = rng.choice([0.05, 0.08, 0.1, 0.12, 0.15])
        elif knob == "DIFF_ATTN_LAMBDA_MAX":
            candidate[knob] = rng.choice([0.5, 0.65, 0.75, 0.9, 1.0])
        elif knob == "DIFF_ATTN_Q2_BLEND":
            candidate[knob] = rng.choice([1.0, 0.95, 0.9, 0.85, 0.75])
        elif knob == "DIFF_ATTN_Q2_SCALE":
            candidate[knob] = rng.choice([0.75, 0.85, 1.0, 1.15, 1.25])
        elif knob == "DIFF_ATTN_Y2_NORM":
            candidate[knob] = rng.choice([False, True])
        elif knob == "DIFF_ATTN_Y2_CENTER_HEADS":
            candidate[knob] = rng.choice([False, True])
        elif knob == "TOTAL_BATCH_SIZE":
            candidate[knob] = rng.choice([2**14, 2**15])
    return candidate


def dedupe_candidates(candidates):
    seen = set()
    unique = []
    for candidate in candidates:
        fingerprint = config_fingerprint(candidate)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(candidate)
    return unique


def next_candidate(state, rng):
    best = state["best"]["config"]
    seen = set(state["seen"])
    for candidate in curated_candidates(best):
        fingerprint = config_fingerprint(candidate)
        if fingerprint not in seen:
            return candidate
    for _ in range(256):
        candidate = mutate_candidate(best, rng)
        fingerprint = config_fingerprint(candidate)
        if fingerprint not in seen:
            return candidate
    raise RuntimeError("Candidate generator exhausted unexpectedly")


def deadline_reached(state):
    deadline = datetime.fromisoformat(state["deadline_at"])
    return datetime.now() >= deadline


def main():
    parser = argparse.ArgumentParser(description="Autonomous diff-attention research loop")
    parser.add_argument("--hours", type=float, default=10.0, help="Research budget in wall-clock hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for experiment generation")
    args = parser.parse_args()

    current_config = read_current_config()
    ensure_research_files(current_config)
    acquire_lock()
    state = load_state()
    state["best"]["config"] = merge_config(state["best"]["config"], current_config)
    if "deadline_at" not in state or not state["deadline_at"]:
        state["deadline_at"] = (datetime.now() + timedelta(hours=args.hours)).isoformat(timespec="seconds")
    save_state(state)

    PID_PATH.write_text(str(os.getpid()) + "\n")
    rng = random.Random(args.seed)

    try:
        while not deadline_reached(state):
            candidate = next_candidate(state, rng)
            fingerprint = config_fingerprint(candidate)
            print(f"[runner] trying {config_description(candidate)}", flush=True)
            result = run_experiment(candidate)
            metrics = result["metrics"]
            improved = (
                result["status"] == "ok"
                and metrics["val_bpb"] is not None
                and metrics["val_bpb"] < state["best"]["val_bpb"]
            )
            if improved:
                state["best"] = {
                    "val_bpb": metrics["val_bpb"],
                    "memory_gb": metrics["memory_gb"] or 0.0,
                    "num_steps": metrics["num_steps"] or 0,
                    "config": candidate,
                    "description": result["description"],
                }
                keep_status = "keep"
            else:
                write_config(state["best"]["config"])
                keep_status = "discard" if result["status"] == "ok" else "crash"

            state["run_count"] += 1
            state["seen"].append(fingerprint)
            state["last_result"] = {
                "timestamp": result["timestamp"],
                "status": keep_status,
                "description": result["description"],
                "log_path": result["log_path"],
                "metrics": metrics,
            }
            save_state(state)
            write_best_summary(state)
            append_result(
                result["timestamp"],
                "pending",
                metrics,
                keep_status,
                result["description"],
                result["log_path"],
            )

            commit_message = (
                f"runner: {keep_status} {result['description']}"
                if keep_status != "keep"
                else f"runner: keep {result['description']} val_bpb={metrics['val_bpb']:.6f}"
            )
            commit_hash = git_commit_all(commit_message)

            lines = RESULTS_PATH.read_text().splitlines()
            if len(lines) >= 2 and lines[-1].startswith(result["timestamp"] + "\t"):
                parts = lines[-1].split("\t")
                parts[1] = commit_hash
                lines[-1] = "\t".join(parts)
                RESULTS_PATH.write_text("\n".join(lines) + "\n")
                commit_hash = git_commit_all(f"runner: stamp result {result['timestamp']}")

            git_push()
            print(
                f"[runner] {keep_status} {result['description']} "
                f"val_bpb={metrics['val_bpb']} best={state['best']['val_bpb']:.6f}",
                flush=True,
            )
            state = load_state()
    finally:
        try:
            write_config(load_state()["best"]["config"])
        except Exception:
            pass
        LOCK_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
