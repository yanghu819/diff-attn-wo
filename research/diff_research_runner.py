#!/usr/bin/env python3
import argparse
import ast
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
    "DIFF_ATTN_LAYER_MASK",
    "DIFF_ATTN_Q2_SOURCE",
    "DIFF_ATTN_LAMBDA_INIT",
    "DIFF_ATTN_WO_INIT",
    "DIFF_ATTN_WO_INIT_SCALE",
    "DIFF_ATTN_LAMBDA_MAX",
    "DIFF_ATTN_LAMBDA_WEIGHT_INIT",
    "DIFF_ATTN_LAMBDA_WEIGHT_INIT_SCALE",
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
SIGNIFICANT_SINGLE_LAYER_GAIN = 0.01
PRIMARY_LAM_SWEEP = [-2.25, -2.0, -1.75]
PRIMARY_LAMBDA_MAX_SWEEP = [0.75, 0.9, 0.5]
PRIMARY_Q2_SCALE_SWEEP = [0.85, 1.15, 0.75, 1.25]
DEFAULT_TOTAL_BATCH_SIZE = 2**15


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def parse_scalar_literal(raw):
    if raw in {"True", "False"}:
        return raw == "True"
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


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


def canonicalize_layer_mask(value):
    if value in ("", None):
        return ""
    if isinstance(value, (tuple, list, set)):
        parts = [str(int(item)) for item in value]
    else:
        parts = [piece.strip() for piece in str(value).split(",")]
    cleaned = []
    seen = set()
    for piece in parts:
        if not piece:
            continue
        normalized = str(int(piece))
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return ",".join(cleaned)


def canonicalize_config(config):
    normalized = dict(config)
    normalized["DIFF_ATTN_LAYER_MASK"] = canonicalize_layer_mask(normalized.get("DIFF_ATTN_LAYER_MASK", ""))
    if normalized["DIFF_ATTN_LAYER_MASK"]:
        normalized["DIFF_ATTN_LAST_LAYERS"] = 0
    return normalized


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
    merged = {key: merged[key] for key in CONFIG_KEYS}
    return canonicalize_config(merged)


def ordered_unique(items):
    output = []
    seen = set()
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def set_diff_layers(config, last_layers=None, layer_mask=None):
    updated = dict(config)
    if layer_mask is not None:
        updated["DIFF_ATTN_LAYER_MASK"] = canonicalize_layer_mask(layer_mask)
        if updated["DIFF_ATTN_LAYER_MASK"]:
            updated["DIFF_ATTN_LAST_LAYERS"] = 0
    if last_layers is not None:
        updated["DIFF_ATTN_LAST_LAYERS"] = last_layers
        if last_layers > 0:
            updated["DIFF_ATTN_LAYER_MASK"] = ""
    return canonicalize_config(updated)


def config_fingerprint(config):
    config = canonicalize_config(config)
    parts = [f"{key}={config[key]}" for key in CONFIG_KEYS]
    return "|".join(parts)


def parse_fingerprint(fingerprint):
    config = {}
    for piece in fingerprint.split("|"):
        key, value = piece.split("=", 1)
        if key == "DIFF_ATTN_LAYER_MASK":
            config[key] = canonicalize_layer_mask(parse_scalar_literal(value))
        else:
            config[key] = parse_scalar_literal(value)
    return canonicalize_config(config)


def config_description(config):
    batch_power = int(round(config["TOTAL_BATCH_SIZE"]).bit_length() - 1)
    pieces = [
        f"layers={config['DIFF_ATTN_LAST_LAYERS']}",
        f"mask={config['DIFF_ATTN_LAYER_MASK'] or '<last>'}",
        f"lam={config['DIFF_ATTN_LAMBDA_INIT']}",
        f"wo_scale={config['DIFF_ATTN_WO_INIT_SCALE']}",
        f"lam_max={config['DIFF_ATTN_LAMBDA_MAX']}",
        f"lam_w_init={config['DIFF_ATTN_LAMBDA_WEIGHT_INIT']}",
        f"lam_w_scale={config['DIFF_ATTN_LAMBDA_WEIGHT_INIT_SCALE']}",
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
        f"_mask{config['DIFF_ATTN_LAYER_MASK'].replace(',', '-') if config['DIFF_ATTN_LAYER_MASK'] else 'last'}"
        f"_lam{str(config['DIFF_ATTN_LAMBDA_INIT']).replace('.', 'p').replace('-', 'm')}"
        f"_wo{str(config['DIFF_ATTN_WO_INIT_SCALE']).replace('.', 'p')}"
        f"_lm{str(config['DIFF_ATTN_LAMBDA_MAX']).replace('.', 'p')}"
        f"_lwi{config['DIFF_ATTN_LAMBDA_WEIGHT_INIT']}"
        f"_lws{str(config['DIFF_ATTN_LAMBDA_WEIGHT_INIT_SCALE']).replace('.', 'p')}"
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


def count_diff_layers(config):
    mask = str(config.get("DIFF_ATTN_LAYER_MASK", "")).strip()
    if mask:
        return len([piece for piece in mask.split(",") if piece.strip()])
    return int(config["DIFF_ATTN_LAST_LAYERS"])


def is_expensive(config):
    return config["TOTAL_BATCH_SIZE"] == 2**14 or count_diff_layers(config) >= 2


def expensive_budget_available(state):
    seen_configs = [parse_fingerprint(item) for item in state.get("seen", [])]
    expensive_count = sum(1 for config in seen_configs if is_expensive(config))
    total_count = len(seen_configs)
    return (expensive_count + 1) * 5 <= max(5, total_count + 1)


def ensure_policy_state(state):
    changed = False
    policy = state.setdefault("policy", {})
    best = state["best"]
    best["config"] = merge_config(best["config"], read_current_config())
    best["description"] = config_description(best["config"])
    normalized_seen = []
    for fingerprint in state.get("seen", []):
        normalized = config_fingerprint(merge_config(parse_fingerprint(fingerprint), best["config"]))
        normalized_seen.append(normalized)
    deduped_seen = ordered_unique(normalized_seen)
    if deduped_seen != state.get("seen", []):
        state["seen"] = deduped_seen
        changed = True
    best_fingerprint = config_fingerprint(best["config"])
    anchor_val = policy.get("single_layer_anchor_val_bpb")
    anchor_fingerprint = policy.get("single_layer_anchor_fingerprint")
    if anchor_val is None or anchor_fingerprint is None:
        policy["single_layer_anchor_val_bpb"] = best["val_bpb"]
        policy["single_layer_anchor_fingerprint"] = best_fingerprint
        changed = True
    elif anchor_val - best["val_bpb"] >= SIGNIFICANT_SINGLE_LAYER_GAIN and anchor_fingerprint != best_fingerprint:
        policy["single_layer_anchor_val_bpb"] = best["val_bpb"]
        policy["single_layer_anchor_fingerprint"] = best_fingerprint
        changed = True
    return changed


def single_layer_stalled(state):
    policy = state.setdefault("policy", {})
    anchor_val = policy.get("single_layer_anchor_val_bpb", state["best"]["val_bpb"])
    return (anchor_val - state["best"]["val_bpb"]) < SIGNIFICANT_SINGLE_LAYER_GAIN


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
        value = best["config"][key]
        if key == "DIFF_ATTN_LAYER_MASK" and value == "":
            value = "<last>"
        lines.append(f"- `{key} = {value}`")
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


def read_depth():
    text = TRAIN_PATH.read_text()
    match = re.search(r"^DEPTH = (.+)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find DEPTH in train.py")
    return int(eval_literal(match.group(1).strip()))


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


def primary_candidates(best):
    base = dict(best)
    candidates = []
    for lam in ordered_unique([value for value in PRIMARY_LAM_SWEEP if value != base["DIFF_ATTN_LAMBDA_INIT"]]):
        candidates.append({**base, "DIFF_ATTN_LAMBDA_INIT": lam})
    for lambda_max in PRIMARY_LAMBDA_MAX_SWEEP:
        if lambda_max == base["DIFF_ATTN_LAMBDA_MAX"]:
            continue
        candidates.append({**base, "DIFF_ATTN_LAMBDA_MAX": lambda_max})
    for q2_scale in PRIMARY_Q2_SCALE_SWEEP:
        if q2_scale == base["DIFF_ATTN_Q2_SCALE"]:
            continue
        candidates.append({**base, "DIFF_ATTN_Q2_SCALE": q2_scale})
    if not base["DIFF_ATTN_Y2_CENTER_HEADS"]:
        candidates.append({**base, "DIFF_ATTN_Y2_CENTER_HEADS": True})
    if not base["DIFF_ATTN_Y2_NORM"]:
        candidates.append({**base, "DIFF_ATTN_Y2_NORM": True})
    if not (base["DIFF_ATTN_Y2_CENTER_HEADS"] and base["DIFF_ATTN_Y2_NORM"]):
        candidates.append({**base, "DIFF_ATTN_Y2_CENTER_HEADS": True, "DIFF_ATTN_Y2_NORM": True})
    return dedupe_candidates(candidates)


def mild_q2_scale_candidates(best):
    current = float(best["DIFF_ATTN_Q2_SCALE"])
    values = []
    for factor in [0.95, 1.05, 0.9, 1.1]:
        value = round(current * factor, 3)
        value = max(0.5, min(1.5, value))
        values.append(value)
    return ordered_unique(value for value in values if value != current)


def secondary_candidates(best):
    base = dict(best)
    candidates = []
    for q2_scale in mild_q2_scale_candidates(base):
        candidates.append({**base, "DIFF_ATTN_Q2_SCALE": q2_scale})

    capped_lambda_max = min(float(base["DIFF_ATTN_LAMBDA_MAX"]), 0.75)
    for init_scale in [0.05, 0.1]:
        candidates.append(
            {
                **base,
                "DIFF_ATTN_LAMBDA_MAX": capped_lambda_max,
                "DIFF_ATTN_LAMBDA_WEIGHT_INIT": "small_random",
                "DIFF_ATTN_LAMBDA_WEIGHT_INIT_SCALE": init_scale,
            }
        )

    depth = read_depth()
    sparse_masks = []
    if depth >= 3:
        sparse_masks.append(str(depth - 2))
    for layer_mask in ordered_unique(mask for mask in sparse_masks if mask != base["DIFF_ATTN_LAYER_MASK"]):
        candidates.append(set_diff_layers(base, layer_mask=layer_mask))
    return dedupe_candidates(candidates)


def expensive_candidates(best):
    base = dict(best)
    candidates = [
        set_diff_layers(base, last_layers=2),
        {**base, "TOTAL_BATCH_SIZE": 2**14},
        {**set_diff_layers(base, last_layers=2), "TOTAL_BATCH_SIZE": 2**14},
    ]
    depth = read_depth()
    if depth >= 4:
        candidates.append(set_diff_layers(base, layer_mask="1,3"))
    return dedupe_candidates(candidates)


def mutate_candidate(best, rng, allow_expensive):
    candidate = dict(best)
    knobs = [
        "DIFF_ATTN_LAMBDA_INIT",
        "DIFF_ATTN_WO_INIT_SCALE",
        "DIFF_ATTN_LAMBDA_MAX",
        "DIFF_ATTN_Q2_BLEND",
        "DIFF_ATTN_Q2_SCALE",
        "DIFF_ATTN_Y2_NORM",
        "DIFF_ATTN_Y2_CENTER_HEADS",
        "DIFF_ATTN_LAMBDA_WEIGHT_INIT",
    ]
    if allow_expensive:
        knobs.extend(["DIFF_ATTN_LAST_LAYERS", "TOTAL_BATCH_SIZE", "DIFF_ATTN_LAYER_MASK"])
    knob_count = rng.randint(1, 3)
    knobs = rng.sample(knobs, k=min(knob_count, len(knobs)))
    for knob in knobs:
        if knob == "DIFF_ATTN_LAST_LAYERS":
            candidate = set_diff_layers(candidate, last_layers=rng.choice([1, 2]))
        elif knob == "DIFF_ATTN_LAYER_MASK":
            depth = read_depth()
            options = [""]
            if depth >= 3:
                options.append(str(depth - 2))
            if depth >= 4:
                options.append("1,3")
            candidate = set_diff_layers(candidate, layer_mask=rng.choice(options))
        elif knob == "DIFF_ATTN_LAMBDA_INIT":
            candidate[knob] = rng.choice([-2.5, -2.25, -2.0, -1.85, -1.75, -1.6])
        elif knob == "DIFF_ATTN_WO_INIT_SCALE":
            candidate[knob] = rng.choice([0.05, 0.08, 0.1, 0.12, 0.15])
        elif knob == "DIFF_ATTN_LAMBDA_MAX":
            candidate[knob] = rng.choice([0.5, 0.65, 0.75, 0.9, 1.0])
        elif knob == "DIFF_ATTN_LAMBDA_WEIGHT_INIT":
            if rng.choice([False, True]):
                candidate["DIFF_ATTN_LAMBDA_WEIGHT_INIT"] = "small_random"
                candidate["DIFF_ATTN_LAMBDA_WEIGHT_INIT_SCALE"] = rng.choice([0.03, 0.05, 0.1])
                candidate["DIFF_ATTN_LAMBDA_MAX"] = min(candidate["DIFF_ATTN_LAMBDA_MAX"], 0.75)
            else:
                candidate["DIFF_ATTN_LAMBDA_WEIGHT_INIT"] = "zero"
        elif knob == "DIFF_ATTN_Q2_BLEND":
            candidate[knob] = rng.choice([1.0, 0.95, 0.9, 0.85, 0.75])
        elif knob == "DIFF_ATTN_Q2_SCALE":
            candidate[knob] = rng.choice([0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25])
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


def first_unseen(candidates, seen):
    for candidate in candidates:
        fingerprint = config_fingerprint(candidate)
        if fingerprint not in seen:
            return candidate
    return None


def next_candidate(state, rng):
    ensure_policy_state(state)
    best = state["best"]["config"]
    seen = set(state["seen"])
    for candidate_group in [primary_candidates(best), secondary_candidates(best)]:
        candidate = first_unseen(candidate_group, seen)
        if candidate is not None:
            return candidate

    stalled = single_layer_stalled(state)
    if stalled:
        candidate = first_unseen(expensive_candidates(best), seen)
        if candidate is not None and expensive_budget_available(state):
            return candidate

    for _ in range(256):
        candidate = mutate_candidate(best, rng, allow_expensive=stalled and expensive_budget_available(state))
        fingerprint = config_fingerprint(candidate)
        if not stalled and is_expensive(candidate):
            continue
        if stalled and is_expensive(candidate) and not expensive_budget_available(state):
            continue
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
    state["best"]["description"] = config_description(state["best"]["config"])
    if "deadline_at" not in state or not state["deadline_at"]:
        state["deadline_at"] = (datetime.now() + timedelta(hours=args.hours)).isoformat(timespec="seconds")
    ensure_policy_state(state)
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
