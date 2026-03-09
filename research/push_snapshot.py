#!/usr/bin/env python3
import base64
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_RETRIES = 5


def sh(args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def github_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    return sh(["gh", "auth", "token"])


def parse_origin():
    remote = sh(["git", "remote", "get-url", "origin"])
    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("https://github.com/"):
        path = remote.removeprefix("https://github.com/")
    elif remote.startswith("git@github.com:"):
        path = remote.removeprefix("git@github.com:")
    else:
        raise RuntimeError(f"Unsupported GitHub remote: {remote}")
    owner, repo = path.split("/", 1)
    return owner, repo


def api(method, url, token, payload=None):
    last_error = None
    for attempt in range(1, API_RETRIES + 1):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("Connection", "close")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < API_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            raise RuntimeError(f"GitHub API {method} {url} -> {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as exc:
            last_error = exc
            if attempt >= API_RETRIES:
                break
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"GitHub API {method} {url} failed after {API_RETRIES} retries: {last_error}")


def tracked_files():
    raw = sh(["git", "ls-files", "-z"])
    if not raw:
        return []
    files = [Path(item) for item in raw.split("\0") if item]
    return sorted(files)


def blob_sha(owner, repo, token, relative_path):
    full_path = ROOT / relative_path
    raw = full_path.read_bytes()
    payload = {"content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"}
    url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs"
    blob = api("POST", url, token, payload)
    mode = "100755" if os.access(full_path, os.X_OK) else "100644"
    return {"path": relative_path.as_posix(), "mode": mode, "type": "blob", "sha": blob["sha"]}


def branch_head(owner, repo, branch, token):
    encoded = urllib.parse.quote(branch, safe="")
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{encoded}"
    try:
        return api("GET", url, token)
    except RuntimeError as exc:
        if "-> 404:" in str(exc):
            return None
        raise


def default_branch_head(owner, repo, token):
    repo_info = api("GET", f"https://api.github.com/repos/{owner}/{repo}", token)
    default_branch = repo_info["default_branch"]
    info = api(
        "GET",
        f"https://api.github.com/repos/{owner}/{repo}/branches/{urllib.parse.quote(default_branch, safe='')}",
        token,
    )
    return info


def main():
    token = github_token()
    owner, repo = parse_origin()
    branch = sh(["git", "branch", "--show-current"])
    local_sha = sh(["git", "rev-parse", "--short", "HEAD"])
    local_subject = sh(["git", "log", "-1", "--pretty=%s"])

    entries = [blob_sha(owner, repo, token, path) for path in tracked_files()]

    remote_branch = branch_head(owner, repo, branch, token)
    if remote_branch is None:
        base = default_branch_head(owner, repo, token)
        parent_sha = base["commit"]["sha"]
        ref_exists = False
    else:
        parent_sha = remote_branch["commit"]["sha"]
        ref_exists = True

    tree = api(
        "POST",
        f"https://api.github.com/repos/{owner}/{repo}/git/trees",
        token,
        {"tree": entries},
    )
    commit = api(
        "POST",
        f"https://api.github.com/repos/{owner}/{repo}/git/commits",
        token,
        {
            "message": f"{local_subject}\n\nlocal-commit: {local_sha}",
            "tree": tree["sha"],
            "parents": [parent_sha],
        },
    )

    ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
    if ref_exists:
        api(
            "PATCH",
            f"{ref_url}/heads/{urllib.parse.quote(branch, safe='')}",
            token,
            {"sha": commit["sha"], "force": True},
        )
    else:
        api(
            "POST",
            ref_url,
            token,
            {"ref": f"refs/heads/{branch}", "sha": commit["sha"]},
        )

    print(commit["sha"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
