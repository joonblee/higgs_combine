#!/usr/bin/env python3
"""Audit the organised production, remove one-off tools and optionally push.

Install this utility as tools/finalise_nisomuon.py at the repository root.
Default: audit/plan only. --apply removes the exact listed obsolete files.
--apply --push also commits selected NIsoMuon files to higgs_combine/main.
ROOT/PNG/cache files, unblind paths and unrelated staged changes are excluded.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SSH = "git@github.com:joonblee/higgs_combine.git"
REMOVE = (
    "diagnose_zero_dy.py", "repair_blind_poi.sh", "refresh_m12_impacts.py",
    "organise_nisomuon.py", "ORGANISE_OUTPUTS.md",
    "tests/test_m12_refresh.py", "tests/test_organise_nisomuon.py",
    "template/datacard_template.txt", "template/datacard.txt.bak",
)
ROOT_LOGS = ("combine_logger.out", "fit_status_blind.txt")
KEEP = (
    "limit_workflow.py", "combine_review.py", "review.sh",
    "audit_bias_intervals.py", "repair_bias_intervals.py", "validate_bias_repair.py",
    "systematics_NPS26009.yml", "python/plotLimits.py", "python/plotting.py",
)
TEXT_RESULTS = {".json", ".pdf", ".txt", ".csv", ".log", ".out", ".py", ".sh", ".md", ".yml", ".yaml"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def git(repo, *args, binary=False):
    out = subprocess.check_output(["git", "-C", str(repo), *map(str, args)], stderr=subprocess.PIPE)
    return out if binary else out.decode().strip()


def paths(repo, *args):
    return set(x.decode() for x in git(repo, *args, binary=True).split(b"\0") if x)


def load(path):
    return json.loads(path.read_text())


def audit_references(base, record, digest_for=None):
    """Check live fingerprints as well as the outer file inventory.

    Original JSONs are historical records; their original absolute paths and
    hashes are intentionally retained. All live input hashes must resolve.
    """
    digest_for = digest_for or sha
    prefix = record.get("copied_from", str(base)).rstrip("/") + "/"
    for p in (base/"review_outputs").rglob("*.json"):
        if "original_json" in p.relative_to(base).parts:
            continue
        value = load(p)
        if not isinstance(value, dict) or not isinstance(value.get("input_sha256"), dict):
            continue
        for name, expected in value["input_sha256"].items():
            # Only inputs within this production tree are relocated/audited here.
            if name.startswith(prefix):
                path = base/name[len(prefix):]
            elif name.startswith(str(base) + "/"):
                path = Path(name)
            else:
                continue
            require(path.is_file() and digest_for(path) == expected,
                    f"Stale/missing input fingerprint in {p}: {name}")
    final = base/"limit_outputs/final_impacts_manifest.json"
    if final.is_file():
        entries = load(final)["b_only_impacts"]
        expected_sets = {(t, m) for t in ("Run2", "Run3", "Run2Run3")
                         for m in (12,15,20,25,30,35,40,45,50,55,60,65,70)}
        require(len(entries) == 39 and {(e['target'], e['mass']) for e in entries} == expected_sets,
                "Final impact manifest does not contain exactly 39 distinct sets")
        for e in entries:
            internal = base/f"limit_outputs/alpha/blind/{e['target']}/impacts_{e['target']}_alpha_M{e['mass']}_internal.json"
            require(internal.is_file() and digest_for(internal) == e["internal_json_sha256"],
                    f"Stale final-impact fingerprint: {e['target']} M{e['mass']}")


def audit(base):
    for name in KEEP:
        require((base/name).is_file(), f"Required file is missing: {name}")
    record = load(base/"review_outputs/organisation_manifest.json")
    for number, (name, digest) in enumerate(record["files_sha256"].items(), 1):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] in ("limit_outputs", "review_outputs"),
                f"Invalid saved inventory path: {name}")
        path = base/relative
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                f"Organised scientific result changed/missing: {name}")
        if number % 100 == 0:
            print(f"[AUDIT] {number} saved files checked", flush=True)
    for target in ("Run2", "Run3", "Run2Run3"):
        folder = base/f"limit_outputs/alpha/blind/{target}"
        for mass in (12,15,20,25,30,35,40,45,50,55,60,65,70):
            for suffix in (".pdf", ".json", "_internal.json"):
                require((folder/f"impacts_{target}_alpha_M{mass}{suffix}").is_file(),
                        f"Missing final impact: {target} M{mass} {suffix}")
        for suffix in (".pdf", ".json", "_internal.json"):
            require((folder/f"limits_{target}_alpha{suffix}").is_file(),
                    f"Missing final limit: {target} {suffix}")
    audit_references(base, record, lambda p: record["files_sha256"].get(str(p.relative_to(base))) or sha(p))
    print("[PASS] Saved scientific file hashes, live dependency fingerprints and 39 final impact sets are intact")
    return record


def save_unique(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory/source.name
    if dest.exists() and sha(dest) != sha(source):
        dest = directory/(source.stem + "_" + sha(source)[:12] + source.suffix)
    if not dest.exists():
        shutil.copy2(source, dest)
    require(sha(dest) == sha(source), f"Archive copy mismatch: {dest}")
    return dest


def clean(base, repo, record):
    removed, archived, log_moves = [], [], []
    head = git(repo, "rev-parse", "HEAD")
    for name in REMOVE:
        path = base/name
        if not path.exists():
            continue
        require(path.is_file() and not path.is_symlink(), f"Unexpected non-regular file: {path}")
        relative = path.relative_to(repo)
        try:
            committed = git(repo, "show", f"HEAD:{relative}", binary=True)
        except subprocess.CalledProcessError:
            committed = None
        # Clean tracked files remain recoverable at source_commit. Preserve a
        # unique uncommitted version before removing its working-directory copy.
        if committed is None or hashlib.sha256(committed).hexdigest() != sha(path):
            dest = save_unique(path, base/"review_outputs/provenance/retired_tools"/Path(name).parent)
            archived.append(str(dest.relative_to(base)))
            record["files_sha256"][str(dest.relative_to(base))] = sha(dest)
        path.unlink()
        removed.append(name)
        print(f"[REMOVE] {name}")
    template = base/"template"
    if template.is_dir() and not any(template.iterdir()):
        template.rmdir()
    for name in ROOT_LOGS:
        path = base/name
        if not path.exists():
            continue
        require(path.is_file() and not path.is_symlink(), f"Unexpected root log: {path}")
        dest = save_unique(path, base/"review_outputs/diagnostics/root_logs")
        path.unlink()
        record["files_sha256"][str(dest.relative_to(base))] = sha(dest)
        log_moves.append({"old": name, "new": str(dest.relative_to(base))})
        print(f"[MOVE] {name} -> {dest.relative_to(base)}")
    for cache in (base/"__pycache__", base/"python/__pycache__", base/"tests/__pycache__"):
        if cache.exists():
            require(cache.is_dir() and not cache.is_symlink(), f"Unexpected cache: {cache}")
            shutil.rmtree(cache)
    if removed or archived or log_moves:
        record.setdefault("maintenance", []).append({"source_commit": head,
            "removed_one_off_files": removed, "archived_uncommitted_versions": archived,
            "log_relocations": log_moves})
        manifest = base/"review_outputs/organisation_manifest.json"
        manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def safe_to_publish(name):
    p = Path(name)
    return (p.parts[0] == "NIsoMuon" and "unblind" not in p.parts
            and "__pycache__" not in p.parts
            and p.suffix.lower() not in {".root", ".png", ".pyc", ".pyo"})


def include_new(name):
    p = Path(name)
    if not safe_to_publish(name):
        return False
    r = p.relative_to("NIsoMuon")
    if len(r.parts) == 1:
        return r.suffix in {".py", ".sh", ".md", ".yml", ".yaml"}
    if r.parts[0] in ("python", "tests"):
        return r.suffix == ".py"
    if r.parts[0] == "sigfit_inputs":
        return r.suffix in (".json", ".csv")
    if r.parts[:3] in (("datacards_NIsoMuon", "alpha", "blind"), ("limit_outputs", "alpha", "blind")):
        return r.suffix in TEXT_RESULTS
    if str(r) in ("limit_outputs/final_impacts_manifest.json", "review_outputs/organisation_manifest.json",
                  "review_outputs/relocation_map.json"):
        return True
    if r.parts[0] == "review_outputs":
        return r.suffix in TEXT_RESULTS
    return False


def commit_results(repo):
    index = paths(repo, "ls-files", "-z", "--", "NIsoMuon")
    head = paths(repo, "ls-tree", "-rz", "--name-only", "HEAD", "--", "NIsoMuon")
    untracked = paths(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", "NIsoMuon")
    selected = {name for name in index | head if safe_to_publish(name)}
    selected.update(name for name in untracked if include_new(name) and not (repo/name).is_symlink())
    selected = sorted(selected)
    # Avoid re-adding already-staged deletions, which no longer appear in index.
    add = [name for name in selected if name in index or (repo/name).exists()]
    for name in add:
        p = repo/name
        if p.is_file():
            require(p.stat().st_size < 95*1024*1024, f"Too large for this GitHub publication: {p}")
    for start in range(0, len(add), 100):
        subprocess.run(["git", "-C", str(repo), "add", "-A", "--", *add[start:start+100]], check=True)
    if not selected:
        return
    status = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet", "--", *selected]).returncode
    require(status in (0, 1), "Cannot inspect staged results")
    if status == 1:
        # --only prevents unrelated previously staged files from entering this commit.
        subprocess.run(["git", "-C", str(repo), "commit", "--only", "-m",
                        "Preserve organised blind results and retire one-off production tools",
                        "--", *selected], check=True)
    subprocess.run(["git", "-C", str(repo), "push", SSH, "HEAD:main"], check=True)
    print(f"[PUSHED] higgs_combine/main {git(repo, 'rev-parse', 'HEAD')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    require(not args.push or args.apply, "--push requires --apply")
    base = args.production_dir.resolve()
    require(base.name == "NIsoMuon", "Run from or specify the NIsoMuon production directory")
    repo = Path(git(base, "rev-parse", "--show-toplevel"))
    require(base == repo/"NIsoMuon", "Expected NIsoMuon immediately under the git repository root")
    require(git(repo, "branch", "--show-current") == "main", "Expected existing main branch")
    origin = git(repo, "remote", "get-url", "origin")
    require(origin in (SSH, "https://github.com/joonblee/higgs_combine.git"), "Unexpected origin repository")
    with (base.parent/".NIsoMuon-organise.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("The production organisation helper is still running")
        record = audit(base)
        for name in REMOVE:
            if (base/name).exists():
                print(f"[PLAN REMOVE] {name}")
        for name in ROOT_LOGS:
            if (base/name).exists():
                print(f"[PLAN MOVE LOG] {name} -> review_outputs/diagnostics/root_logs/")
        if not args.apply:
            print("[DRY RUN] No production files changed; add --apply, optionally --push")
            return
        clean(base, repo, record)
        if args.push:
            commit_results(repo)
    print("[DONE] Inputs, scientific outputs and review/bias helpers retained; no GitLab or AN changes")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        sys.exit(f"[ERROR] {exc}")
