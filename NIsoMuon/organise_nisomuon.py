#!/usr/bin/env python3
"""Consolidate the reviewed NIsoMuon results without running Combine.

Default: validate and print a plan. --apply installs verified copies and moves
old trees to a sibling rollback directory. --delete-obsolete also removes that
rollback directory after the complete installation has passed its hash audit.
No git add/commit/push, no AN edits, and no changes to datacards or inputs.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MASSES = (12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)
TARGETS = ("Run2", "Run3", "Run2Run3")
CANONICAL = ("limit_outputs", "review_outputs")
REVIEW_BASE = "review_outputs_updated.P2ejoE9l"
M20 = "bias_M20_verified.7x_fn8sj"
M70 = "bias_M70_verified.avf0d2lz"
DEPENDENCIES = (
    "bias_M20_verified.d4jw2m6v", "bias_repair_M20_fallback.7jr2691k",
    "bias_repair_M20_poiseed.39z1r77d", "bias_M70_audit.xp1jigmo",
)
# Only these explicitly superseded runs are disposable without an exact copy.
OBSOLETE = (
    "bias_M20_fallback.0xJKrZIc", "bias_M20_probe.BmNJspFs",
    "bias_repair_M20.WBgJ4pg8", "review_outputs_old", "review_logs_old",
)


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def regular_tree(root):
    """Reject links and special files in trees that will be copied/removed."""
    require(not root.is_symlink(), f"Refusing symbolic link: {root}")
    if root.is_file():
        yield root
        return
    require(root.is_dir(), f"Missing directory: {root}")
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(current) / name
            require(not p.is_symlink(), f"Refusing symbolic link: {p}")
            require(p.is_dir() or p.is_file(), f"Refusing special file: {p}")
        for name in files:
            yield Path(current) / name


def refs(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from refs(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from refs(item)


def snapshot_source(base, stored):
    # The snapshot contains the original server's absolute paths.
    marker = "/NIsoMuon/"
    require(marker in stored, f"Unexpected snapshot path: {stored}")
    relative = Path(stored.split(marker, 1)[1])
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe snapshot path: {stored}")
    require(relative.parts[0] in ("limit_outputs", "limit_outputs_impacts_updated",
                                "limit_outputs_impacts_retry"), str(relative))
    return base / relative


def check_bias(directory, mass):
    result = load(directory / "validation.json")
    require(result.get("passed") is True and not result.get("errors"),
            f"Bias validation did not pass: {directory}")
    require(result.get("n_verified_repairs") == {20: 32, 70: 4}[mass],
            f"Unexpected repaired-toy count: {directory}")
    merged = load(directory / "merged_toys.json")
    require(len(merged) == 3, f"Expected three ensembles: {directory}")
    for key, toys in merged.items():
        require(set(toys) == set(map(str, range(1, 501))),
                f"Not 500 distinct original toy IDs: {directory}/{key}")
        for toy, row in toys.items():
            interval = row["interval"]
            require(len(interval) == 3 and all(map(math.isfinite, interval))
                    and interval[0] < interval[1] < interval[2],
                    f"Invalid interval: {key}/{toy}")
    summary = directory / f"bias_summary_Run2Run3_M{mass}.json"
    require(summary.is_file() and len(load(summary)) == 3, f"Missing summary: {summary}")
    inputs = load(directory / "input_manifest.json")
    # The audit's original input hashes must still match before reorganising.
    for name, digest in inputs["input_sha256"].items():
        path = Path(name)
        require(path.is_file() and sha(path) == digest, f"Bias input changed/missing: {path}")


def impact_member(name, target, mass):
    tag = re.escape(f"{target}_alpha_M{mass}") + r"(?![0-9])"
    return (re.search(tag, name) is not None
            or re.fullmatch(rf"impact(?:_poi)?_repair_M{mass}", name) is not None
            or name == f"impact_snapshot_M{mass}.json")


def make_plan(base, review):
    manifest = load(review / "preservation/snapshot_manifest.json")
    expected = {(t, m) for t in TARGETS for m in MASSES}
    entries = manifest["b_only_impacts"]
    require(len(entries) == 39 and {(e['target'], e['mass']) for e in entries} == expected,
            "The review snapshot must contain exactly 39 distinct impact sets")
    impacts = []
    for e in entries:
        target, mass = e["target"], e["mass"]
        source = snapshot_source(base, e["source_directory"])
        stem = f"impacts_{target}_alpha_M{mass}"
        internal = source / (stem + "_internal.json")
        card = base / f"datacards_NIsoMuon/alpha/blind/combined/{target}/datacard_M-{mass}_{target}.txt"
        require(card.is_file() and sha(card) == e["card_sha256"], f"Card changed: {card}")
        require(internal.is_file() and sha(internal) == e["internal_json_sha256"],
                f"Not the reviewed internal JSON: {internal}")
        payload = load(internal)
        fit = payload["POIs"][0]["fit"]
        require(len(fit) == 3 and all(map(math.isfinite, fit)) and fit[0] < 0 < fit[2],
                f"Expected a negative-r B-only interval: {internal}")
        reference = review / f"review_validation/b_only_impacts/{target}/M{mass}"
        for suffix in (".pdf", ".json", "_internal.json"):
            p, q = source / (stem + suffix), reference / (stem + suffix)
            require(p.is_file() and q.is_file() and sha(p) == sha(q),
                    f"Production/review snapshot mismatch: {p}")
        require((source / f"workspace_{target}_alpha_M{mass}.root").is_file(),
                f"Missing matching workspace: {source} M{mass}")
        members = [p for p in source.iterdir() if impact_member(p.name, target, mass)]
        require(any(p.name.startswith("higgsCombine_initialFit_") and p.suffix == ".root"
                    for p in members), f"Missing original impact initial fit: {source} M{mass}")
        for subdir in e.get("validation_directories", []):
            require(Path(subdir).name == subdir and (source/subdir/"validation.json").is_file(),
                    f"Missing impact validation: {source}/{subdir}")
        impacts.append({"target": target, "mass": mass, "source": source,
                        "members": sorted(members),
                        "asset_hashes": {stem+s: sha(source/(stem+s))
                                         for s in (".pdf", ".json", "_internal.json")}})
    for mass, name in ((20, M20), (70, M70)):
        check_bias(base / name, mass)
    run = load(base / REVIEW_BASE / "run_manifest.json")
    for entry in run["cards"].values():
        require(sha(Path(entry["path"])) == entry["sha256"], "Review datacard hash mismatch")
    require(sha(Path(run["limit_json"])) == run["limit_sha256"], "Review limit hash mismatch")

    # These complete raw trees carry reused profile checks and original toy fits.
    moves = {REVIEW_BASE: "review_outputs", M20: "review_outputs/Run2Run3/M20/bias/verified",
             M70: "review_outputs/Run2Run3/M70/bias/verified"}
    for name in DEPENDENCIES:
        require((base/name).is_dir(), f"Missing bias provenance directory: {name}")
        moves[name] = "review_outputs/provenance/" + name
    for p in sorted(base.iterdir()):
        if not p.is_dir() or p.name in moves or p.name in OBSOLETE:
            continue
        if p.name.startswith(("diag_", "fitdiag_blind_retry.", "diagnostics_zero_dy")):
            moves[p.name] = "review_outputs/diagnostics/" + p.name
        elif p.name.startswith(("bias_M", "bias_repair_", "review_outputs_updated.")):
            # E.g. hul4v2hz: its role is unknown, so keep its contents, not assume redundant.
            moves[p.name] = "review_outputs/additional_runs/" + p.name
        elif p.name == "review_logs":
            moves[p.name] = "review_outputs/logs/previous_driver_logs"
    retired = set(moves) | set(CANONICAL)
    retired.update(p.name for p in base.glob("limit_outputs_impacts_*") if p.is_dir())
    retired.update(name for name in OBSOLETE if (base/name).exists())
    if (base/"__pycache__").exists():
        retired.add("__pycache__")
    if (base/"python/__pycache__").exists():
        retired.add("python/__pycache__")
    if (base/"tests/__pycache__").exists():
        retired.add("tests/__pycache__")
    # Do not discard an unrecognised impact output just because its name matches.
    allowed_impact = {i["source"].relative_to(base).parts[0] for i in impacts}
    for name in list(retired):
        if name.startswith("limit_outputs_impacts_") and name not in allowed_impact:
            moves[name] = "review_outputs/additional_runs/" + name

    # Refuse to retire inputs referenced by retained JSON unless a destination exists.
    mapping = {str(base/name): str(base/dest) for name, dest in moves.items()}
    for i in impacts:
        mapping[str(i["source"])] = str(base/f"limit_outputs/alpha/blind/{i['target']}")
        for p in i["members"]:
            dest = base / f"limit_outputs/alpha/blind/{i['target']}" / p.name
            mapping[str(p)] = str(dest)
    for top in moves:
        for p in regular_tree(base/top):
            if p.suffix != ".json":
                continue
            for value in refs(load(p)):
                if not value.startswith(str(base) + "/"):
                    continue
                relative = Path(value).relative_to(base)
                if relative.parts[0] not in retired or relative.parts[0] in CANONICAL:
                    continue
                if Path(value).exists():
                    require(any(value == old or value.startswith(old + "/") for old in mapping),
                            f"Unpreserved dependency in {p}: {value}")
    for name in retired:
        if (base/name).exists():
            list(regular_tree(base/name))
    return {"manifest": manifest, "impacts": impacts, "moves": moves,
            "retired": sorted(retired), "mapping": mapping}


def relocated(value, mapping):
    if isinstance(value, str):
        for old in sorted(mapping, key=len, reverse=True):
            if value == old or value.startswith(old + "/"):
                return mapping[old] + value[len(old):]
        return value
    if isinstance(value, list):
        return [relocated(x, mapping) for x in value]
    if isinstance(value, dict):
        return {relocated(k, mapping): relocated(v, mapping) for k, v in value.items()}
    return value


def copy_checked(source, destination, ledger):
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for p in regular_tree(source):
            copy_checked(p, destination / p.relative_to(source), ledger)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha(source)
    require(sha(destination) == digest, f"Copy failed hash verification: {destination}")
    ledger[str(source)] = {"destination": str(destination), "sha256": digest}
    if len(ledger) % 100 == 0:
        print(f"[COPY] {len(ledger)} files verified", flush=True)


def stage_results(base, plan, stage):
    ledger = {}
    mapping = dict(plan["mapping"])
    copy_checked(base/"limit_outputs", stage/"limit_outputs", ledger)
    copy_checked(base/REVIEW_BASE, stage/"review_outputs", ledger)
    # The existing review_outputs is superseded, but retain any differing or unique
    # files so a standalone diagnostic in that directory cannot disappear.
    if (base/"review_outputs").is_dir():
        for p in regular_tree(base/"review_outputs"):
            target = stage/"review_outputs"/p.relative_to(base/"review_outputs")
            if not target.is_file() or sha(p) != sha(target):
                saved = stage/"review_outputs/provenance/previous_review_outputs"/p.relative_to(base/"review_outputs")
                copy_checked(p, saved, ledger)
                mapping[str(p)] = str(base/saved.relative_to(stage))
    for item in plan["impacts"]:
        target, mass, source = item["target"], item["mass"], item["source"]
        dest = stage / f"limit_outputs/alpha/blind/{target}"
        # Remove old matching roots, JSONs and repair dirs before installing a
        # complete selected set. This prevents mixing fits from different runs.
        if dest.is_dir():
            for p in dest.iterdir():
                if impact_member(p.name, target, mass):
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
        for p in item["members"]:
            copy_checked(p, dest/p.name, ledger)
        for name in ("impact_translation_alpha.json", "_plotImpacts_NPS26009_top30.py"):
            if (source/name).is_file():
                copy_checked(source/name, dest/name, ledger)
        for name, digest in item["asset_hashes"].items():
            require(sha(dest/name) == digest, f"Impact changed during organisation: {source/name}")
    # Fresh M12 run directories contain copies of the same ROOT/PDF assets now
    # installed above. Keep unique logs/metadata, and map identical raw assets to
    # their final files rather than keeping another physical copy.
    final_assets = {}
    for p in regular_tree(stage/"limit_outputs"):
        if p.suffix in (".root", ".pdf"):
            final_assets.setdefault(sha(p), str(base/p.relative_to(stage)))
    for name, destination in plan["moves"].items():
        if name != REVIEW_BASE:
            if name.startswith("limit_outputs_impacts_"):
                (stage/destination).mkdir(parents=True, exist_ok=True)
                for p in regular_tree(base/name):
                    duplicate = final_assets.get(sha(p)) if p.suffix in (".root", ".pdf") else None
                    if duplicate:
                        mapping[str(p)] = duplicate
                    else:
                        copy_checked(p, stage/destination/p.relative_to(base/name), ledger)
            else:
                copy_checked(base/name, stage/destination, ledger)

    # Preserve the original summary used to generate/refit the toys. The normal
    # bias_summary name is replaced with the verified, merged final ensemble.
    for mass in (20, 70):
        dest = stage/f"review_outputs/Run2Run3/M{mass}/bias"
        filename = f"bias_summary_Run2Run3_M{mass}.json"
        original = dest/filename
        require(original.is_file(), f"Missing original summary: {original}")
        preserved = dest/"original"/filename
        copy_checked(original, preserved, ledger)
        old = str(base/REVIEW_BASE/f"Run2Run3/M{mass}/bias"/filename)
        mapping[old] = str(base/preserved.relative_to(stage))
        copy_checked(dest/"verified"/filename, original, ledger)

    # Some saved argparse options use NIsoMuon-relative rather than absolute paths.
    for old, new in list(mapping.items()):
        mapping[str(Path(old).relative_to(base))] = str(Path(new).relative_to(base))

    # Keep exact historical JSONs before producing live copies with relocated
    # paths. Command logs retain their original text as historical records.
    raw = stage/"review_outputs/provenance/original_json"
    all_json = [p for root in CANONICAL for p in (stage/root).rglob("*.json")]
    for p in all_json:
        value = load(p)
        updated = relocated(value, mapping)
        if updated != value:
            original = raw/p.relative_to(stage)
            copy_checked(p, original, ledger)
            dump(p, updated)

    def staged(path):
        p = Path(path)
        try:
            relative = p.relative_to(base)
        except ValueError:
            return p
        if relative.parts[0] in CANONICAL:
            return stage/relative
        return p

    # Refresh hashes for relocated input JSONs; exact original manifests above
    # keep the hashes and paths that were recorded at execution time.
    for p in all_json:
        value = load(p)
        if isinstance(value, dict) and isinstance(value.get("input_sha256"), dict):
            for name in value["input_sha256"]:
                q = staged(name)
                require(q.is_file(), f"Relocated input is missing: {name}")
                value["input_sha256"][name] = sha(q)
            dump(p, value)

    # Existing file references must resolve after relocation, including those
    # inside selected impact repair JSONs. Historical nonexistent command outputs
    # are not interpreted as required inputs.
    old_json = [p for top in plan["moves"] for p in regular_tree(base/top) if p.suffix == ".json"]
    old_json += [p for i in plan["impacts"] for member in i["members"]
                 for p in regular_tree(member) if p.suffix == ".json"]
    for p in old_json:
        for name in refs(load(p)):
            if name.startswith(str(base) + "/") and Path(name).is_file():
                new = relocated(name, mapping)
                require(staged(new).is_file(), f"Missing relocated file reference from {p}: {new}")

    manifest = relocated(plan["manifest"], mapping)
    for e in manifest["b_only_impacts"]:
        e["source_directory"] = str(base/f"limit_outputs/alpha/blind/{e['target']}")
    dump(stage/"limit_outputs/final_impacts_manifest.json", manifest)
    dump(stage/"review_outputs/relocation_map.json", {
        "path_prefixes": mapping,
        "historical_json": "provenance/original_json",
        "note": "Live JSON paths are relocated; exact historical JSON and logs are preserved."})
    # A persistent hash inventory is also used after promotion.
    inventory = {str(p.relative_to(stage)): sha(p) for root in CANONICAL
                 for p in regular_tree(stage/root)}
    dump(stage/"review_outputs/organisation_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "copied_from": str(base), "files_sha256": inventory,
        "removed_old_roots": plan["retired"],
        "scope": "File organisation only; no new minimisations or numerical certification."})
    return inventory


def install(base, stage, plan, inventory, backup):
    # Same-filesystem renames: keep originals until both destinations are in place
    # and every installed file has passed its checksum. Roll back on failure.
    moved, installed = [], []
    backup.mkdir()
    try:
        for root in CANONICAL:
            if (base/root).exists():
                (base/root).rename(backup/root)
                moved.append(root)
            (stage/root).rename(base/root)
            installed.append(root)
        for name in plan["retired"]:
            if name in CANONICAL or not (base/name).exists():
                continue
            (backup/name).parent.mkdir(parents=True, exist_ok=True)
            (base/name).rename(backup/name)
            moved.append(name)
        for relative, digest in inventory.items():
            require(sha(base/relative) == digest, f"Installed hash mismatch: {relative}")
        dump(backup/"rollback.json", {"base": str(base), "moved": moved,
                                      "installed": installed})
    except BaseException:
        for root in reversed(installed):
            if (base/root).exists():
                (base/root).rename(stage/root)
        for name in reversed(moved):
            (base/name).parent.mkdir(parents=True, exist_ok=True)
            (backup/name).rename(base/name)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=Path.cwd())
    parser.add_argument("--review-repo", type=Path, help="Existing current combine_review checkout")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--delete-obsolete", action="store_true",
                        help="After successful installation, permanently delete the old-tree rollback copy")
    args = parser.parse_args()
    require(not args.delete_obsolete or args.apply, "--delete-obsolete requires --apply")
    base = args.production_dir.resolve()
    require(base.name == "NIsoMuon" and (base/"limit_workflow.py").is_file()
            and (base/"datacards_NIsoMuon").is_dir(), "Not a NIsoMuon production directory")
    lock_stream = (base.parent/".NIsoMuon-organise.lock").open("a")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another organisation helper is already running")
    # Refuse to reorganise files while a Combine/workflow process is still using
    # this production tree. Other shells/editors in the same directory are fine.
    for proc in Path("/proc").glob("[0-9]*"):
        if proc.name == str(os.getpid()):
            continue
        try:
            command = (proc/"cmdline").read_bytes().split(b"\0")
            cwd = (proc/"cwd").resolve()
            names = {Path(x.decode(errors="replace")).name for x in command[:4]}
            fitting = bool(names & {"combine", "combineTool.py", "limit_workflow.py",
                                    "combine_review.py", "refresh_m12_impacts.py",
                                    "validate_bias_repair.py", "repair_bias_intervals.py"})
            using_base = cwd == base or base in cwd.parents or any(str(base).encode() in x for x in command)
            require(not (fitting and using_base), f"Active fit/workflow process {proc.name}; finish it first")
        except (OSError, PermissionError):
            continue
    finished = base/"review_outputs/organisation_manifest.json"
    if finished.is_file() and not (base/REVIEW_BASE).exists():
        recorded = load(finished)
        for name, digest in recorded["files_sha256"].items():
            require((base/name).is_file() and sha(base/name) == digest,
                    f"Organised result changed/missing: {name}")
        print("[ALREADY ORGANISED] Saved file hashes match; nothing deleted.")
        return
    with tempfile.TemporaryDirectory(prefix="nps26009-cleanup-review-") as work:
        review = args.review_repo.resolve() if args.review_repo else Path(work)/"combine_review"
        if not args.review_repo:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "master",
                            "git@github.com:joonblee/combine_review.git", str(review)], check=True)
        plan = make_plan(base, review)
        print("[PASS] 39 reviewed B-only impacts; M20/M70 final merged 3x500 toy intervals")
        print("[KEEP] datacards_NIsoMuon, all input/code/template/python/tests directories")
        print("[FINAL] limit_outputs/alpha/blind/{Run2,Run3,Run2Run3}: all 13 masses")
        print("[FINAL] review_outputs/Run2Run3/{M20,M70}, each with bias/verified")
        for old, new in plan["moves"].items():
            print(f"[RELOCATE] {old} -> {new}")
        for old in plan["retired"]:
            print(f"[RETIRE] {old}")
        if not args.apply:
            print("[DRY RUN] No production files changed. Add --apply to install.")
            return
        # Estimate conservatively: base limit tree plus source trees and selected
        # impacts. Old trees stay on disk until the complete copy has passed.
        sources = [base/"limit_outputs", base/"review_outputs"] + [base/name for name in plan["moves"]]
        sources = [p for p in sources if p.exists()]
        sources += [p for i in plan["impacts"] for p in i["members"]]
        needed = sum(p.stat().st_size for source in sources for p in regular_tree(source))
        require(shutil.disk_usage(base.parent).free > needed * 1.15,
                f"Insufficient space for verified staging: need about {needed*1.15/2**30:.1f} GiB")
        stage = Path(tempfile.mkdtemp(prefix="NIsoMuon-organise-stage-", dir=base.parent))
        backup = base.parent/("NIsoMuon-before-organise-" + stage.name.rsplit("-", 1)[-1])
        print(f"[STAGING] {stage}", flush=True)
        try:
            inventory = stage_results(base, plan, stage)
            install(base, stage, plan, inventory, backup)
        except BaseException:
            print(f"[STOPPED] Original trees retained/restored. Staging: {stage}", file=sys.stderr)
            raise
        shutil.rmtree(stage)
        if args.delete_obsolete:
            shutil.rmtree(backup)
            print("[DELETED] Superseded trees and rollback copies after installed-file hash verification")
        else:
            print(f"[ROLLBACK COPY] {backup}")
        print("[DONE] Organisation complete. Git repositories and AN were not pushed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        sys.exit(f"[ERROR] {error}")
