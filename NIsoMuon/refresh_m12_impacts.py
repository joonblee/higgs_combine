#!/usr/bin/env python3
"""Run Run2/Run3 M12 impacts, validate and sync all three negative-r results.

Initialise the original Combine environment first. The completed Run2Run3
directory is required; it is never rerun. GitLab is never contacted. --push
commits only the selected M12 files in the existing GitHub default branches.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import limit_workflow as workflow

TARGETS = ("Run2", "Run3", "Run2Run3")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(command, cwd=None):
    subprocess.run([str(x) for x in command], cwd=cwd, check=True)


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *map(str, args)], text=True).strip()


def card_for(production, target):
    return production / f"datacards_NIsoMuon/alpha/blind/combined/{target}/datacard_M-12_{target}.txt"


def validate(source, card, target):
    stem = f"impacts_{target}_alpha_M12"
    internal = load(source / (stem + "_internal.json"))
    physical = load(source / (stem + ".json"))
    pdf = source / (stem + ".pdf")
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError(f"Missing/invalid PDF: {pdf}")
    report = load(source / "impact_repair_M12/validation.json")
    if (not report["passed"] or report["target"] != target
            or report["mass"] != 12 or report["expected_signal"] != 0):
        raise RuntimeError(f"Not a passing B-only M12 validation: {source}")
    fit = workflow.impact_fit_triplet(internal)
    if fit is None or not all(map(math.isfinite, fit)) or not fit[0] < 0 < fit[2]:
        raise RuntimeError(f"M12 POI does not cross zero: {fit}")
    if abs(fit[1]) > max(1e-5, 1e-3 * min(-fit[0], fit[2])):
        raise RuntimeError(f"B-only central r differs materially from zero: {fit}")
    rmin, rmax = report["initial_poi"]["rmin"], report["initial_poi"]["rmax"]
    if workflow.impact_poi_interval_issues(internal, rmin, rmax, True):
        raise RuntimeError("M12 POI interval reaches a fit-range endpoint")
    workspace = source / f"workspace_{target}_alpha_M12.root"
    bounds = workflow.impact_workspace_parameters(workspace)
    missing, bad = workflow.impact_interval_issues(internal, bounds)
    if (missing or bad or len(internal["params"]) != report["expected"]
            or report["expected"] != len(bounds)):
        raise RuntimeError(f"Incomplete nuisance intervals: missing={missing}, bad={bad}")
    unit = workflow.alpha_unit_from_card(card)
    expected = copy.deepcopy(internal)
    workflow.rescale_impact_payload(expected, unit)
    # Compare only the fields that are converted; nuisance coordinates stay unchanged.
    if {p['name'] for p in expected['params']} != {p['name'] for p in physical['params']}:
        raise RuntimeError("Physical/internal nuisance names differ")
    def same(a, b):
        return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-20)
                                      for x, y in zip(a, b))
    if len(expected["POIs"]) != len(physical["POIs"]):
        raise RuntimeError("Physical/internal POI counts differ")
    for a, b in zip(expected["POIs"], physical["POIs"]):
        if a["name"] != b["name"] or not same(a["fit"], b["fit"]):
            raise RuntimeError("Physical POI conversion differs")
    converted = {p['name']: p for p in physical['params']}
    for p in expected['params']:
        q = converted[p['name']]
        if (not same(p['r'], q['r']) or not same(p['fit'], q['fit'])
                or not math.isclose(p['impact_r'], q['impact_r'], rel_tol=1e-10, abs_tol=1e-20)):
            raise RuntimeError(f"Alpha conversion differs: {p['name']}")
    return {"passed": True, "target": target, "mass": 12, "expected_signal": 0,
            "source_directory": str(source), "card_sha256": sha(card),
            "workspace_sha256": sha(workspace), "internal_json_sha256": sha(source/(stem+"_internal.json")),
            "physical_json_sha256": sha(source/(stem+".json")), "pdf_sha256": sha(pdf),
            "initial_poi": list(fit), "rmin": rmin, "rmax": rmax,
            "n_nuisances": len(bounds), "alpha_internal_unit": unit}


def clone_repo(work, repo, branch, sparse=None):
    destination = work / repo
    command = ["git", "clone", "--depth", "1", "--single-branch", "--branch", branch]
    if sparse:
        command += ["--filter=blob:none", "--no-checkout"]
    run(command + [f"git@github.com:joonblee/{repo}.git", destination])
    if sparse:
        run(["git", "-C", destination, "sparse-checkout", "init", "--no-cone"])
        run(["git", "-C", destination, "sparse-checkout", "set", "--no-cone", *sparse])
        run(["git", "-C", destination, "checkout", branch])
    return destination


def check_repo(directory, repo, branch):
    if git(directory, "branch", "--show-current") != branch:
        raise RuntimeError(f"Expected {repo} branch {branch}: {directory}")
    url = git(directory, "remote", "get-url", "origin")
    if not re.search(rf"[:/]joonblee/{re.escape(repo)}(?:\.git)?$", url):
        raise RuntimeError(f"Wrong origin for {repo}: {url}")


def commit_selected(directory, paths, message, push):
    paths = sorted(set(str(p) for p in paths))
    run(["git", "-C", directory, "diff", "--check", "--", *paths])
    if not push:
        print(f"[PREPARED] {directory}: {len(paths)} selected paths; not committed/pushed", flush=True)
        return
    run(["git", "-C", directory, "add", "-f", "--", *paths])
    changed = subprocess.run(["git", "-C", str(directory), "diff", "--cached", "--quiet", "--", *paths]).returncode
    if changed == 1:
        run(["git", "-C", directory, "commit", "--only", "-m", message, "--", *paths])
    elif changed != 0:
        raise RuntimeError("Cannot inspect selected git changes")
    branch = git(directory, "branch", "--show-current")
    run(["git", "-C", directory, "push", f"git@github.com:joonblee/{git(directory, 'remote', 'get-url', 'origin').rstrip('/').split('/')[-1].removesuffix('.git')}.git", f"HEAD:{branch}"])


def update_an_captions(text, reports):
    old_sentence = (r"This stored $12\GeV$ diagnostic uses $r\geq0$, giving one-sided impacts "
                    r"at the background-only boundary.")
    for target in TARGETS:
        path = f"figures/limit/impacts_{target}_alpha_M12.pdf"
        figures = list(re.finditer(r"\\begin\{figure\}(?:\[[^\]]*\])?.*?\\end\{figure\}", text, re.S))
        selected = [m for m in figures if "{" + path + "}" in m.group()]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one AN figure for {path}")
        m = selected[0]; block = m.group()
        cap = re.search(r"(\\caption\{)(.*?)(\}\s*\\label\{)", block, re.S)
        if not cap:
            raise RuntimeError(f"Cannot locate M12 caption: {target}")
        caption = cap[2].replace(old_sentence, "").rstrip()
        sentence = (r"The diagnostic fit allows negative internal $r$ in the range $["
                    + f"{reports[target]['rmin']:g}, {reports[target]['rmax']:g}"
                    + r"]$; physical upper limits retain $r\geq0$.")
        if sentence not in caption:
            caption += " " + sentence
        new_block = block[:cap.start(2)] + caption + block[cap.end(2):]
        text = text[:m.start()] + new_block + text[m.end():]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=Path.cwd())
    parser.add_argument("--full-source", type=Path, required=True,
                        help="Completed Run2Run3 output directory containing impacts JSON/PDF")
    parser.add_argument("--run2-source", type=Path, help="Reuse completed new Run2 results instead of rerunning")
    parser.add_argument("--run3-source", type=Path, help="Reuse completed new Run3 results instead of rerunning")
    parser.add_argument("--impact-parallel", type=int, default=40)
    parser.add_argument("--push", action="store_true", help="Commit/push selected files to the three GitHub repos")
    args = parser.parse_args()
    if args.impact_parallel < 1:
        parser.error("--impact-parallel must be positive")
    production = args.production_dir.resolve()
    producer_root = Path(git(production, "rev-parse", "--show-toplevel"))
    check_repo(producer_root, "higgs_combine", "main")
    work = Path(tempfile.mkdtemp(prefix="nps26009-M12-sync."))
    print(f"[WORK] {work}", flush=True)
    # Validate the already completed full result before launching more jobs.
    full = args.full_source.resolve()
    validate(full, card_for(production, "Run2Run3"), "Run2Run3")
    output = Path(tempfile.mkdtemp(prefix="limit_outputs_impacts_M12_negative.", dir=production))
    sources = {"Run2Run3": full}
    driver = []
    for target, supplied in (("Run2", args.run2_source), ("Run3", args.run3_source)):
        if supplied:
            sources[target] = supplied.resolve()
            continue
        command = [sys.executable, str(production/"limit_workflow.py"), "--stage", "run",
                   "--task", "impacts", "--target", target, "--parameter", "alpha",
                   "--mode", "blind", "--masses", "12", "--allow-negative-r",
                   "--r-min", "-50", "--r-max", "100", "--impact-parallel", str(args.impact_parallel),
                   "--strict", "--output-base", str(output)]
        log = output / f"run_{target}_M12.log"
        card_hash = sha(card_for(production, target))
        print(f"[RUN] {target} M12, parallel={args.impact_parallel}; log={log}", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=production, stdout=stream, stderr=subprocess.STDOUT, env=env)
        driver.append({"target": target, "command": command, "exit_code": result.returncode,
                       "card_sha256": card_hash, "log": str(log)})
        dump(output / "driver_manifest.json", driver)
        if result.returncode or sha(card_for(production, target)) != card_hash:
            raise RuntimeError(f"{target} failed or its card changed; inspect {log}")
        sources[target] = output / f"alpha/blind/{target}"
    dump(work / "completed_sources.json", {t: str(p) for t, p in sources.items()})
    reports = {t: validate(sources[t], card_for(production,t),t) for t in TARGETS}
    dump(work / "validation.json", reports)
    review = clone_repo(work, "combine_review", "master")
    an = clone_repo(work, "CMS-NPS-26-009_AN", "main", sparse=["/AGENTS.md", "/NPS-26-009.tex"] +
                    [f"/figures/limit/impacts_{t}_alpha_M12.pdf" for t in TARGETS])
    check_repo(review, "combine_review", "master"); check_repo(an, "CMS-NPS-26-009_AN", "main")
    manifest_path = review / "preservation/snapshot_manifest.json"
    manifest = load(manifest_path)
    # Stage all validation and text edits before copying any production result.
    for target in TARGETS:
        reviewed_card = review / f"review_validation/b_only_impacts/{target}/M12/datacard_M-12_{target}.txt"
        if sha(reviewed_card) != reports[target]["card_sha256"]:
            raise RuntimeError(f"Production/review M12 cards differ: {target}")
        if len([r for r in manifest['b_only_impacts'] if r['target']==target and r['mass']==12]) != 1:
            raise RuntimeError(f"Ambiguous M12 manifest entry: {target}")
    tex = an / "NPS-26-009.tex"
    new_tex = update_an_captions(tex.read_text(), reports)
    selected = {"producer": [], "review": ["preservation/snapshot_manifest.json"],
                "an": ["NPS-26-009.tex"]}
    def copy_file(source, destination, backup=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and backup is not None:
            backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(destination, backup)
        shutil.copy2(source, destination)
        if sha(source) != sha(destination):
            raise RuntimeError(f"Copy verification failed: {destination}")
    for target in TARGETS:
        source = sources[target]; stem = f"impacts_{target}_alpha_M12"
        assets = [stem+suffix for suffix in (".pdf", ".json", "_internal.json")]
        # Both the preferred updated directory and the old viewing path get the
        # matching ROOT scans too: never pair a new JSON with old fit products.
        roots = [p.name for p in source.glob("*.root") if
                 p.name == f"workspace_{target}_alpha_M12.root" or f"Impacts_{target}_alpha_M12" in p.name]
        for base in ("limit_outputs_impacts_updated", "limit_outputs"):
            destination = production / f"{base}/alpha/blind/{target}"
            for name in assets + roots:
                copy_file(source/name, destination/name, work/"backups"/target/base/name)
            for subdir in ("impact_repair_M12", "impact_poi_repair_M12"):
                if (source/subdir).is_dir():
                    dest = destination/subdir
                    if dest.exists():
                        shutil.copytree(dest, work/"backups"/target/base/subdir)
                        shutil.rmtree(dest)
                    shutil.copytree(source/subdir, dest)
            dump(destination / "impact_snapshot_M12.json", reports[target])
            selected["producer"] += [str((destination/name).relative_to(producer_root)) for name in assets]
            for path in destination.glob("impact*repair_M12/validation.json"):
                selected["producer"].append(str(path.relative_to(producer_root)))
            selected["producer"].append(str((destination/"impact_snapshot_M12.json").relative_to(producer_root)))
        current = production / f"limit_outputs_impacts_updated/alpha/blind/{target}"
        dest = review / f"review_validation/b_only_impacts/{target}/M12"
        for name in assets:
            copy_file(current/name, dest/name)
            selected["review"].append(str((dest/name).relative_to(review)))
        # The compact snapshot carries numerical validation, not raw ROOT scans.
        old_validations = list(dest.rglob("*validation.json"))
        for path in old_validations:
            selected["review"].append(str(path.relative_to(review)))
            backup = work/"backups/review"/path.relative_to(review)
            backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, backup); path.unlink()
        copy_file(current/"impact_repair_M12/validation.json", dest/"impact_repair_M12/validation.json")
        selected["review"].append(str((dest/"impact_repair_M12/validation.json").relative_to(review)))
        dump(dest/"impact_snapshot_M12.json", reports[target])
        selected["review"].append(str((dest/"impact_snapshot_M12.json").relative_to(review)))
        entries = [r for r in manifest['b_only_impacts'] if r['target']==target and r['mass']==12]
        if len(entries) != 1: raise RuntimeError(f"Ambiguous M12 manifest entry: {target}")
        entries[0].update(source_directory=str(current), card_sha256=reports[target]['card_sha256'],
                          internal_json_sha256=reports[target]['internal_json_sha256'],
                          validation_directories=['impact_repair_M12'])
        picture = Path(f"figures/limit/{stem}.pdf")
        copy_file(current/(stem+".pdf"), an/picture)
        selected['an'].append(str(picture))
        print(f"[INSTALLED] {target}: {current/(stem+'.pdf')}", flush=True)
    manifest['m12_negative_r_update'] = reports
    dump(manifest_path, manifest); tex.write_text(new_tex)
    dump(work/"selected_paths.json", selected)
    # No complete AN build is attempted in this sparse checkout. Only the three
    # figures and their range sentences change; compile in the full AN checkout.
    for directory, key, repo in ((producer_root,'producer','higgs_combine'),
                                (review,'review','combine_review'), (an,'an','CMS-NPS-26-009_AN')):
        check_repo(directory, repo, 'master' if key=='review' else 'main')
        commit_selected(directory, selected[key], "Update all three M12 B-only impacts with negative-r diagnostics", args.push)
    print(f"[DONE] M12 results synchronised; GitHub push={args.push}; AN compilation not run", flush=True)
    print(f"[BACKUPS AND MANIFESTS] {work}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
