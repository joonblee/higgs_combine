"""Shared-filesystem HTCondor DAG backend for combine_review.py.

Physics commands are captured from the local helper, then partitioned without
changing the dataset, nuisance model, grid or seeds. No observed data are used.
Resume refreshes the exact known 8e52715 snapshot to repair Sub-DAG directory
preparation and enable checked conditional profiles for tiny Gaussian impacts.
Known incomplete repair DAGs also receive tighter seeded retries on resume;
completed workers and their original acceptance criteria are retained.
Known pre-audit scans receive a point-completeness audit and an Asimov plot
label on resume; completed scan chunks and all unrelated jobs are retained.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from uuid import uuid4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def option(command, flag, default=None):
    return command[command.index(flag) + 1] if flag in command else default


def replace(command, flag, value):
    command = list(command)
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])
    return command


def root_name(command):
    method = option(command, "-M")
    name = option(command, "-n", "Test")
    mass = option(command, "-m", "120")
    seed = option(command, "-s")
    # Combine's bin/combine.cpp adds the seed for positive toy counts,
    # nondefault seeds, or --saveToys, regardless of the selected method.
    seed = "123456" if seed is None else seed
    toy_suffix = float(option(command, "-t", "0")) > 0 or seed != "123456" or "--saveToys" in command
    return f"higgsCombine{name}.{method}.mH{mass}" + (f".{seed}" if toy_suffix else "") + ".root"


def capture(review, args, task, card):
    """Capture the existing local command builders; no Combine jobs run here."""
    cfg = copy.copy(args)
    cfg.dry_run = True
    cfg.force = False
    cfg.scan_parallel = cfg.hybrid_parallel = cfg.impact_parallel = 1
    commands = []
    original = review.run_command
    def record(cmd, cwd, **kwargs):
        commands.append({"cmd": list(map(str, cmd)), "cwd": str(Path(cwd).resolve())})
        return 0
    try:
        review.run_command = record
        with contextlib.redirect_stdout(io.StringIO()):
            getattr(review, "task_" + task.replace("-", "_"))(cfg, card)
    finally:
        review.run_command = original
    return commands


def suite_configs(args):
    specs = [(m, "validate", {}) for m in (12, 20, 30, 70)]
    for m in (12, 30, 70):
        specs.extend([(m, "fitdiag", {"inject_r": 1., "r_min": 0.}),
                      (m, "sb-impacts", {"inject_r": 1.})])
    for m, points, low, high in ((20, 600, -50., 100.), (70, 240, -2., 5.)):
        specs.append((m, "correlation", {"r_min": low, "r_max": high}))
        specs.append((m, "scan", {"scan_points": points, "r_min": low, "r_max": high}))
        specs.append((m, "bias", {"r_min": low, "r_max": 100., "bias_injections": None,
                                   "bias_multipliers": [0., 1., 2.]}))
    specs.append((70, "hybrid", {"r_max": 100.}))
    return specs


class Planner:
    def __init__(self, review, batch, manifest):
        self.review, self.batch, self.manifest = review, batch, manifest
        self.jobs = manifest["jobs"]

    def add(self, kind, args, dependencies=(), commands=(), files=(), **extra):
        index = len(self.jobs)
        directory = self.batch / "jobs" / f"job_{index}"
        directory.mkdir(parents=True)
        job = {"index": index, "kind": kind, "dependencies": list(dict.fromkeys(dependencies)),
               "workdir": str(directory), "config": vars(copy.copy(args)),
               "commands": list(commands), "files": list(map(str, files)), **extra}
        self.jobs.append(job)
        return index

    def remote_command(self, kind, args, command, destination, dependencies=(), before=()):
        index = self.add(kind, args, dependencies)
        job = self.jobs[index]
        work = Path(job["workdir"])
        artifact = work / root_name(command)
        job.update(commands=[{"cmd": command, "cwd": str(work)}], files=[str(artifact), str(destination)],
                   copies_before=list(before), copies_after=[[str(artifact), str(destination)]])
        return index

    def final(self, task, args, dependencies, skip=(), merges=(), files=(), copies=()):
        return self.add(task + "_collect", args, dependencies, files=files,
                        helper=task, skip_commands=list(skip), merges=list(merges),
                        copies_before=list(copies))

    def task(self, args, task, card):
        r = self.review
        output = r.outdir(args)
        label = r.mass_label(args.mass)
        if task in ("validate", "correlation", "fitdiag", "workspace"):
            required = {
                "validate": [output / f"validation_M{label}.json"],
                "correlation": [output / f"correlation_{args.target}_M{label}.pdf",
                                output / f"fitdiag_summary_r0_M{label}.json"],
                "fitdiag": [output / f"fitdiag_summary_M{label}.json"],
                "workspace": [r.workspace_path(args)],
            }[task]
            self.add(task, args, helper=task, files=required)
            return
        commands = capture(r, args, task, card)
        if task == "scan":
            original = commands[0]
            command = original["cmd"]
            chunks, files = [], []
            for start in range(0, args.scan_points, args.scan_chunk_size):
                cmd = replace(command, "--firstPoint", start)
                cmd = replace(cmd, "--lastPoint", min(start + args.scan_chunk_size, args.scan_points) - 1)
                index = self.add("scan_chunk", args)
                work = Path(self.jobs[index]["workdir"])
                path = work / root_name(cmd)
                self.jobs[index].update(commands=[{"cmd": cmd, "cwd": str(work)}], files=[str(path)])
                chunks.append(index)
                files.append(str(path))
            merged = output / root_name(command)
            self.final(task, args, chunks, skip=[original],
                       merges=[{"destination": str(merged), "sources": files}],
                       files=[merged, output / f"scan_{args.target}_M{label}_Bonly.pdf",
                              output / f"scan_validation_M{label}.json", output / f"scan_validated_M{label}.root"])
            self.jobs[-1]['scan_validation'] = 'NIsoMuon_scan_validation_v1'
        elif task == "hybrid":
            generation, *grid = commands
            generated = Path(generation["cwd"]) / root_name(generation["cmd"])
            parent = self.remote_command("hybrid_asimov", args, generation["cmd"], generated)
            points = []
            for point in grid:
                dest = Path(point["cwd"]) / root_name(point["cmd"])
                points.append(self.remote_command("hybrid_point", args, point["cmd"], dest, [parent]))
            self.final(task, args, [parent, *points], skip=grid,
                       files=[output / f"hybrid_comparison_M{label}.json"])
        elif task == "sb-impacts":
            initial, fits, _collection = commands
            # Same complete floating-parameter list used by the nominal workflow.
            import limit_workflow
            parameters = sorted(limit_workflow.impact_workspace_parameters(r.workspace_path(args)))
            if not parameters:
                raise RuntimeError("No floating impact parameters in the submitted workspace")
            tag = option(initial["cmd"], "-n")
            mass = r.mass_for_combine(args.mass)
            initial_name = f"higgsCombine_initialFit_{tag}.MultiDimFit.mH{mass}.root"
            first = self.add("impact_initial", args)
            first_work = Path(self.jobs[first]["workdir"])
            first_file = first_work / initial_name
            self.jobs[first].update(commands=[{"cmd": initial["cmd"], "cwd": str(first_work)}],
                                    files=[str(first_file)])
            children, copies = [], [[str(first_file), str(output / initial_name)]]
            for nuisance in parameters:
                cmd = [*fits["cmd"], "--named", nuisance]
                index = self.add("impact_nuisance", args, [first])
                work = Path(self.jobs[index]["workdir"])
                path = work / f"higgsCombine_paramFit_{tag}_{nuisance}.MultiDimFit.mH{mass}.root"
                self.jobs[index].update(commands=[{"cmd": cmd, "cwd": str(work)}],
                    copies_before=[[str(first_file), str(work / initial_name)]], files=[str(path)])
                children.append(index)
                copies.append([str(path), str(output / path.name)])
            self.final(task, args, [first, *children], skip=[initial, fits], copies=copies,
                       files=[output / f"impacts_{args.target}_M{label}_SB.json",
                              output / f"sb_impacts_metadata_M{label}.json"])
        elif task == "bias":
            all_fits, merges = [], []
            for generation, fitting in zip(commands[::2], commands[1::2]):
                generated = Path(generation["cwd"]) / root_name(generation["cmd"])
                parent = self.remote_command("bias_generate", args, generation["cmd"], generated)
                fragments, fit_logs = [], []
                for start in range(1, args.bias_toys + 1, args.bias_chunk_size):
                    index = self.add("bias_chunk", args, [parent])
                    job = self.jobs[index]
                    work = Path(job["workdir"])
                    chunk = []
                    for toy in range(start, min(start + args.bias_chunk_size, args.bias_toys + 1)):
                        cmd = replace(fitting["cmd"], "--pickToy", toy)
                        cmd = replace(cmd, "-n", option(cmd, "-n") + f"_toy{toy:04d}")
                        chunk.append({"cmd": cmd, "cwd": str(work)})
                        fragments.append(str(work / root_name(cmd)))
                    job.update(commands=chunk, files=[str(work / root_name(c["cmd"])) for c in chunk])
                    fit_logs.extend(str(work / f"command_{i}.log") for i in range(len(chunk)))
                    all_fits.append(index)
                dest = Path(fitting["cwd"]) / root_name(fitting["cmd"])
                tag = re.search(r"_r([^.]*)\.MultiDimFit", dest.name).group(1)
                merges.append({"destination": str(dest), "sources": fragments,
                               "logs": fit_logs, "log_destination": str(dest.parent / f"bias_fit_r{tag}.log")})
            final = self.final(task, args, all_fits, merges=merges,
                        files=[output / "bias" / f"bias_summary_{args.target}_M{label}.json"])
            if self.manifest["suite"]:
                add_parallel_audit(self, args, [final])
        else:
            raise RuntimeError(f"Unsupported Condor task: {task}")


def prepare(review, args):
    if args.repair_sb_impacts or args.force:
        raise RuntimeError("New Condor batches do not accept --force/--repair-sb-impacts; use --stage resume for failed jobs")
    if args.mass is None and not args.review_suite:
        raise RuntimeError("Supply a mass, or --review-suite")
    if args.mass is not None and args.review_suite:
        raise RuntimeError("--review-suite chooses its representative masses; omit the positional mass")
    for name in ("condor_memory", "scan_chunk_size", "bias_chunk_size", "bias_toys", "hybrid_toys", "hybrid_grid_points"):
        if getattr(args, name) < 1:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    if args.condor_max_jobs < 0 or args.scan_points < 5 or args.r_max <= 0:
        raise RuntimeError("Invalid Condor cap, scan point count or rMax")
    if args.inject_r is not None and (not math.isfinite(args.inject_r) or args.inject_r < 0 or args.inject_r >= args.r_max):
        raise RuntimeError("Injection must be finite, nonnegative and below rMax")
    cmssw = Path(args.condor_cmssw_base or os.environ.get("CMSSW_BASE", "")).resolve()
    arch = args.condor_scram_arch or os.environ.get("SCRAM_ARCH")
    if not arch or not (cmssw / "src").is_dir():
        raise RuntimeError("Initialise the Combine CMSSW environment or supply --condor-cmssw-base/--condor-scram-arch")
    token = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    batch = Path(args.condor_dir or f"condor_jobs/review_{token}").resolve()
    if re.search(r'[\s$"\\]', str(batch)):
        raise RuntimeError("Condor batch paths must not contain whitespace, dollar signs, quotes or backslashes")
    if batch.exists():
        raise RuntimeError(f"Batch already exists: {batch}; use a new --condor-dir or --stage resume")
    for value in (args.condor_requirements, args.condor_filesystem_domain):
        if value and any(c in value for c in '\n\r'):
            raise RuntimeError("Condor expressions must be single-line")
    specs = suite_configs(args) if args.review_suite else [(args.mass, t, {}) for t in args.tasks]
    # Preflight source cards/limits before creating the batch; this backend is for counting cards.
    source_cards, source_limits = {}, {}
    for mass, task, changes in specs:
        cfg = copy.copy(args)
        cfg.mass = mass
        card = review.card_path(cfg)
        for line in card.read_text().splitlines():
            if line.strip().startswith("shapes "):
                raise RuntimeError("Condor review currently supports self-contained counting cards only")
        source_cards[mass] = card
        if task in ("bias", "hybrid") or (task == "sb-impacts" and changes.get("inject_r", args.inject_r) is None):
            source = review.expected_limit_json(cfg)
            review.existing_expected_r95(cfg)
            source_limits[str(source)] = source
    if "bias" in [s[1] for s in specs] and args.review_suite and args.bias_toys < 2:
        raise RuntimeError("Bias interval audits require at least two toys")
    batch.mkdir(parents=True)
    inputs = {}
    script_dir = Path(review.__file__).resolve().parent
    for name in ("combine_review.py", "review_condor.py", "review_scan.py", "review_bias_parallel.py", "limit_workflow.py", "audit_bias_intervals.py",
                 "repair_bias_intervals.py", "validate_bias_repair.py"):
        source = script_dir / name
        if not source.is_file():
            raise RuntimeError(f"Required review source missing: {source}")
        dest = batch / name
        shutil.copy2(source, dest)
        inputs[str(dest)] = digest(dest)
    for mass, source in source_cards.items():
        dest = batch / "cards" / args.target / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        inputs[str(dest)] = digest(dest)
    limit_dest = None
    for source in source_limits.values():
        limit_dest = batch / "expected_limits.json"
        shutil.copy2(source, limit_dest)
        inputs[str(limit_dest)] = digest(limit_dest)
    manifest = {"schema": "NIsoMuon_review_condor_v1", "batch_id": token, "batch": str(batch),
                "output_base": str(Path(args.output_base).resolve()), "suite": args.review_suite,
                "cmssw_base": str(cmssw), "scram_arch": arch, "inputs": inputs, "jobs": [],
                "memory": args.condor_memory, "max_jobs": args.condor_max_jobs,
                "requirements": args.condor_requirements, "filesystem_domain": args.condor_filesystem_domain}
    planner = Planner(review, batch, manifest)
    workspaces = set()
    for mass, task, changes in specs:
        cfg = copy.copy(args)
        cfg.mass = mass
        cfg.__dict__.update(changes)
        cfg.card_base = str(batch / "cards")
        cfg.output_base = str(batch / "results")
        cfg.expected_json = str(limit_dest) if limit_dest else None
        cfg.backend, cfg.stage, cfg.review_suite = "local", "run", False
        cfg.dry_run = cfg.force = cfg.condor_dry_run = False
        cfg.scan_parallel = cfg.hybrid_parallel = cfg.impact_parallel = 1
        cfg.condor_worker = cfg.condor_index = None
        cfg.tasks = [task]
        card = review.card_path(cfg)
        if task != "validate" and mass not in workspaces:
            workspace = review.ensure_workspace(cfg, card)
            inputs[str(workspace)] = digest(workspace)
            workspaces.add(mass)
        review.write_metadata(cfg, card)
        planner.task(cfg, task, card)
    dump(batch / "manifest.json", manifest)
    write_submission(batch, manifest, list(range(len(manifest["jobs"]))), "workflow.dag")
    print(f"[CONDOR] {len(manifest['jobs'])} single-CPU jobs; batch={batch}")
    print(f"[CONDOR] collect: python3 combine_review.py --stage collect --condor-dir {shlex.quote(str(batch))}")
    print(f"[CONDOR] resume:  python3 combine_review.py --stage resume --condor-dir {shlex.quote(str(batch))}")
    if not (args.condor_dry_run or args.dry_run):
        submit(batch, manifest, "workflow.dag")
    else:
        print("[CONDOR] Prepared only: workspaces and snapshots created; no jobs submitted")
    return batch


def write_submission(batch, manifest, indices, dag_name):
    wrapper = batch / "run_job.sh"
    wrapper.write_text("\n".join([
        "#!/bin/bash", "set -eo pipefail", "export LC_ALL=C ROOT_HIST=0",
        "source /cvmfs/cms.cern.ch/cmsset_default.sh",
        "export SCRAM_ARCH=" + shlex.quote(manifest["scram_arch"]),
        "cd " + shlex.quote(str(Path(manifest["cmssw_base"]) / "src")),
        'eval "$(scramv1 runtime -sh)"',
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "cd " + shlex.quote(str(batch)),
        "exec python3 -u " + shlex.quote(str(batch / "combine_review.py")) +
        " --condor-worker " + shlex.quote(str(batch / "manifest.json")) + ' --condor-index "$1"', ""]))
    wrapper.chmod(0o755)
    lines = [f"executable = {wrapper}", f"initialdir = {batch}", "arguments = $(jobindex)",
             "universe = vanilla", "getenv = True", "should_transfer_files = NO",
             "request_cpus = 1", f"request_memory = {manifest['memory']}MB", "request_disk = 1GB",
             "jobbatchname = NIsoMuon_review_" + manifest["batch_id"],
             "output = jobs/job_$(jobindex)/condor.out", "error = jobs/job_$(jobindex)/condor.err",
             "log = condor.log"]
    if manifest.get("requirements"):
        lines.append("requirements = " + manifest["requirements"])
    if manifest.get("filesystem_domain"):
        lines.append("+FileSystemDomain = " + json.dumps(manifest["filesystem_domain"]))
    lines.append("queue 1\n")
    (batch / "submit.jds").write_text("\n".join(lines))
    selected = set(indices)
    dag = []
    for index in indices:
        job = manifest["jobs"][index]
        if job.get("subdag"):
            # DAGMan validates DIR while parsing the outer DAG, before the
            # audit parent has generated the inner DAG description.
            Path(job['subdag']).parent.mkdir(parents=True, exist_ok=True)
            dag.append(f"SUBDAG EXTERNAL J{index} {job['subdag']} DIR {Path(job['subdag']).parent}")
            # Lazy generation is required: the audit parent creates this DAG.
        else:
            dag.extend([f"JOB J{index} submit.jds", f'VARS J{index} jobindex="{index}"'])
        parents = [i for i in job["dependencies"] if i in selected]
        if parents:
            dag.append("PARENT " + " ".join(f"J{i}" for i in parents) + f" CHILD J{index}")
    (batch / dag_name).write_text("\n".join(dag) + "\n")
    config = []
    if any(j.get('subdag') for j in manifest['jobs']):
        config.append('DAGMAN_GENERATE_SUBDAG_SUBMITS = True')
    if manifest.get("parent_completions") is not None and manifest["max_jobs"]:
        config.append(f"DAGMAN_MAX_JOBS_SUBMITTED = {manifest['max_jobs']}")
    if config:
        (batch / "dagman.config").write_text('\n'.join(config) + '\n')
        with (batch / dag_name).open("a") as stream:
            stream.write("CONFIG dagman.config\n")


def submit(batch, manifest, dag):
    command = ["condor_submit_dag"]
    if any(j.get('subdag') for j in manifest['jobs']):
        command.append('-no_recurse')
    if manifest["max_jobs"]:
        command += ["-maxjobs", str(manifest["max_jobs"])]
    subprocess.run([*command, dag], cwd=batch, check=True)


def load_batch(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != "NIsoMuon_review_condor_v1" or manifest.get("batch") != str(path):
        raise RuntimeError("Invalid or relocated review batch; keep the submitted shared-storage path")
    for name, fingerprint in manifest["inputs"].items():
        if not Path(name).is_file() or digest(name) != fingerprint:
            raise RuntimeError(f"Submitted input/code/workspace changed: {name}")
    return path, manifest


def complete(manifest, index, cache=None):
    cache = {} if cache is None else cache
    if index in cache:
        return cache[index]
    job = manifest["jobs"][index]
    if job.get("subdag"):
        parents = {str(i): complete(manifest, i, cache)["completion_id"] for i in job["dependencies"]}
        child_path = Path(job["subdag"]).parent
        if not (child_path / "manifest.json").is_file():
            raise RuntimeError(f"job {index} {job['kind']}: repair DAG not prepared")
        _, child = load_batch(child_path)
        if child.get("parent_completions") != parents:
            raise RuntimeError(f"job {index}: stale repair DAG after audit reran")
        children = {str(j['index']): complete(child, j['index'])['completion_id'] for j in child['jobs']}
        status = {"completion_id": hashlib.sha256(json.dumps([parents, children], sort_keys=True).encode()).hexdigest()}
        cache[index] = status
        return status
    path = Path(job["workdir"]) / "status.json"
    if not path.is_file():
        raise RuntimeError(f"job {index} {job['kind']}: pending")
    status = json.loads(path.read_text())
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if not status.get("success"):
        raise RuntimeError(f"job {index} {job['kind']}: {status.get('error', 'failed without an error record')}; logs: {job['workdir']}")
    if status.get("batch_id") != manifest["batch_id"] or status.get("job_sha256") != fingerprint:
        raise RuntimeError(f"job {index} {job['kind']}: stale status; logs: {job['workdir']}")
    if not status.get("outputs"):
        raise RuntimeError(f"job {index}: no output fingerprints")
    for name, value in status["outputs"].items():
        if not Path(name).is_file() or not Path(name).stat().st_size or digest(name) != value:
            raise RuntimeError(f"job {index}: output missing or changed: {name}")
    for parent in job["dependencies"]:
        previous = complete(manifest, parent, cache)
        if status.get("dependencies", {}).get(str(parent)) != previous["completion_id"]:
            raise RuntimeError(f"job {index}: stale after dependency {parent} reran")
    cache[index] = status
    return status


def copy_pairs(pairs):
    for source, target in pairs:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def execute(review, manifest, index):
    job = manifest["jobs"][index]
    work = Path(job["workdir"])
    status = {"success": False, "batch_id": manifest["batch_id"], "completion_id": uuid4().hex,
              "job_sha256": hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()}
    cfg = argparse.Namespace(**job["config"])
    try:
        cache = {}
        status["dependencies"] = {str(i): complete(manifest, i, cache)["completion_id"] for i in job["dependencies"]}
        dump(work / "status.json", status)
        # A retry must not accept files left by an interrupted command. Pinned
        # input workspaces are immutable and are never removed here.
        for name in job["files"]:
            path = Path(name)
            if name not in manifest.get("inputs", {}) and path.is_file():
                path.unlink()
        copy_pairs(job.get("copies_before", []))
        for i, command in enumerate(job["commands"]):
            if review.run_command(command["cmd"], Path(command["cwd"]), log=work / f"command_{i}.log"):
                raise RuntimeError(f"Command {i} failed; see {work}")
        for merge in job.get("merges", []):
            destination = Path(merge["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if review.run_command(["hadd", "-f", str(destination), *merge["sources"]], work, log=work / (destination.stem + ".log")):
                raise RuntimeError("ROOT merge failed")
            if merge.get("logs"):
                with Path(merge["log_destination"]).open("w") as output:
                    for log in merge["logs"]:
                        output.write(Path(log).read_text())
        copy_pairs(job.get("copies_after", []))
        if job.get("helper"):
            review._REVIEW_COMMAND_CACHE = {review.command_key(c["cmd"], c["cwd"]) for c in job.get("skip_commands", [])}
            try:
                helper = job["helper"]
                if helper == "workspace":
                    review.ensure_workspace(cfg, review.card_path(cfg))
                else:
                    getattr(review, "task_" + helper.replace("-", "_"))(cfg, review.card_path(cfg))
            finally:
                review._REVIEW_COMMAND_CACHE = set()
        if job.get("audit"):
            run_audit(review, manifest, cfg, work)
        if job.get("audit_prepare"):
            run_audit_prepare(review, manifest, job, cfg, work, status["completion_id"])
        if job.get("bias_one") or job.get("bias_merge"):
            import review_bias_parallel
            review_bias_parallel.execute(manifest, job)
        files = job["files"]
        for name in files:
            if not Path(name).is_file() or not Path(name).stat().st_size:
                raise RuntimeError(f"Required output missing or empty: {name}")
        status.update(success=True, outputs={name: digest(name) for name in files})
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dump(work / "status.json", status)


def run_audit(review, manifest, args, work):
    output = review.outdir(args)
    checks = output / "bias_checks"
    checks.mkdir(parents=True, exist_ok=True)
    scripts = Path(manifest["batch"])
    command = ["python3", str(scripts / "audit_bias_intervals.py"), "--source-dir", str(output),
               "--target", args.target, "--mass", review.mass_label(args.mass), "--expected-toys", str(args.bias_toys)]
    log = checks / "audit.log"
    code = review.run_command(command, checks, log=log)
    if code not in (0, 2):
        raise RuntimeError(f"Bias audit input failure: {log}")
    matches = re.findall(r"^\[SAVED\] (.+)$", log.read_text(), re.M)
    if not matches:
        raise RuntimeError(f"No audit report produced: {log}")
    audit_path = Path(matches[-1])
    audit = json.loads(audit_path.read_text())
    result = {"passed": bool(audit["passed"]), "audit": str(audit_path), "repair": None}
    if code == 2:
        command = ["python3", str(scripts / "repair_bias_intervals.py"), "--audit", str(audit_path),
                   "--source-dir", str(output), "--target", args.target, "--mass", review.mass_label(args.mass)]
        repair_log = checks / "repair.log"
        if review.run_command(command, checks, log=repair_log):
            raise RuntimeError(f"Bias repair failed: {repair_log}")
        paths = re.findall(r"^\[OUTPUT\] (.+)$", repair_log.read_text(), re.M)
        if not paths:
            raise RuntimeError("Missing bias repair output path")
        validation = Path(paths[-1]) / "validation.json"
        result.update(passed=bool(json.loads(validation.read_text())["passed"]), repair=str(validation))
    dump(checks / "review_bias_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Bias intervals did not pass audit/repair")


def active_batch(batch):
    """Prevent concurrent workers or multiple DAG managers during recovery."""
    constraint = "JobStatus != 4 && regexp(" + json.dumps("^" + re.escape(str(batch)) + "(/|$)") + ", Iwd)"
    result = subprocess.run(["condor_q", "-constraint", constraint, "-af", "ClusterId", "ProcId"],
                            text=True, capture_output=True, check=True)
    if result.stdout.strip():
        raise RuntimeError("This batch still has queued/running/held jobs or a DAG manager; wait or remove that batch before resume")


def collect(batch, manifest):
    cache = {}
    errors = []
    for job in manifest["jobs"]:
        try:
            if job['kind'] == 'scan_collect' and job.get('scan_validation') != 'NIsoMuon_scan_validation_v1':
                raise RuntimeError(f"job {job['index']}: scan point audit missing; run --stage resume first")
            complete(manifest, job["index"], cache)
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("Review batch incomplete; nothing collected:\n  " + "\n  ".join(errors[:12]))
    destination = Path(manifest["output_base"])
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "review_collection.json"
    results = batch / "results"
    for target in sorted(results.iterdir()):
        for source in sorted(target.iterdir()):
            if not source.is_dir():
                continue
            dest = destination / target.name / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            stage = dest.parent / ("." + dest.name + "." + uuid4().hex)
            shutil.copytree(source, stage)
            if dest.exists():
                backup = batch / "previous_outputs" / target.name / (source.name + "." + uuid4().hex)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(backup))
            stage.rename(dest)
    dump(marker, {"batch_id": manifest["batch_id"], "manifest": str(batch / "manifest.json"),
                  "n_jobs": len(manifest["jobs"]), "mode": "blind", "suite": manifest["suite"]})
    print(f"[COLLECT] Review outputs saved: {destination}")


# Only this published backend is migrated: its commands are retained verbatim.
LEGACY_BIAS_BACKEND_SHA256 = "5a93b7f6e3a67c0fcba33815cbe53208745706509b0b250114f5c101b3885de6"
LEGACY_BIAS_HELPER_SHA256 = "42c38cbf2f61a14e35d668c5de9fd384086122f0a5923a17c99de3080f0cca2a"


def upgrade_backend(review, batch, manifest):
    """Repair only the known output-name contract; retain original fit commands.

    An old chunk is recovered only if all commands returned zero and its sole
    recorded failure was the missing expected filename. This establishes worker
    completion, not fit quality: downstream interval audits remain required.
    """
    old_module = batch / "review_condor.py"
    old_helper = batch / "combine_review.py"
    current_module = Path(__file__).resolve()
    current_helper = Path(review.__file__).resolve()
    if digest(old_module) == digest(current_module) and digest(old_helper) == digest(current_helper):
        print("[UPGRADE] Batch already uses this backend")
        return manifest
    if digest(old_module) != LEGACY_BIAS_BACKEND_SHA256 or digest(old_helper) != LEGACY_BIAS_HELPER_SHA256:
        raise RuntimeError("--upgrade-backend only supports the published v1 bias-filename bug; this batch snapshot is different")
    updated = copy.deepcopy(manifest)
    mapping = {}
    for job in updated["jobs"]:
        if job["kind"] != "bias_chunk":
            continue
        if len(job["commands"]) != len(job["files"]):
            raise RuntimeError("Unexpected bias chunk command/output count")
        names = []
        for command, old in zip(job["commands"], job["files"]):
            new = str(Path(command["cwd"]) / root_name(command["cmd"]))
            if new != old:
                mapping[old] = new
            names.append(new)
        if len(names) != len(job["files"]):
            raise RuntimeError("Unexpected bias chunk layout")
        job["files"] = names
    if not mapping:
        raise RuntimeError("No legacy bias chunks found; no batch files changed")
    for job in updated["jobs"]:
        for merge in job.get("merges", []):
            merge["sources"] = [mapping.get(name, name) for name in merge["sources"]]
    archive = batch / ("backend_upgrade_" + uuid4().hex[:8])
    archive.mkdir()
    backups = [(batch / "manifest.json", archive / "manifest.json"),
               (old_module, archive / old_module.name), (old_helper, archive / old_helper.name)]
    recovered = []
    cache = {}
    for job, old_job in zip(updated["jobs"], manifest["jobs"]):
        if job["kind"] != "bias_chunk":
            continue
        path = Path(job["workdir"]) / "status.json"
        if not path.is_file():
            continue
        status = json.loads(path.read_text())
        old_fingerprint = hashlib.sha256(json.dumps(old_job, sort_keys=True).encode()).hexdigest()
        if (status.get("success") or status.get("batch_id") != manifest["batch_id"]
            or status.get("job_sha256") != old_fingerprint
            or not status.get("error", "").startswith("RuntimeError: Required output missing or empty:")):
            continue
        # Reaching the original output check means every chunk command exited 0.
        if not all(Path(name).is_file() and Path(name).stat().st_size for name in job["files"]):
            continue
        if not all((Path(job["workdir"]) / f"command_{i}.log").is_file() for i in range(len(job["commands"]))):
            continue
        try:
            for parent in job["dependencies"]:
                previous = complete(manifest, parent, cache)
                if status.get("dependencies", {}).get(str(parent)) != previous["completion_id"]:
                    raise RuntimeError("stale toy generation")
        except (RuntimeError, OSError, ValueError):
            continue
        backups.append((path, archive / f"status_job_{job['index']}.json"))
        repaired = copy.deepcopy(status)
        repaired["previous_error"] = repaired.pop("error")
        repaired.update(success=True, recovered_output_names=True,
            job_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
            outputs={name: digest(name) for name in job["files"]})
        recovered.append((path, repaired))
    for source, destination in backups:
        shutil.copy2(source, destination)
    try:
        shutil.copy2(current_module, old_module)
        shutil.copy2(current_helper, old_helper)
        for path in (old_module, old_helper):
            updated["inputs"][str(path)] = digest(path)
        updated["backend_upgrade"] = {"archive": str(archive), "recovered_bias_chunks": len(recovered),
                                      "reason": "Combine appends default seed when -t is positive"}
        for path, status in recovered:
            dump(path, status)
        dump(batch / "manifest.json", updated)
    except Exception:
        for destination, source in reversed(backups):
            shutil.copy2(source, destination)
        raise
    print(f"[UPGRADE] Fixed {len(mapping)} toy filenames; recovered {len(recovered)} completed bias chunks; backup={archive}")
    return updated


def add_parallel_audit(planner, args, dependencies, replace_index=None):
    output = planner.review.outdir(args)
    child = planner.batch / 'bias_repairs' / args.target / ('M' + planner.review.mass_label(args.mass))
    audit = planner.add('bias_audit_prepare', args, dependencies, audit_prepare=True,
        repair_dir=str(child), files=[output / 'bias_checks' / 'audit_reference.json', child / 'manifest.json'])
    options = dict(kind='bias_repair_dag', dependencies=[audit], subdag=str(child / 'workflow.dag'),
                   files=[str(output / 'bias_checks' / 'review_bias_validation.json')], commands=[], config=vars(copy.copy(args)))
    if replace_index is None:
        index = planner.add('bias_repair_dag', args, [audit], subdag=str(child / 'workflow.dag'), files=options['files'])
    else:
        index = replace_index
        old = planner.jobs[index]
        planner.jobs[index] = {'index': index, 'workdir': old['workdir'], **options}
    return audit, index


def run_audit_prepare(review, manifest, job, args, work, completion_id):
    output = review.outdir(args)
    checks = output / 'bias_checks'
    checks.mkdir(parents=True, exist_ok=True)
    command = ['python3', str(Path(manifest['batch']) / 'audit_bias_intervals.py'),
               '--source-dir', str(output), '--target', args.target,
               '--mass', review.mass_label(args.mass), '--expected-toys', str(args.bias_toys)]
    log = checks / 'audit.log'
    status = review.run_command(command, checks, log=log)
    if status not in (0, 2):
        raise RuntimeError(f'Bias audit input failure: {log}')
    paths = re.findall(r'^\[SAVED\] (.+)$', log.read_text(), re.M)
    if not paths:
        raise RuntimeError('Missing structural bias audit report')
    report = Path(paths[-1])
    import review_bias_parallel
    review_bias_parallel.prepare(review, manifest, job, args, report, completion_id)
    dump(checks / 'audit_reference.json', {'audit': str(report), 'sha256': digest(report)})


SERIAL_BACKEND_SHA256 = '06c23a7cdc2c63969ee61c3dd31cf33393987408080453b51d03b9dedbaab69e'
SERIAL_HELPER_SHA256 = '7a0a82a736c18d2f165123889c5c9df476439dcc89ae122556f0e7cadb768956'
SERIAL_REPAIR_SHA256 = 'd93809b8c8c1921f07995f3b40dccb472a7298190b1416b12403710acd79f9fe'


def parallelize_bias(review, batch, manifest):
    # The caller first checks that this batch and all its child DAGs are inactive.
    original_module = batch / 'review_condor.py'
    original_helper = batch / 'combine_review.py'
    original_repair = batch / 'repair_bias_intervals.py'
    fingerprints = (digest(original_module), digest(original_helper), digest(original_repair))
    supported = {(SERIAL_BACKEND_SHA256, SERIAL_HELPER_SHA256, SERIAL_REPAIR_SHA256),
                 (digest(Path(__file__)), digest(Path(review.__file__)), SERIAL_REPAIR_SHA256)}
    if fingerprints not in supported:
        if any(j.get('subdag') for j in manifest['jobs']):
            print('[BIAS PARALLEL] Batch already uses repair sub-DAGs')
            return manifest
        raise RuntimeError('--parallel-bias-repair supports the known 910d62a snapshot; upgrade the v1 filename bug separately first')
    updated = copy.deepcopy(manifest)
    planner = Planner(review, batch, updated)
    selected = []
    for job in manifest['jobs']:
        if job.get('audit'):
            try:
                complete(manifest, job['index'])
                continue
            except (RuntimeError, OSError, ValueError):
                selected.append(job)
    if not selected:
        print('[BIAS PARALLEL] No incomplete serial audits; completed results retained')
        return manifest
    for job in selected:
        add_parallel_audit(planner, argparse.Namespace(**job['config']), job['dependencies'], job['index'])
    archive = batch / ('bias_parallel_upgrade_' + uuid4().hex[:8])
    archive.mkdir()
    names = ('combine_review.py', 'review_condor.py', 'review_bias_parallel.py', 'repair_bias_intervals.py')
    original_sources = []
    for name in names:
        path = batch / name
        if path.is_file():
            shutil.copy2(path, archive / name)
            original_sources.append(name)
    shutil.copy2(batch / 'manifest.json', archive / 'manifest.json')
    try:
        for name in names:
            source = Path(review.__file__).resolve().parent / name
            shutil.copy2(source, batch / name)
            updated['inputs'][str(batch / name)] = digest(batch / name)
        updated['parallel_bias_upgrade'] = {'archive': str(archive), 'audit_indices': [j['index'] for j in selected]}
        dump(batch / 'manifest.json', updated)
    except Exception:
        for name in names:
            if name in original_sources:
                shutil.copy2(archive / name, batch / name)
            elif (batch / name).exists():
                (batch / name).unlink()
        shutil.copy2(archive / 'manifest.json', batch / 'manifest.json')
        raise
    print(f'[BIAS PARALLEL] Converted {len(selected)} incomplete audits; completed fit jobs retained; backup={archive}')
    return updated


def refresh_child_dags(batch, manifest, indices):
    for index in indices:
        job = manifest['jobs'][index]
        if not job.get('subdag'):
            continue
        # If its audit has not completed, it will create the child DAG later.
        try:
            for parent in job['dependencies']:
                complete(manifest, parent)
        except (RuntimeError, OSError, ValueError):
            continue
        child_path, child = load_batch(Path(job['subdag']).parent)
        pending, cache = [], {}
        for node in child['jobs']:
            try:
                complete(child, node['index'], cache)
            except (RuntimeError, OSError, ValueError):
                pending.append(node['index'])
        if pending:
            # Resubmit lazily as a fresh child DAG, retaining its physics
            # outputs/statuses while archiving old DAGMan scheduling files.
            archive = child_path / ('scheduling_archive_' + uuid4().hex[:8])
            archive.mkdir()
            for old in child_path.glob('workflow.dag.*'):
                if old.is_file():
                    old.rename(archive / old.name)
            write_submission(child_path, child, pending, 'workflow.dag')


# Exact published 8e52715 snapshot: compatible orchestration/profile repair.
COMPATIBLE_REVIEW_PATCH = {
    'combine_review.py': 'a474ca889e693bccda97b4b155b1f009e934250f30f748cd908869f95d242ff4',
    'review_condor.py': '936cb13e8a91ac4669a28a352930d0568acc761e6703ff68979df48cbd506c7a',
    'review_bias_parallel.py': '29e9777e444b689359dcce8d44af86d16f28f93f2c602471e533fb072d5af7e3',
    'limit_workflow.py': '823d838417b5f64c24b307757c27b3ee2e73b009e48e6eab0710a86363bba5ae',
}


def refresh_compatible_review_patch(review, batch, manifest):
    """Refresh this known failed snapshot without changing cards or fit configs."""
    if not all((batch / name).is_file() and digest(batch / name) == value
               for name, value in COMPATIBLE_REVIEW_PATCH.items()):
        return manifest
    archive = batch / ('review_patch_' + uuid4().hex[:8])
    archive.mkdir()
    shutil.copy2(batch / 'manifest.json', archive / 'manifest.json')
    for name in COMPATIBLE_REVIEW_PATCH:
        shutil.copy2(batch / name, archive / name)
    updated = copy.deepcopy(manifest)
    try:
        for name in COMPATIBLE_REVIEW_PATCH:
            source = Path(review.__file__).resolve().parent / name
            shutil.copy2(source, batch / name)
            updated['inputs'][str(batch / name)] = digest(batch / name)
        updated['compatible_review_patch'] = {
            'archive': str(archive), 'patch': 'create-subdag-dir-and-profile-tiny-gaussian',
            'preserves_cards_workspaces_toys_and_job_configs': True}
        dump(batch / 'manifest.json', updated)
    except Exception:
        for name in COMPATIBLE_REVIEW_PATCH:
            shutil.copy2(archive / name, batch / name)
        shutil.copy2(archive / 'manifest.json', batch / 'manifest.json')
        raise
    print(f'[REVIEW PATCH] Updated known 8e52715 snapshot; completed jobs retained; backup={archive}')
    return updated


# The published per-toy helper in 8e52715 and 7a6410d.
PRE_REFINEMENT_HELPER = '21924587c679f7f5d930e24dc5929bb76e11e9bec023135aef57910463ab000c'


def refresh_bias_refinement(review, batch, manifest):
    """Upgrade only retry initialisation in inactive, incomplete repair DAGs.

    Job configurations, completion UUIDs and accepted numerical criteria stay
    identical. The audit parent's child-manifest fingerprint is refreshed only
    for this verified code-only change. Every replaced file is backed up.
    """
    if not any(j.get('subdag') for j in manifest['jobs']):
        return manifest
    source = Path(review.__file__).resolve().parent / 'repair_bias_intervals.py'
    replacement = source.read_bytes()
    current = digest(source)
    replacements = {}
    def encoded(value):
        return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    def helper(path):
        fingerprint = digest(path)
        if fingerprint == current:
            return False
        if fingerprint != PRE_REFINEMENT_HELPER:
            raise RuntimeError(f'Unknown bias-repair helper snapshot: {path}')
        replacements[path] = replacement
        return True
    updated = copy.deepcopy(manifest)
    if helper(batch / 'repair_bias_intervals.py'):
        updated['inputs'][str(batch / 'repair_bias_intervals.py')] = current
    children = []
    for job in manifest['jobs']:
        if not job.get('subdag'):
            continue
        child_path = Path(job['subdag']).parent
        if not (child_path / 'manifest.json').exists():
            continue
        try:
            complete(manifest, job['index'])
            continue
        except (RuntimeError, OSError, ValueError):
            pass
        # Stale audits require their own recovery, not a helper-only patch.
        for index in job['dependencies']:
            complete(manifest, index)
        _, child = load_batch(child_path)
        if not helper(child_path / 'repair_bias_intervals.py'):
            continue
        child = copy.deepcopy(child)
        child['inputs'][str(child_path / 'repair_bias_intervals.py')] = current
        content = encoded(child)
        replacements[child_path / 'manifest.json'] = content
        for index in job['dependencies']:
            status_path = Path(manifest['jobs'][index]['workdir']) / 'status.json'
            status = json.loads(status_path.read_text())
            name = str(child_path / 'manifest.json')
            if name not in status.get('outputs', {}):
                raise RuntimeError(f'Audit has no child-manifest fingerprint: {status_path}')
            status['outputs'][name] = hashlib.sha256(content).hexdigest()
            replacements[status_path] = encoded(status)
        children.append(str(child_path))
    if not replacements:
        return manifest
    archive = batch / ('bias_refinement_upgrade_' + uuid4().hex[:8])
    archive.mkdir()
    updated['bias_refinement_patch'] = {'archive': str(archive), 'children': children,
        'preserves_job_configs_and_acceptance_checks': True}
    replacements[batch / 'manifest.json'] = encoded(updated)
    backups = {}
    for index, path in enumerate(replacements):
        backups[path] = archive / str(index)
        shutil.copy2(path, backups[path])
    dump(archive / 'files.json', {str(p): str(b) for p, b in backups.items()})
    try:
        for path, content in replacements.items():
            temporary = path.with_name(path.name + '.refine.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)
    except Exception:
        for path, saved in backups.items():
            shutil.copy2(saved, path)
        raise
    print(f'[BIAS REFINE] Updated retry helper in {len(children)} incomplete repair DAGs; '
          f'completed jobs retained; backup={archive}')
    return updated


# Published pre-audit helpers (7a6410d on the user's batch, or 1f1485a).
PRE_SCAN_AUDIT_HELPERS = {
    'combine_review.py': {'a474ca889e693bccda97b4b155b1f009e934250f30f748cd908869f95d242ff4'},
    'review_condor.py': {'178ba99ee48db49ffd45dae29e84a91d0c39ce701888281a5da67e495e11027c',
                         '2a93744c22019fe3a9b1ee71013b419607f4e0ca2091255b2f78a3020081a15a'},
}


def refresh_scan_validation(review, batch, manifest):
    """Invalidate only old scan collectors, preserving their verified chunks."""
    selected = [j for j in manifest['jobs'] if j['kind'] == 'scan_collect'
                and j.get('scan_validation') != 'NIsoMuon_scan_validation_v1']
    if not selected:
        return manifest
    source = Path(review.__file__).resolve().parent
    for name, known in PRE_SCAN_AUDIT_HELPERS.items():
        if digest(batch / name) not in known | {digest(source / name)}:
            raise RuntimeError(f'Unknown pre-audit scan helper snapshot: {batch / name}')
    module = batch / 'review_scan.py'
    if module.exists() and digest(module) != digest(source / module.name):
        raise RuntimeError(f'Unknown scan audit module: {module}')
    updated = copy.deepcopy(manifest)
    for old in selected:
        job = updated['jobs'][old['index']]
        label = review.mass_label(job['config']['mass'])
        output = review.outdir(argparse.Namespace(**job['config']))
        job['scan_validation'] = 'NIsoMuon_scan_validation_v1'
        job['files'].extend(str(output / name) for name in
                            (f'scan_validation_M{label}.json', f'scan_validated_M{label}.root')
                            if str(output / name) not in job['files'])
    archive = batch / ('scan_validation_upgrade_' + uuid4().hex[:8])
    archive.mkdir()
    names = ('combine_review.py', 'review_condor.py', 'review_scan.py')
    shutil.copy2(batch / 'manifest.json', archive / 'manifest.json')
    existing = {name for name in names if (batch / name).exists()}
    for name in existing:
        shutil.copy2(batch / name, archive / name)
    for job in selected:
        status = Path(job['workdir']) / 'status.json'
        if status.exists():
            shutil.copy2(status, archive / f"job_{job['index']}_status.json")
    try:
        for name in names:
            shutil.copy2(source / name, batch / name)
            updated['inputs'][str(batch / name)] = digest(batch / name)
        updated['scan_validation_patch'] = {'archive': str(archive),
            'collectors': [j['index'] for j in selected],
            'preserves_scan_chunks_cards_workspaces_and_unrelated_jobs': True}
        dump(batch / 'manifest.json', updated)
    except Exception:
        for name in names:
            if name in existing:
                shutil.copy2(archive / name, batch / name)
            elif (batch / name).exists():
                (batch / name).unlink()
        shutil.copy2(archive / 'manifest.json', batch / 'manifest.json')
        raise
    print(f"[SCAN AUDIT] Updated {len(selected)} scan collectors; completed chunks and "
          f"unrelated jobs retained; backup={archive}")
    return updated


PRE_SCAN_FIT_RESULT_MODULE = 'd2d7efa7e0fde37ec4346243000e4899924f3a98abf33ecaf7c4eb76fdd15c85'
PRE_SCAN_FIT_RESULT_BACKEND = 'c3a410139887efe968d46c7577d96cc232ba68e34ff28be66c70cdf08137b062'
PRE_SCAN_CONTINUATION_MODULE = 'e884c7f4a004f9e782e58299a6471bfa59dc0be3527d93b2401e41b6486d3d87'
PRE_SCAN_CONTINUATION_BACKEND = 'c962fddc36a7e896b1ccfbc88689226f024595296df7a6f763b4b7ae603cc58d'


def refresh_scan_fit_result(review, batch, manifest):
    """Repair the known grid saveFitResult assumption without invalidating successes."""
    if not any(j['kind'] == 'scan_collect' for j in manifest['jobs']):
        return manifest
    source = Path(review.__file__).resolve().parent
    module = batch / 'review_scan.py'
    if digest(module) == digest(source / module.name):
        return manifest
    continuation = digest(module) == PRE_SCAN_CONTINUATION_MODULE
    if (digest(module) not in {PRE_SCAN_FIT_RESULT_MODULE, PRE_SCAN_CONTINUATION_MODULE} or
            digest(batch / 'review_condor.py') not in
            {PRE_SCAN_FIT_RESULT_BACKEND, PRE_SCAN_CONTINUATION_BACKEND, digest(source / 'review_condor.py')}):
        raise RuntimeError('Unknown scan saveFitResult helper snapshot')
    archive = batch / (('scan_continuation_upgrade_' if continuation else 'scan_fit_result_upgrade_') + uuid4().hex[:8])
    archive.mkdir()
    names = ('review_scan.py', 'review_condor.py', 'manifest.json')
    for name in names:
        shutil.copy2(batch / name, archive / name)
    updated = copy.deepcopy(manifest)
    try:
        for name in names[:-1]:
            shutil.copy2(source / name, batch / name)
            updated['inputs'][str(batch / name)] = digest(batch / name)
        updated['scan_continuation_patch' if continuation else 'scan_fit_result_patch'] = {'archive': str(archive),
            'preserves_job_configs_and_successful_collectors': True}
        dump(batch / 'manifest.json', updated)
    except Exception:
        for name in names:
            shutil.copy2(archive / name, batch / name)
        raise
    action = 'SCAN CONTINUATION' if continuation else 'SCAN FIT RESULT'
    print(f'[{action}] Updated checked scan retry helper; successful jobs retained; backup={archive}')
    return updated


def main(review, args):
    if args.condor_worker:
        if args.condor_index is None:
            raise RuntimeError("Missing --condor-index")
        batch, manifest = load_batch(Path(args.condor_worker).parent)
        if not 0 <= args.condor_index < len(manifest["jobs"]):
            raise RuntimeError("Invalid --condor-index")
        execute(review, manifest, args.condor_index)
        return 0
    if getattr(args, "upgrade_backend", False) and args.stage != "resume":
        raise RuntimeError("--upgrade-backend requires --stage resume")
    if getattr(args, "parallel_bias_repair", False) and args.stage != "resume":
        raise RuntimeError("--parallel-bias-repair requires --stage resume")
    if args.stage == "run":
        prepare(review, args)
        return 0
    if not args.condor_dir:
        raise RuntimeError("--condor-dir is required for collect/resume")
    batch, manifest = load_batch(args.condor_dir)
    if args.stage == "collect":
        collect(batch, manifest)
    else:
        active_batch(batch)
        if not args.condor_dry_run:
            manifest = refresh_compatible_review_patch(review, batch, manifest)
            manifest = refresh_bias_refinement(review, batch, manifest)
        if getattr(args, "upgrade_backend", False) and getattr(args, "parallel_bias_repair", False):
            raise RuntimeError("Run the filename migration and parallel migration separately")
        if getattr(args, "parallel_bias_repair", False):
            if args.condor_dry_run:
                raise RuntimeError("--parallel-bias-repair cannot be combined with --condor-dry-run")
            manifest = parallelize_bias(review, batch, manifest)
        if getattr(args, "upgrade_backend", False):
            if args.condor_dry_run:
                raise RuntimeError("--upgrade-backend cannot be combined with --condor-dry-run")
            manifest = upgrade_backend(review, batch, manifest)
        if not args.condor_dry_run:
            manifest = refresh_scan_validation(review, batch, manifest)
            manifest = refresh_scan_fit_result(review, batch, manifest)
        indices, cache = [], {}
        for job in manifest["jobs"]:
            try:
                complete(manifest, job["index"], cache)
            except (RuntimeError, OSError, ValueError):
                indices.append(job["index"])
        if not indices:
            print("[RESUME] All jobs complete; use --stage collect")
            return 0
        refresh_child_dags(batch, manifest, indices)
        dag = "resume_" + uuid4().hex[:8] + ".dag"
        write_submission(batch, manifest, indices, dag)
        print(f"[RESUME] {len(indices)} failed/pending/stale nodes")
        if not args.condor_dry_run:
            submit(batch, manifest, dag)
    return 0
