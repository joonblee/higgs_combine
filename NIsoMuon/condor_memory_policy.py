#!/usr/bin/env python3
"""Bounded memory recovery; query the live queue only, never job history."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess
import time


@lru_cache(maxsize=1)
def native_retry_supported():
    try:
        version = subprocess.check_output(['condor_version'], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    match = re.search(r'CondorVersion:\s*(\d+)\.(\d+)\.(\d+)', version)
    return bool(match and tuple(map(int, match.groups())) >= (25, 2, 1))


def retry_lines(maximum='8192'):
    """25.2.1 introduced native memory retries; older submitters must not ignore them."""
    if native_retry_supported():
        return ['retry_request_memory_increase = RequestMemory',
                f'retry_request_memory_max = {maximum}MB']
    print('[MEMORY] Native memory retry unavailable. For OOM-held jobs use '
          'condor_memory_policy.py --retry-held with --batch or --dag-id; '
          'add --watch for automatic recovery.')
    return []


def recovery_commands(ad, ceiling=8192):
    """Retry only memory-limit holds, by doubling up to the ceiling; never generic fit failures."""
    request = ad.get('RequestMemory')
    if (ad.get('JobUniverse') != 5 or ad.get('JobStatus') != 5
            or ad.get('HoldReasonCode') != 34
            or not isinstance(request, (float, int)) or not 0 < request < ceiling):
        return []
    next_memory = min(ceiling,max(2048,int(request*2)))
    selection = (f'ClusterId == {int(ad["ClusterId"])} && ProcId == {int(ad["ProcId"])} '
                 f'&& JobStatus == 5 && HoldReasonCode == 34 && RequestMemory == {request}')
    return [
        ['condor_qedit', '-constraint', selection, 'RequestMemory', str(next_memory)],
        ['condor_release', '-constraint', selection.replace(
            f'RequestMemory == {request}', f'RequestMemory == {next_memory}')],
    ]


def initial_memory(ad, payload, default=1024):
    node = re.fullmatch(r'J(\d+)', str(ad.get('DAGNodeName', '')))
    if node and 'jobs' in payload:
        index = int(node.group(1))
        if not 0 <= index < len(payload['jobs']):
            raise ValueError('Unknown DAG node: '+node.group(0))
        if payload['jobs'][index]['kind'] == 'workspace':
            return max(default, 2048)
    return default


def patch_submit(batch, payload, default):
    """Change scheduling only; preserve the signed physics/source manifest."""
    path = batch/'submit.jds'
    if not path.is_file():
        raise ValueError('Missing submit.jds: '+str(batch))
    original = path.read_text()
    indices = [str(j['index']) for j in payload.get('jobs', []) if j['kind']=='workspace']
    condition = ' || '.join('$(jobindex) == '+i for i in indices)
    expression = (f'ifThenElse(({condition}), {max(default,2048)}, {default})'
                  if condition else str(default))
    updated, count = re.subn(r'^request_memory\s*=.*$', 'request_memory = '+expression,
                            original, flags=re.MULTILINE)
    if count != 1:
        raise ValueError('Expected one request_memory in '+str(path))
    # Existing native policy (if any) remains in force. Queue retries are also
    # available through --watch on old HTCondor installations.
    backup = batch/'submit.jds.before_1gb_memory'
    if not backup.exists():
        backup.write_text(original)
    temporary = path.with_suffix('.jds.memory.tmp')
    temporary.write_text(updated)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--batch')
    target.add_argument('--dag-id', type=int)
    parser.add_argument('--set-idle', action='store_true', help='Also lower idle/future requests; running jobs unchanged.')
    parser.add_argument('--retry-held', action='store_true', help='Release only memory-limit holds below the retry ceiling.')
    parser.add_argument('--memory', type=int, default=1024)
    parser.add_argument('--retry-memory', type=int, default=8192)
    parser.add_argument('--watch', action='store_true', help='Repeat queue-only checks every 30 seconds; stop with Ctrl+C.')
    parser.add_argument('--dry-run', action='store_true')
    opt = parser.parse_args(argv)
    if opt.memory < 1 or opt.retry_memory < opt.memory or not (opt.set_idle or opt.retry_held):
        parser.error('Use positive memory, retry-memory >= memory and --set-idle and/or --retry-held.')
    constraint = 'Owner == '+json.dumps(subprocess.check_output(['id','-un'], text=True).strip())+' && JobUniverse == 5'
    constraint += (' && Iwd == '+json.dumps(str(Path(opt.batch).resolve())) if opt.batch
                   else ' && DAGManJobId == '+str(opt.dag_id))
    manifests, patched = {}, set()
    while True:
        ads = json.loads(subprocess.check_output(['condor_q','-constraint',constraint,
            '-attributes','ClusterId,ProcId,Iwd,DAGNodeName,JobUniverse,JobStatus,HoldReasonCode,RequestMemory,NumJobStarts','-json'], text=True))
        if not ads:
            print('[MEMORY] No matching live jobs.');return
        edits = retries = 0
        idle_groups = defaultdict(list)
        for ad in ads:
            batch = Path(ad['Iwd']).resolve()
            if batch not in manifests:
                manifests[batch] = json.loads((batch/'manifest.json').read_text())
            payload = manifests[batch]
            if opt.set_idle and batch not in patched:
                if not opt.dry_run:patch_submit(batch,payload,opt.memory)
                print('[MEMORY] Future requests:',batch,'workspace >=2048 MB, others',opt.memory,'MB')
                patched.add(batch)
            commands = []
            if opt.set_idle and ad['JobStatus']==1 and ad.get('NumJobStarts',0)==0:
                memory = initial_memory(ad,payload,opt.memory)
                # Never undo an OOM escalation on a subsequent watch pass.
                if ad.get('RequestMemory') != memory:
                    idle_groups[memory].append((int(ad['ClusterId']),int(ad['ProcId'])))
                    edits += 1
            if opt.retry_held:
                recovery = recovery_commands(ad,opt.retry_memory)
                commands += recovery
                retries += bool(recovery)
            for command in commands:
                if opt.dry_run:print('[DRY RUN]',command)
                else:subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        for memory, identities in idle_groups.items():
            for start in range(0,len(identities),200):
                selected = ' || '.join(f'(ClusterId == {c} && ProcId == {p})'
                                      for c,p in identities[start:start+200])
                command = ['condor_qedit','-constraint',
                    f'JobStatus == 1 && (NumJobStarts =?= undefined || NumJobStarts == 0) && ({selected})',
                    'RequestMemory',str(memory)]
                if opt.dry_run:print('[DRY RUN]',command)
                else:subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        print(f'[MEMORY] idle updated={edits}, OOM retries={retries}; running jobs unchanged.',flush=True)
        # Lower idle requests only once; later passes only recover OOM holds.
        opt.set_idle = False
        if not opt.watch:return
        time.sleep(30)


if __name__ == '__main__':
    try:main()
    except KeyboardInterrupt:pass
