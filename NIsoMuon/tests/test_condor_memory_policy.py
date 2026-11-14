"""Memory recovery must never release unrelated holds or disturb active fits."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import condor_memory_policy as policy


class MemoryPolicyTests(unittest.TestCase):
    def test_native_feature_is_version_gated(self):
        for text,expected in (('$CondorVersion: 24.12.1 $',False),
                              ('$CondorVersion: 25.2.0 $',False),
                              ('$CondorVersion: 25.2.1 $',True),
                              ('$CondorVersion: 26.0.0 $',True)):
            policy.native_retry_supported.cache_clear()
            with patch.object(policy.subprocess,'check_output',return_value=text):
                self.assertEqual(policy.native_retry_supported(),expected)
        policy.native_retry_supported.cache_clear()

    def test_only_oom_hold_below_ceiling_is_released(self):
        row=dict(ClusterId=100,ProcId=0,JobUniverse=5,JobStatus=5,
                 HoldReasonCode=34,RequestMemory=1024)
        commands=policy.recovery_commands(row)
        self.assertEqual(commands[0][-2:],['RequestMemory','2048'])
        self.assertIn('HoldReasonCode == 34',commands[1][-1])
        self.assertIn('RequestMemory == 2048',commands[1][-1])
        for changed in (dict(JobStatus=2),dict(JobStatus=1),dict(HoldReasonCode=1),
                        dict(HoldReasonCode=12),dict(RequestMemory=8192),
                        dict(RequestMemory=16384),dict(JobUniverse=7)):
            self.assertEqual(policy.recovery_commands(dict(row,**changed)),[])

    def test_repeated_oom_doubles_until_cap(self):
        for before,after in ((1024,2048),(2048,4096),(4096,8192)):
            row=dict(ClusterId=1,ProcId=0,JobUniverse=5,JobStatus=5,
                     HoldReasonCode=34,RequestMemory=before)
            self.assertEqual(policy.recovery_commands(row)[0][-1],str(after))

    def test_queue_update_keeps_workspace_and_active_or_retried_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            batch=Path(temp)
            manifest=json.dumps(dict(jobs=[dict(index=0,kind='workspace'),
                                          dict(index=1,kind='impact_nuisance')]))
            (batch/'manifest.json').write_text(manifest)
            (batch/'submit.jds').write_text('request_memory = 16000MB\nqueue 1\n')
            rows=[dict(ClusterId=100+i,ProcId=0,Iwd=str(batch),DAGNodeName='J'+str(node),
                       JobUniverse=5,JobStatus=state,RequestMemory=memory,NumJobStarts=starts,
                       HoldReasonCode=hold)
                  for i,(node,state,memory,starts,hold) in enumerate(
                      [(0,1,16000,0,0),(1,1,16000,0,0),(1,2,16000,1,0),
                       (1,1,2048,1,0),(1,5,1024,1,34),(1,5,1024,1,12)])]
            def output(command,**kwargs):
                return 'joonblee\n' if command[0]=='id' else json.dumps(rows)
            with patch.object(policy.subprocess,'check_output',side_effect=output), \
                 patch.object(policy.subprocess,'run') as run,contextlib.redirect_stdout(io.StringIO()):
                policy.main(['--batch',str(batch),'--set-idle','--retry-held'])
            commands=[c.args[0] for c in run.call_args_list]
            self.assertEqual(len(commands),4) # one release pair and two memory groups
            combined=' '.join(' '.join(c) for c in commands)
            for untouched in (102,103,105):
                self.assertNotIn('ClusterId == '+str(untouched),combined)
            self.assertIn('2048',(batch/'submit.jds').read_text())
            self.assertIn('1024',(batch/'submit.jds').read_text())
            self.assertEqual((batch/'manifest.json').read_text(),manifest)
            self.assertEqual((batch/'submit.jds.before_1gb_memory').read_text(),
                             'request_memory = 16000MB\nqueue 1\n')


if __name__=='__main__':unittest.main()
