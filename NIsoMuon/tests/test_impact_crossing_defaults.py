import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import limit_workflow as base
import full_spectrum_inference as inference
import full_spectrum_impact_repair as repair


class CrossingTest(unittest.TestCase):
    def test_defaults_and_override(self):
        args=base.build_parser().parse_args([])
        self.assertEqual(base.impact_scan_options(args),
            ['--stepSize','0.1','--setCrossingTolerance','0.0001'])
        args.impact_step_size=.02
        args.impact_crossing_tolerance=1e-8
        self.assertEqual(base.impact_scan_options(args)[1::2],['0.02','1e-08'])
        for value in (0,-1,float('nan'),float('inf')):
            args.impact_step_size=value
            with self.assertRaises(base.WorkflowError): base.impact_scan_options(args)

    def test_isolated_worker_preserves_original_and_uses_same_injection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'original'; source.mkdir()
            cmssw=root/'cmssw'; (cmssw/'src').mkdir(parents=True)
            card=source/'card.txt';card.write_text('original card\n')
            old=source/'status.json';old.write_text('{"state":"running"}')
            job=dict(index=0,kind='impact_final',mass=30.,target='Run2Run3',
                injected_r=1.,card=str(card),card_sha256=inference.digest(card),
                workdir=str(source))
            payload=dict(jobs=[job],config=dict(mode='blind',parameter='alpha'),batch_id='original')
            payload['manifest_sha256']=inference.manifest_digest(payload)
            (source/'manifest.json').write_text(json.dumps(payload))
            batch=root/'new'
            with patch.dict(os.environ,CMSSW_BASE=str(cmssw),SCRAM_ARCH='test'):
                repair.main(['--source-batch',str(source),'--batch',str(batch),
                             '--indices','0','--dry-run'])
            def finish(payload,args,newjob,card,label,mass,directory):
                self.assertNotEqual(directory,source)
                self.assertEqual(args.impact_expect_signal,1.)
                self.assertEqual(args.impact_crossing_tolerance,1e-4)
                self.assertEqual(newjob,job)
                return dict(files=[])
            with patch.object(base,'run_condor_impact_final',side_effect=finish):
                repair.run_worker(batch,0)
            self.assertEqual(json.loads(old.read_text())['state'],'running')
            self.assertEqual(json.loads((batch/'job_00000/status.json').read_text())['state'],'success')
            with self.assertRaises(FileExistsError): repair.run_worker(batch,0)
            (batch/'source_manifest.json').write_text('{}')
            with self.assertRaises(ValueError): repair.run_worker(batch,0)


if __name__=='__main__':unittest.main()
