"""Reader caches retain numerical propagation and reject changed inputs."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]))
import limit_workflow as base
import limit_workflow_sideband_fit as shared
import dy_closure_shape
import full_spectrum_toys as toys
import json
import contextlib
import io


class ReaderEfficiencyTests(unittest.TestCase):
    def test_histogram_cache_is_bounded_and_keeps_recent_inputs(self):
        reader=self.reader();reader.histogram_cache_limit=2
        handle=SimpleNamespace(Get=lambda path:object())
        with patch.object(reader,'_file',return_value=handle),patch.object(handle,'Get',wraps=handle.Get) as get:
            a=reader.histogram('mc.root','a')
            reader.histogram('mc.root','b')
            self.assertIs(a,reader.histogram('mc.root','a'))
            reader.histogram('mc.root','c')
            self.assertEqual(len(reader.histogram_cache),2)
            self.assertIs(a,reader.histogram('mc.root','a'))
            reader.histogram('mc.root','b')
            self.assertEqual(get.call_count,4)

    def reader(self, cls=base.RootReader):
        reader=object.__new__(cls)
        reader.files={};reader.integral_cache={};reader.validated_input_cache={};reader.histogram_cache={}
        return reader

    def test_qcd_inputs_share_setup_and_per_window_reads_but_not_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.root';path.write_text('input')
            reader=self.reader()
            with patch.object(reader,'_file',return_value=object()), \
                 patch.object(base,'read_qcd_stat_inputs',return_value=('metadata','native','basis')) as read, \
                 patch.object(base,'qcd_root_window_statistics',side_effect=[{'sigma':1.},{'sigma':2.}]) as integrate:
                inputs=reader.qcd_inputs(str(path),'hist','2018')
                a=reader.qcd_statistics(str(path),'hist',11.,20.,'2018',10.)
                b=reader.qcd_statistics(str(path),'hist',20.,30.,'2018',3.)
                self.assertEqual(read.call_count,1)
                self.assertEqual(integrate.call_count,2)
                self.assertEqual((a['sigma'],b['sigma']),(1.,2.))
                path.write_text('changed input')
                with self.assertRaisesRegex(base.WorkflowError,'Input changed after validation'):
                    reader.qcd_inputs(str(path),'hist','2018')

    def test_dy_validation_shared_in_one_reader_and_rechecked_in_next(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'dy.root';path.write_text('DY')
            readers=[self.reader(),self.reader()]
            with patch.object(base.RootReader,'_file',return_value=object()), \
                 patch.object(dy_closure_shape,'read_inputs',return_value=object()) as read:
                first=base.read_dy_closure_inputs(readers[0],str(path))
                self.assertIs(first,base.read_dy_closure_inputs(readers[0],str(path)))
                self.assertEqual(read.call_count,1)
                base.read_dy_closure_inputs(readers[1],str(path))
                self.assertEqual(read.call_count,2)

    def test_histogram_is_loaded_once_for_distinct_window_integrals(self):
        class Histogram:
            def GetXaxis(self): return self
            def GetNbinsX(self): return 100
            def GetXmin(self): return 0.
            def GetXmax(self): return 100.
            def FindFixBin(self,x): return int(x)+1
            def GetBinLowEdge(self,i): return float(i-1)
            def GetBinUpEdge(self,i): return float(i)
            def IntegralAndError(self,first,last,error):
                error.value=(last-first+1)**.5
                return float(last-first+1)
        for cls in (base.RootReader,shared.BlindedReader):
            reader=self.reader(cls)
            handle=SimpleNamespace(Get=lambda path:Histogram())
            with patch.object(reader,'_file',return_value=handle), patch.object(handle,'Get',wraps=handle.Get) as get:
                a=reader.integral('mc.root','hist',11.,20.)
                b=reader.integral('mc.root','hist',20.,30.)
                self.assertEqual((a.value,b.value),(9.,10.))
                self.assertEqual((a.error,b.error),(3.,10.**.5))
                self.assertEqual(get.call_count,1)
                self.assertEqual(a,reader.integral('mc.root','hist',11.,20.))
                self.assertEqual(get.call_count,1)
            reader.close()
            self.assertFalse(reader.histogram_cache)
            self.assertFalse(reader.integral_cache)
        blind=self.reader(shared.BlindedReader)
        blind.integral_cache[('data.root','hist',11.,20.)]='must never be returned'
        with patch.object(blind,'_file',side_effect=AssertionError('OS data accessed')):
            with self.assertRaisesRegex(base.WorkflowError,'forbidden'):
                blind.integral('data.root','hist',11.,20.)

    def test_toy_submission_hashes_workspaces_once_not_per_chunk(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'original';work.mkdir()
            (work/'workspace.root').write_bytes(b'workspace'*1000)
            summary=work/'fit_summary.json';summary.write_text('{}')
            output=root/'batch';original=Path.open;reads=[]
            def opened(path,*args,**kwargs):
                if path.suffix=='.root' and args and args[0]=='rb':
                    reads.append(str(path))
                return original(path,*args,**kwargs)
            with patch.object(toys.audit,'find_summaries',return_value=[summary]), \
                 patch.object(toys,'plan_case',return_value=(dict(fit_card_sha256='card'),dict(mass=30.),1.,-5.)), \
                 patch.object(toys.base,'condor_environment',return_value=(root/'cmssw','test')), \
                 patch.object(Path,'open',new=opened),contextlib.redirect_stdout(io.StringIO()):
                toys.main(['--results',str(root),'--batch',str(output),'--backend','condor',
                    '--dry-run','--toys','30','--toys-per-job','1','--multipliers','0,1,2'])
            payload=json.loads((output/'manifest.json').read_text())
            self.assertEqual(len(payload['tasks']),90)
            self.assertEqual(len(reads),3) # source + fit snapshot + truth snapshot
            self.assertEqual(len(set(reads)),3)
            for task in payload['tasks']:
                self.assertEqual(task['inputs'],payload['tasks'][0]['inputs'])

    def test_workspace_hash_is_streamed_cached_and_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'workspace.root';content=b'abc'*1000000;path.write_bytes(content)
            original=Path.open;reads=[]
            def opened(p,*args,**kwargs):
                if p==path and args and args[0]=='rb':reads.append(p)
                return original(p,*args,**kwargs)
            with patch.object(Path,'read_bytes',side_effect=AssertionError('whole file read')), \
                 patch.object(Path,'open',new=opened):
                with base.validation_cache():
                    for _ in range(10):
                        self.assertEqual(base.validation_digest(path),hashlib.sha256(content).hexdigest())
                self.assertEqual(len(reads),1)
                path.write_bytes(b'new content')
                with base.validation_cache():
                    self.assertEqual(base.validation_digest(path),hashlib.sha256(b'new content').hexdigest())
                self.assertEqual(len(reads),2)


if __name__=='__main__':unittest.main()
