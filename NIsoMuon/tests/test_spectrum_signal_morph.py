"""Full-spectrum morph moves the peak and propagates anchor counting errors."""
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parents[1]))
import signal_spectrum_interpolation as morph
import full_spectrum_fit as fs


class Axis:
    def __init__(self, edges): self.edges=edges
    def GetBinLowEdge(self,i): return self.edges[i-1]
class Hist:
    def __init__(self,edges,values,variances=None):
        self.edges,self.values=edges,values
        self.variances=variances or values
    def GetXaxis(self): return Axis(self.edges)
    def GetNbinsX(self): return len(self.values)
    def GetBinContent(self,i): return self.values[i-1]
    def GetBinError(self,i): return math.sqrt(self.variances[i-1])


class MorphTests(unittest.TestCase):
    def test_fractional_integral_and_covariance(self):
        h=morph.HistogramIntegral(Hist([0.,1.,2.,3.],[10.,20.,30.]))
        for value,expected in zip(h.partial(.5,2.5),(40.,30.,40.)): self.assertAlmostEqual(value,expected)
        self.assertEqual(h.partial(-100.,100.),(60.,60.,60.))

    def test_alignment_moves_peak_and_preserves_full_yield(self):
        edges=[i*.1 for i in range(1001)]
        def signal(mass,n):
            values=[math.exp(-.5*((.05+i*.1-mass)/1.)**2) for i in range(1000)]
            return morph.HistogramIntegral(Hist(edges,[n*v/sum(values) for v in values]))
        left,right=signal(20.,100.),signal(40.,400.)
        result=morph.morph_integral(29.,31.,.5,left,right,30.,20.,40.,1.,1.,1.)
        self.assertGreater(result.value,130.)  # Both anchor peaks are transported to 30.
        full=morph.morph_integral(-100.,200.,.5,left,right,30.,20.,40.,1.,1.,1.)
        self.assertAlmostEqual(full.value,200.)
        self.assertAlmostEqual(full.error,200.*math.sqrt(.25/100.+.25/400.))
        parts=[morph.morph_integral(a,b,.5,left,right,30.,20.,40.,1.,1.,1.).value
               for a,b in zip((0.,29.,30.,31.),(29.,30.,31.,100.))]
        self.assertAlmostEqual(sum(parts),200.)

    def test_sumw2_derivative_matches_finite_difference(self):
        values=([3.,5.,2.],[7.,8.,4.]); variance=0.
        def result(values):
            left,right=[morph.HistogramIntegral(Hist([0.,1.,2.,3.],v)) for v in values]
            return morph.morph_integral(.3,2.2,.4,left,right,1.5,1.,2.,1.,.8,1.2)
        for anchor in (0,1):
            for index in range(3):
                plus=[list(v) for v in values];minus=[list(v) for v in values]
                plus[anchor][index]+=1e-5;minus[anchor][index]-=1e-5
                derivative=(result(plus).value-result(minus).value)/2e-5
                variance+=derivative**2*values[anchor][index]
        self.assertAlmostEqual(result(values).error**2,variance,places=7)

    def test_exact_anchor_and_no_mass_extrapolation(self):
        files={20.:{'2018':'anchor20.root'},40.:{'2018':'anchor40.root'}}
        self.assertEqual(morph.requested_files(files,20.,['2018'],True),files[20.])
        self.assertTrue(morph.requested_files(files,30.,['2018'],True)['2018'].startswith(morph.PREFIX))
        with self.assertRaisesRegex(Exception,'use --interpolate-signal'):
            morph.requested_files(files,30.,['2018'],False)
        with self.assertRaisesRegex(Exception,'extrapolation is disabled'):
            morph.requested_files(files,50.,['2018'],True)

    def test_default_signature_unchanged_and_condor_restores_anchors(self):
        default=fs.make_parser().parse_args([])
        self.assertNotIn('interpolate_signal',fs.signature(default))
        enabled=fs.make_parser().parse_args(['--interpolate-signal'])
        self.assertTrue(fs.signature(enabled)['interpolate_signal'])
        import full_spectrum_condor as batch
        self.assertIn('signal_spectrum_interpolation.py',batch.SOURCES)
        payload=dict(options=vars(enabled),batch='/tmp/morph',input_base_dir='/tmp/input',
                     resolution_coefficients={'2018':[.001,.01]},
                     signal_interpolation_anchors={'20.0':{'2018':'left.root'}},
                     signal_interpolation_audit=[dict(mass=30.)])
        _,args=batch.worker_options(payload)
        self.assertEqual(args.signal_interpolation_anchors,payload['signal_interpolation_anchors'])



class ReaderRootTests(unittest.TestCase):
    def test_root_collections_and_systematic_variations_use_same_peak_morph(self):
        import importlib.util
        if not importlib.util.find_spec('ROOT'): self.skipTest('PyROOT unavailable')
        import subprocess
        script=r'''
import math,tempfile
from pathlib import Path
import ROOT
import limit_workflow as b
import signal_spectrum_interpolation as m
ROOT.TH1.AddDirectory(False)
with tempfile.TemporaryDirectory() as directory:
 args=b.build_parser().parse_args(['--base-dir',directory,'--trigger',''])
 args.resolution_coefficients={'2018':(0.,.02)}
 signals={12.:{'2018':'NIsoMuon_Zp_M-12.root'},15.:{'2018':'NIsoMuon_Zp_M-15.root'}}
 for source,scale in (('nominal',1.),('experimental',2.),('theory',.5)):
  for mass,files in signals.items():
   path=Path(b.file_for_process(args,'2018','sig',files['2018'],source))
   path.parent.mkdir(parents=True,exist_ok=True)
   f=ROOT.TFile.Open(str(path),'RECREATE')
   h=ROOT.TH1D('hist','',5000,0.,100.)
   for i in range(1,5001):
    y=scale*100*.02/(mass*.02*math.sqrt(2*math.pi))*math.exp(-.5*((h.GetBinCenter(i)-mass)/(mass*.02))**2)
    h.SetBinContent(i,y);h.SetBinError(i,math.sqrt(y))
   h.Write();f.Close()
 reader=b.RootReader();wrapped=m.SpectrumSignalReader(reader,args,signals,13.5,'2018')
 values=[]
 for source in ('nominal','experimental','theory'):
  file=b.file_for_process(args,'2018','sig',m.PREFIX+'13p5.root',source)
  result=wrapped.integral(file,'hist',13.,14.)
  assert result.value>0 and result.error>0
  values.append(result.value)
 assert abs(values[1]/values[0]-2.)<1e-10
 assert abs(values[2]/values[0]-.5)<1e-10
 reader.close()
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
