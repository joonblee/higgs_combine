"""Signal SR counts match the original counting driver, including variations."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest
from collections import Counter
from unittest import mock
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parents[1]))
import limit_workflow as base
import signal_spectrum_interpolation as signal
import full_spectrum_fit as fs
import full_spectrum_condor as batch
import limit_workflow_sideband_fit as shared


class CountingSignalTests(unittest.TestCase):
    def test_dense_preflight_checks_anchors_once_and_rechecks_new_invocations(self):
        args=fs.configuration(fs.make_parser().parse_args(['--target','2018']))
        args.resolution_coefficients={'2018':(0.,.01)}
        files={20.:{'2018':'left.root'},40.:{'2018':'right.root'}}
        masses=[28.,30.,32.]
        args.signal_counting_windows={base.mass_label(m):{'2018':[
            dict(region='SR',low=m-1.,high=m+1.)]} for m in masses}
        nominal=base.hist_path(args.region)
        requests=[('nominal',nominal),('experimental','variation')]
        class Histogram:
            def GetNbinsX(self): return 2
            def GetXaxis(self): return self
            def GetBinLowEdge(self,index): return (11.,40.,80.)[index-1]
        class Reader:
            def __init__(self): self.reads=Counter();self.missing=False
            def _file(self,filename):
                def get(path):
                    self.reads[(filename,path)]+=1
                    return None if self.missing and path=='variation' else Histogram()
                return SimpleNamespace(Get=get)
            def integral(self,filename,path,low,high):
                value=100. if filename=='left.root' else 400.
                return base.YieldResult(value,value*.1,1,2)
        reader=Reader()
        with mock.patch.object(base,'file_for_process',side_effect=lambda args,era,process,name,source:name), \
             mock.patch.object(base,'whole_bin_window',side_effect=lambda hist,low,high:(low,high)), \
             mock.patch.object(base,'signal_template_requests',return_value=requests):
            expected=[signal.SpectrumSignalReader(reader,args,files,m,'2018').preflight() for m in masses]
            reader.reads.clear()
            records=signal.preflight(reader,args,files,masses,['2018'])
            self.assertEqual([r['raw_sr_yield'] for r in records],[v.value for v in expected])
            self.assertEqual([r['raw_sr_mc_error'] for r in records],[v.error for v in expected])
            for name in ('left.root','right.root'):
                self.assertEqual(reader.reads[(name,'variation')],2) # axis check + count read
                self.assertEqual(reader.reads[(name,nominal)],2) # shared geometry/axis + count
            reader.missing=True
            with self.assertRaisesRegex(base.WorkflowError,'Missing counting anchor histogram'):
                signal.preflight(reader,args,files,[30.],['2018'])

    def test_same_positive_yield_and_sumw2_interpolation(self):
        left,right=base.YieldResult(100.,10.,1,5),base.YieldResult(400.,40.,1,7)
        old=base.interpolate_signal_yield(30.,20.,left,40.,right)
        new=signal.interpolate_region(30.,20.,left,40.,right,is_sr=True)
        self.assertEqual(new,old)
        self.assertAlmostEqual(new.value,200.)
        self.assertAlmostEqual(new.error,200.*math.sqrt(.25*.1**2+.25*.1**2))

    def test_empty_or_signed_sidebands_do_not_invent_positive_sr_counts(self):
        zero=base.YieldResult(0.,0.,1,0);positive=base.YieldResult(10.,2.,1,2)
        result=signal.interpolate_region(30.,20.,zero,40.,positive)
        self.assertEqual((result.value,result.error),(5.,1.))
        with self.assertRaises(base.WorkflowError):
            signal.interpolate_region(30.,20.,zero,40.,positive,is_sr=True)
        negative=base.YieldResult(-2.,3.,1,2)
        result=signal.interpolate_region(30.,20.,negative,40.,positive)
        self.assertEqual(result.value,4.)
        self.assertAlmostEqual(result.error,math.hypot(1.5,1.))

    def test_mapped_edges_partition_each_anchor_without_overlap(self):
        edges=[i*.02 for i in range(5001)]
        target=[11.,15.,27.,28.5,31.5,35.,50.,80.]
        for anchor_window in ((19.,21.),(38.,42.)):
            mapped=[signal.map_edge(x,(28.5,31.5),anchor_window,edges) for x in target]
            self.assertEqual(mapped,sorted(mapped))
            self.assertEqual((mapped[0],mapped[-1]),(11.,80.))
            self.assertEqual(mapped[3:5],list(anchor_window))
            self.assertTrue(all(any(abs(x-y)<1e-8 for y in edges) for x in mapped))
            self.assertAlmostEqual(sum(b-a for a,b in zip(mapped,mapped[1:])),69.)

    def test_clipped_sr_edges_take_priority_over_missing_sideband(self):
        edges=[float(i) for i in range(101)]
        self.assertEqual(signal.map_edge(11.,(11.,14.),(12.,18.),edges),12.)
        self.assertEqual(signal.map_edge(80.,(65.,80.),(60.,75.),edges),75.)

    def test_exact_anchor_and_no_extrapolation(self):
        files={20.:{'2018':'left.root'},40.:{'2018':'right.root'}}
        self.assertEqual(signal.requested_files(files,20.,['2018'],True),files[20.])
        self.assertTrue(signal.requested_files(files,30.,['2018'],True)['2018'].startswith(signal.PREFIX))
        with self.assertRaisesRegex(base.WorkflowError,'use --interpolate-signal'):
            signal.requested_files(files,30.,['2018'],False)
        with self.assertRaisesRegex(base.WorkflowError,'extrapolation is disabled'):
            signal.requested_files(files,50.,['2018'],True)

    def test_condor_geometry_and_old_model_invalidation(self):
        opt=fs.make_parser().parse_args(['--interpolate-signal','--fit-layout','window'])
        geometry={'30':{'2018':[dict(region='SR',low=28.,high=32.)]}}
        payload=dict(options=vars(opt),batch='/tmp/test',input_base_dir='/tmp/inputs',
                     resolution_coefficients={'2018':[0.,.01]},windows=geometry,
                     signal_interpolation_anchors={'20.0':{'2018':'left.root'}})
        _,args=batch.worker_options(payload)
        self.assertEqual(args.signal_counting_windows,geometry)
        self.assertEqual(fs.signature(opt)['signal_interpolation_method'],'counting_v2')
        self.assertIn('signalSR25_v1',str(fs.model_directory(opt,30.)))
        self.assertIn('signalCounting_v2',str(fs.model_directory(opt,30.)))
        del opt.signal_counting_version
        self.assertNotIn('signal_interpolation_method',fs.signature(opt))
        self.assertNotIn('signal_normalisation_reference',fs.signature(opt))
        self.assertNotIn('signalSR25_v1',str(fs.model_directory(opt,30.)))
        self.assertNotIn('signalCounting_v2',str(fs.model_directory(opt,30.)))

    def test_window_reference_matches_counting_25_sr_and_preserves_leakage(self):
        opt=fs.make_parser().parse_args(['--fit-layout','window','--target','2018'])
        args=fs.configuration(opt);args.resolution_coefficients={'2018':(0.,.01)}
        opt.norm_scope='era'
        raw=[]
        for region,low,high,sig in (('L0000',11.,28.5,1.),('SR',28.5,31.5,10.),('R0000',31.5,80.,2.)):
            raw.append(dict(era='2018',region=region,low=low,high=high,channel='bin_2018_'+region,
                rates=dict(sig=sig,QCD=100.,DY=20.,tt=30.,ST=1.,Others=1.),
                nuisances={},qcd_shape=2.,qcd_modes=[1.],dy_stat=1.,dy_nf=.1,warnings=[]))
        part=dict(era='2018',rows=raw,transfer_ratios=[[.1,.2]],input_audits=[])
        model=shared.assemble_model(args,opt,30.,[part],full_spectrum=True)
        self.assertAlmostEqual(model['alpha_internal_unit'],2.5)
        self.assertEqual(model['raw_reference_signal_yield'],10.)
        self.assertAlmostEqual(sum(r['rates']['sig'] for r in model['rows'] if r['region']=='SR'),25.)
        self.assertAlmostEqual(sum(r['rates']['sig'] for r in model['rows']),32.5)
        self.assertAlmostEqual(model['eras']['2018']['signal_sb_over_sr'],.3)
        card=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        self.assertIn('# alpha_internal_signal_yield = 25',card)
        for r in raw:
            self.assertAlmostEqual(shared.nominal_predictions(card)[r['channel']],r['asimov'])

    @unittest.skipUnless(importlib.util.find_spec('ROOT'),'PyROOT unavailable')
    def test_root_sr_counts_and_systematics_equal_legacy_driver(self):
        script=r'''
import math,tempfile
from pathlib import Path
import ROOT
import limit_workflow as b
import limit_workflow_sideband_fit as shared
import full_spectrum_fit as fs
import signal_spectrum_interpolation as signal
ROOT.TH1.AddDirectory(False)
with tempfile.TemporaryDirectory() as directory:
 opt=fs.make_parser().parse_args(['--backend','local','--target','2018','--fit-layout','window','--interpolate-signal','--masses','30','--base-dir',directory])
 args=fs.configuration(opt);args.resolution_coefficients={'2018':(0.,.01)}
 files={20.:{'2018':'NIsoMuon_Zp_M-20.root'},40.:{'2018':'NIsoMuon_Zp_M-40.root'}}
 requests=b.signal_template_requests(args,'2018')
 for mass,n in ((20.,100.),(40.,400.)):
  hist=ROOT.TH1D('seed','',5000,0.,100.)
  for i in range(1,5001):
   x=hist.GetBinCenter(i)
   y=n*.02/(mass*.01*math.sqrt(2*math.pi))*math.exp(-.5*((x-mass)/(mass*.01))**2)+.0001
   hist.SetBinContent(i,y);hist.SetBinError(i,math.sqrt(y))
  for source in ('nominal','experimental','theory'):
   path=Path(b.file_for_process(args,'2018','sig',files[mass]['2018'],source));path.parent.mkdir(parents=True,exist_ok=True)
   handle=ROOT.TFile.Open(str(path),'RECREATE')
   for src,name in requests:
    if src!=source:continue
    dirname,objectname=name.rsplit('/',1)
    target=handle.GetDirectory(dirname) or handle.mkdir(dirname);target.cd()
    h=hist.Clone(objectname)
    if source!='nominal':h.Scale(1.1 if source=='experimental' else .8)
    h.Write();handle.cd()
   handle.Close()
 legacy_reader=b.RootReader();legacy=b.SignalInterpolator(legacy_reader,args,files,['2018'])
 blinded=shared.BlindedReader()
 windows=fs.window_bins([11.+i*.02 for i in range(3451)],*b.mass_window(args,'2018',30.),reference=fs.SIDEBAND_PRESETS['coarse'])
 args.signal_counting_windows={'30':{'2018':windows}}
 new=signal.SpectrumSignalReader(blinded,args,files,30.,'2018');new.preflight()
 sr=next(r for r in windows if r['region']=='SR')
 for source,path in requests:
  virtual=b.file_for_process(args,'2018','sig',signal.PREFIX+'30.root',source)
  result=new.integral(virtual,path,sr['low'],sr['high'])
  original=legacy.evaluate('2018',30.,source,path)
  assert abs(result.value-original.value)<1e-10,(source,path,result,original)
  assert abs(result.error-original.error)<1e-10
 # All anchor intervals are disjoint whole bins and cover the full search domain.
 for anchor in (20.,40.):
  intervals=[tuple(signal.map_edge(x,new.target_window,new.anchor_windows[anchor],new.edges[anchor]) for x in (r['low'],r['high'])) for r in windows]
  assert all(abs(a[1]-c[0])<1e-10 for a,c in zip(intervals,intervals[1:]))
  path=b.hist_path(args.region);file=b.file_for_process(args,'2018','sig',files[anchor]['2018'],'nominal')
  total=sum(blinded.integral(file,path,a,c).value for a,c in intervals if a<c)
  assert abs(total-blinded.integral(file,path,11.,80.).value)<1e-9
 # The new sidebands retain signal counts, instead of dropping them.
 virtual=b.file_for_process(args,'2018','sig',signal.PREFIX+'30.root','nominal')
 assert sum(new.integral(virtual,b.hist_path(args.region),r['low'],r['high']).value for r in windows if r['region']!='SR')>0.
 # Dense preflight must reuse anchor checks without changing SR yields/errors.
 masses=[28.,30.,32.]
 expected={}
 for mass in masses:
  args.signal_counting_windows[b.mass_label(mass)]={'2018':fs.window_bins(
   [11.+i*.02 for i in range(3451)],*b.mass_window(args,'2018',mass),reference=fs.SIDEBAND_PRESETS['coarse'])}
  expected[mass]=signal.SpectrumSignalReader(blinded,args,files,mass,'2018').preflight()
 original=signal.SpectrumSignalReader._hist
 reads={}
 def tracked(owner,source,anchor,path):
  key=(anchor,source,path);reads[key]=reads.get(key,0)+1
  return original(owner,source,anchor,path)
 signal.SpectrumSignalReader._hist=tracked
 try:
  records=signal.preflight(blinded,args,files,masses,['2018'])
 finally:
  signal.SpectrumSignalReader._hist=original
 for record in records:
  wanted=expected[record['mass']]
  assert record['raw_sr_yield']==wanted.value
  assert record['raw_sr_mc_error']==wanted.error
 for anchor in (20.,40.):
  for source,path in requests:
   assert reads[(anchor,source,path)]==(2 if source=='nominal' else 1),(anchor,source,path,reads)
 # A new preflight must check inputs again rather than keeping a global cache.
 signal.preflight(blinded,args,files,[30.],['2018'])
 legacy_reader.close();blinded.close()
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
