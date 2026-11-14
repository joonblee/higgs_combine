"""Correlated era aggregation and source-specific prefit responses."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
import full_spectrum_component_uncertainties as component


class ComponentTests(unittest.TestCase):
    def test_common_edges_coarsen_without_splitting(self):
        rows=[dict(era='a',low=11.,high=30.),dict(era='a',low=30.,high=80.),
              dict(era='b',low=11.,high=20.),dict(era='b',low=20.,high=80.)]
        own,edges,groups=component.common_groups(rows,['a','b'])
        self.assertEqual(edges,[11.,80.]);self.assertEqual(groups,[[0,1,2,3]])
        self.assertEqual(len(own),4)

    def test_card_prior_classification(self):
        priors,free=component.nuisance_priors('s lnN 1.1\nf param 0 2\nn rateParam x DY 1 [0,10]\nx rateParam y DY (@0) n')
        self.assertEqual(priors,dict(s=(0.,1.),f=(0.,2.)));self.assertEqual(free,{'n'})
        self.assertEqual(component.category('FS_QCDFitMode1_2018',{}),'Statistics')
        self.assertEqual(component.category('CMS_NPS26009_QCDStat_mode4_2018',{}),'Statistics')
        self.assertEqual(component.category('CMS_NPS26009_DYClosureLinearShape',{}),'Modelling')

    @unittest.skipUnless(importlib.util.find_spec("ROOT"),"PyROOT unavailable")
    def test_workspace_era_correlation_is_not_added_independently(self):
        script=r'''
import ROOT
import full_spectrum_component_uncertainties as c
w=ROOT.RooWorkspace('w')
w.factory('shared[0,-7,7]');w.factory('local[0,-7,7]')
rows=[]
for era in ('a','b'):
 row=dict(era=era,channel=era,low=11.,high=80.);rows.append(row)
 for process in c.PROCESSES:
  formula='100*(1+0.1*@0)' if era=='a' else '100*(1-0.1*@0)'
  if process=='DY':formula='100*(1+0.2*@0)'
  w.factory('expr::n_exp_bin'+era+'_proc_'+process+'("'+formula+'",shared)')
model=dict(rows=rows,normalisations={})
payload=c.evaluate_workspace(w,model,dict(shared=(0.,1.)),set(),['a','b'])
assert payload['processes']['QCD']['relative']['Total']['up']==[0.]
assert abs(payload['processes']['DY']['relative']['Total']['up'][0]-.2)<1e-12
assert abs(w.var('shared').getVal())<1e-12
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
