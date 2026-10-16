import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refresh_m12_impacts as refresh


def make_source(root, target):
    directory = root / target; directory.mkdir(parents=True)
    payload = {'POIs': [{'name':'r', 'fit':[-1.,0.,1.]}],
               'params':[{'name':'theta', 'fit':[-1.,0.,1.], 'prefit':[-1.,0.,1.],
                          'r':[.1,0.,-.1], 'impact_r':.1, 'type':'Gaussian'}]}
    physical = copy.deepcopy(payload); refresh.workflow.rescale_impact_payload(physical, .0003)
    stem = f'impacts_{target}_alpha_M12'
    refresh.dump(directory/(stem+'_internal.json'), payload)
    refresh.dump(directory/(stem+'.json'), physical)
    (directory/(stem+'.pdf')).write_bytes(b'%PDF-test')
    (directory/f'workspace_{target}_alpha_M12.root').touch()
    (directory/f'higgsCombine_initialFit_.Impacts_{target}_alpha_M12.MultiDimFit.mH12.root').touch()
    refresh.dump(directory/'impact_repair_M12/validation.json',
                 {'passed': True, 'target':target, 'mass':12., 'expected_signal':0.,
                  'expected':1, 'present':1, 'initial_poi':{'rmin':-50.,'rmax':100.}})
    return directory


def card(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# alpha_internal_unit = 0.0003\n')
    return path


def figures():
    return 'UNCHANGED INTRO\n' + '\n'.join(
        '\\begin{figure}[p]\n\\includegraphics{figures/limit/impacts_'+target+'_alpha_M12.pdf}\n'
        '\\caption{Existing caption. '+
        (r'This stored $12\GeV$ diagnostic uses $r\geq0$, giving one-sided impacts at the background-only boundary.' if target=='Run2Run3' else '')+
        '}\n\\label{'+target+'}\n\\end{figure}' for target in refresh.TARGETS) + '\nUNCHANGED END\n'


class RefreshTests(unittest.TestCase):
    def test_validate_negative_r_and_exact_alpha_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=make_source(root,'Run2'); c=card(root/'card.txt')
            with patch.object(refresh.workflow,'impact_workspace_parameters',return_value={'theta':(-5,5)}):
                report=refresh.validate(source,c,'Run2')
            self.assertTrue(report['passed']);self.assertEqual(report['initial_poi'],[-1,0,1])

    def test_reject_physical_cached_result_and_clipped_alpha(self):
        for clipped_internal in (True,False):
            with self.subTest(clipped_internal=clipped_internal),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); source=make_source(root,'Run2'); c=card(root/'card.txt')
                p=source/('impacts_Run2_alpha_M12_internal.json' if clipped_internal else 'impacts_Run2_alpha_M12.json')
                data=refresh.load(p);data['POIs'][0]['fit'][0]=0.;refresh.dump(p,data)
                with patch.object(refresh.workflow,'impact_workspace_parameters',return_value={'theta':(-5,5)}):
                    with self.assertRaises(RuntimeError):refresh.validate(source,c,'Run2')

    def test_sb_results_and_missing_nuisances_are_rejected(self):
        for sb in (True,False):
            with self.subTest(sb=sb),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); source=make_source(root,'Run2'); c=card(root/'card.txt')
                if sb:
                    p=source/'impact_repair_M12/validation.json';data=refresh.load(p);data['expected_signal']=1.;refresh.dump(p,data)
                with patch.object(refresh.workflow,'impact_workspace_parameters',return_value={'theta':(-5,5),'missing':(-5,5)}):
                    with self.assertRaises(RuntimeError):refresh.validate(source,c,'Run2')

    def test_only_three_m12_captions_change_and_operation_is_idempotent(self):
        reports={t:{'rmin':-50.,'rmax':100.} for t in refresh.TARGETS}
        text=figures(); result=refresh.update_an_captions(text,reports)
        self.assertEqual(result.count('negative internal $r$'),3)
        self.assertNotIn('This stored',result)
        self.assertTrue(result.startswith('UNCHANGED INTRO\n'))
        self.assertTrue(result.endswith('\nUNCHANGED END\n'))
        self.assertEqual(result,refresh.update_an_captions(result,reports))
        with self.assertRaises(RuntimeError):refresh.update_an_captions('No figures',reports)

    def test_three_target_installation_preserves_other_masses_and_excludes_roots_from_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); production=base/'producer/NIsoMuon';production.mkdir(parents=True)
            review=base/'review';review.mkdir();an=base/'an';an.mkdir();work=base/'work';work.mkdir()
            sources={t:make_source(base/'sources',t) for t in refresh.TARGETS}
            entries=[]
            for target in refresh.TARGETS:
                card(refresh.card_for(production,target))
                card(review/f'review_validation/b_only_impacts/{target}/M12/datacard_M-12_{target}.txt')
                entries.append({'target':target,'mass':12,'internal_json_sha256':'old'})
                for output in ('limit_outputs','limit_outputs_impacts_updated'):
                    p=production/f'{output}/alpha/blind/{target}/impacts_{target}_alpha_M12.pdf'
                    p.parent.mkdir(parents=True);p.write_bytes(b'%PDF-old')
            refresh.dump(review/'preservation/snapshot_manifest.json',{'b_only_impacts':entries+[{'target':'Run2','mass':20,'sentinel':'keep'}]})
            tex=an/'NPS-26-009.tex';tex.write_text(figures())
            sentinel=production/'limit_outputs/alpha/blind/Run2/M20_keep.txt';sentinel.write_text('keep')
            argv=['refresh_m12_impacts.py','--production-dir',str(production),
                  '--full-source',str(sources['Run2Run3']),'--run2-source',str(sources['Run2']),
                  '--run3-source',str(sources['Run3'])]
            def fake_git(directory,*args):
                self.assertEqual(args,('rev-parse','--show-toplevel'))
                return str(base/'producer')
            commits=[]
            with patch.object(sys,'argv',argv),patch.object(refresh,'git',side_effect=fake_git), \
                 patch.object(refresh,'check_repo'),patch.object(refresh.tempfile,'mkdtemp',side_effect=[str(work),str(base/'unused')]), \
                 patch.object(refresh,'clone_repo',side_effect=[review,an]), \
                 patch.object(refresh.workflow,'impact_workspace_parameters',return_value={'theta':(-5,5)}), \
                 patch.object(refresh,'commit_selected',side_effect=lambda d,p,m,b:commits.append((d,p,b))):
                refresh.main()
            self.assertEqual(sentinel.read_text(),'keep')
            for target in refresh.TARGETS:
                stem=f'impacts_{target}_alpha_M12'
                self.assertEqual((an/f'figures/limit/{stem}.pdf').read_bytes(),b'%PDF-test')
                for output in ('limit_outputs','limit_outputs_impacts_updated'):
                    d=production/f'{output}/alpha/blind/{target}'
                    self.assertEqual((d/(stem+'.pdf')).read_bytes(),b'%PDF-test')
                    self.assertTrue((d/f'workspace_{target}_alpha_M12.root').exists())
                    self.assertEqual((work/'backups'/target/output/(stem+'.pdf')).read_bytes(),b'%PDF-old')
            self.assertEqual(refresh.load(review/'preservation/snapshot_manifest.json')['b_only_impacts'][-1],{'target':'Run2','mass':20,'sentinel':'keep'})
            self.assertFalse(any(p.endswith('.root') for _,paths,_ in commits for p in paths))
            self.assertEqual(len(commits),3);self.assertTrue(all(not push for _,_,push in commits))

    def test_commit_does_not_include_unrelated_staged_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def command(*args):subprocess.run(['git','-C',str(root),*args],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            command('init','-b','main');command('config','user.name','Test');command('config','user.email','test@example.invalid')
            command('remote','add','origin','git@github.com:joonblee/higgs_combine.git')
            (root/'selected.txt').write_text('old');(root/'unrelated.txt').write_text('old')
            command('add','.');command('commit','-m','initial')
            (root/'selected.txt').write_text('new');(root/'unrelated.txt').write_text('new');command('add','unrelated.txt')
            pushed=[]
            def run(command,cwd=None):
                if 'push' in command:pushed.append(command)
                else:subprocess.run([str(x) for x in command],cwd=cwd,check=True,stdout=subprocess.DEVNULL)
            with patch.object(refresh,'run',side_effect=run):refresh.commit_selected(root,['selected.txt'],'update',True)
            self.assertEqual(refresh.git(root,'show','HEAD:unrelated.txt'),'old')
            self.assertIn('unrelated.txt',refresh.git(root,'diff','--cached','--name-only'))
            self.assertEqual(len(pushed),1)


if __name__=='__main__':unittest.main()
