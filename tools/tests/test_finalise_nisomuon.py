import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("finalise", Path(__file__).parents[1]/"finalise_nisomuon.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FinaliseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.base = self.repo/"NIsoMuon"
        self.base.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.git("remote", "add", "origin", m.SSH)
        for name in m.KEEP + m.REMOVE:
            p = self.base/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# fixture " + name + "\n")
        inputs = self.base/"sigfit_inputs/resolution_coefficients.json"
        inputs.parent.mkdir()
        inputs.write_text('{"input": 1}\n')
        (self.repo/"unrelated.txt").write_text("old unrelated\n")
        # This raw file is already tracked in a hypothetical older repository.
        # Its local modification must not enter the cleanup commit either.
        (self.base/"tracked.root").write_bytes(b"old ROOT")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        # Unique untracked helper must be archived before deletion.
        (self.base/"diagnose_zero_dy.py").write_text("# unique uncommitted diagnostic\n")
        for name in m.ROOT_LOGS:
            (self.base/name).write_text("saved root diagnostic " + name)
        record = {"files_sha256": {}}
        for target in ("Run2", "Run3", "Run2Run3"):
            folder = self.base/f"limit_outputs/alpha/blind/{target}"
            folder.mkdir(parents=True)
            for mass in (12,15,20,25,30,35,40,45,50,55,60,65,70):
                for suffix in (".pdf", ".json", "_internal.json"):
                    p = folder/f"impacts_{target}_alpha_M{mass}{suffix}"
                    p.write_text("final result\n")
                    record['files_sha256'][str(p.relative_to(self.base))] = m.sha(p)
            for suffix in (".pdf", ".json", "_internal.json"):
                p = folder/f"limits_{target}_alpha{suffix}"
                p.write_text("final limit\n")
                record['files_sha256'][str(p.relative_to(self.base))] = m.sha(p)
        (self.base/"review_outputs").mkdir()
        (self.base/"review_outputs/organisation_manifest.json").write_text(json.dumps(record))
        self.record = record

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE).decode().strip()

    def test_audit_refuses_changed_scientific_result_before_deletion(self):
        p = self.base/next(iter(self.record['files_sha256']))
        p.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed/missing"):
            m.audit(self.base)
        self.assertTrue((self.base/"organise_nisomuon.py").is_file())

    def test_cleanup_keeps_inputs_and_review_code_archives_unique_tools_and_logs(self):
        record = m.audit(self.base)
        m.clean(self.base, self.repo, record)
        for name in m.KEEP:
            self.assertTrue((self.base/name).is_file())
        for name in m.REMOVE + m.ROOT_LOGS:
            self.assertFalse((self.base/name).exists())
        self.assertFalse((self.base/"template").exists())
        archive = self.base/"review_outputs/provenance/retired_tools/diagnose_zero_dy.py"
        self.assertIn("unique uncommitted", archive.read_text())
        for name in m.ROOT_LOGS:
            self.assertTrue((self.base/"review_outputs/diagnostics/root_logs"/name).is_file())
        self.assertEqual(json.loads((self.base/"sigfit_inputs/resolution_coefficients.json").read_text()), {"input": 1})
        # The updated inventory remains valid after maintenance.
        m.audit(self.base)

    def test_unrecognised_template_inputs_are_preserved(self):
        p = self.base/"template/user_custom_model.txt"
        p.write_text("keep custom model")
        m.clean(self.base, self.repo, m.audit(self.base))
        self.assertEqual(p.read_text(), "keep custom model")

    def test_publication_commits_clean_results_excludes_root_unblind_and_unrelated_staged_files(self):
        m.clean(self.base, self.repo, m.audit(self.base))
        (self.repo/"unrelated.txt").write_text("staged unrelated change\n")
        self.git("add", "unrelated.txt")
        (self.base/"tracked.root").write_bytes(b"new ROOT not for publication")
        folder = self.base/"limit_outputs/alpha/blind/Run2"
        (folder/"raw.root").write_bytes(b"private raw fit")
        (folder/"duplicate.png").write_bytes(b"duplicate plot")
        unblind = self.base/"limit_outputs/alpha/unblind/Run2/result.json"
        unblind.parent.mkdir(parents=True)
        unblind.write_text('{"unblind": 1}')
        original = subprocess.run
        pushes = []
        def run(command, **kwargs):
            if command[:3] == ['git', '-C', str(self.repo)] and 'push' in command:
                pushes.append(command)
                return subprocess.CompletedProcess(command, 0)
            return original(command, **kwargs)
        with patch.object(m.subprocess, 'run', side_effect=run):
            m.commit_results(self.repo)
        self.assertEqual(len(pushes), 1)
        self.assertEqual(pushes[0][-2:], [m.SSH, 'HEAD:main'])
        self.assertEqual(self.git('show', 'HEAD:unrelated.txt'), 'old unrelated')
        self.assertIn('unrelated.txt', self.git('diff', '--cached', '--name-only'))
        self.assertEqual(self.git('show', 'HEAD:NIsoMuon/tracked.root'), 'old ROOT')
        committed = self.git('ls-tree', '-r', '--name-only', 'HEAD')
        self.assertNotIn('raw.root', committed)
        self.assertNotIn('unblind', committed)
        self.assertNotIn('duplicate.png', committed)
        self.assertNotIn('NIsoMuon/organise_nisomuon.py', committed)
        self.assertIn('impacts_Run2_alpha_M12.pdf', committed)

    def test_publication_selection_is_limited_to_ni_somuon_blind_artifacts(self):
        self.assertTrue(m.include_new('NIsoMuon/limit_outputs/alpha/blind/Run2/impacts_M12.json'))
        self.assertFalse(m.include_new('NIsoMuon/limit_outputs/alpha/unblind/Run2/limits.json'))
        self.assertFalse(m.include_new('NIsoMuon/review_outputs/diagnostics/fit.root'))
        self.assertFalse(m.include_new('OtherAnalysis/card.txt'))
        self.assertFalse(m.include_new('NIsoMuon/random_unknown_input.json'))

    def test_dry_run_does_not_modify_production(self):
        before = {str(p.relative_to(self.base)): m.sha(p) for p in self.base.rglob('*') if p.is_file()}
        with patch.object(m.sys, 'argv', ['finalise_nisomuon.py', '--production-dir', str(self.base)]):
            m.main()
        after = {str(p.relative_to(self.base)): m.sha(p) for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
