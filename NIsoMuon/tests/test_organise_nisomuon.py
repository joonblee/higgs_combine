import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("organise", Path(__file__).parents[1]/"organise_nisomuon.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class OrganisationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/"NIsoMuon"
        self.review = self.root/"combine_review"
        self.base.mkdir()
        (self.base/"limit_workflow.py").write_text("# fixture")
        (self.base/"sigfit_inputs").mkdir()
        (self.base/"sigfit_inputs/input.root").write_bytes(b"untouched input")
        self.entries = []
        for target in mod.TARGETS:
            for mass in mod.MASSES:
                folder = "limit_outputs_impacts_retry" if mass in (30, 70) else "limit_outputs_impacts_updated"
                source = self.base/f"{folder}/alpha/blind/{target}"
                source.mkdir(parents=True, exist_ok=True)
                stem = f"impacts_{target}_alpha_M{mass}"
                payload = {"POIs": [{"name": "r", "fit": [-1, 0, 1]}], "params": []}
                mod.dump(source/(stem+"_internal.json"), payload)
                mod.dump(source/(stem+".json"), payload)
                (source/(stem+".pdf")).write_bytes(b"%PDF-1.4\n" + stem.encode())
                (source/f"workspace_{target}_alpha_M{mass}.root").write_bytes((stem+"workspace").encode())
                (source/f"higgsCombine_initialFit_.Impacts_{target}_alpha_M{mass}.MultiDimFit.root").write_bytes((stem+"initial").encode())
                mod.dump(source/f"impact_repair_M{mass}/validation.json", {"passed": True})
                card = self.base/f"datacards_NIsoMuon/alpha/blind/combined/{target}/datacard_M-{mass}_{target}.txt"
                card.parent.mkdir(parents=True, exist_ok=True)
                card.write_text(f"counting {target} {mass}")
                reference = self.review/f"review_validation/b_only_impacts/{target}/M{mass}"
                reference.mkdir(parents=True)
                for suffix in (".pdf", ".json", "_internal.json"):
                    (reference/(stem+suffix)).write_bytes((source/(stem+suffix)).read_bytes())
                self.entries.append({"target": target, "mass": mass, "source_directory": str(source),
                                     "card_sha256": mod.sha(card),
                                     "internal_json_sha256": mod.sha(source/(stem+"_internal.json")),
                                     "validation_directories": [f"impact_repair_M{mass}"]})
        mod.dump(self.review/"preservation/snapshot_manifest.json", {"b_only_impacts": self.entries})
        # Existing results include independent unblind files which must survive.
        (self.base/"limit_outputs/alpha/unblind").mkdir(parents=True)
        (self.base/"limit_outputs/alpha/unblind/keep.txt").write_text("untouched")
        mod.dump(self.base/"limit_outputs/limits.json", {"limit": 7})
        for name in mod.DEPENDENCIES:
            (self.base/name).mkdir()
            (self.base/name/"evidence.root").write_bytes(name.encode())
        mod.dump(self.base/mod.DEPENDENCIES[0]/"validation.json", {"prior": "kept"})
        for mass, name in ((20, mod.M20), (70, mod.M70)):
            raw = self.base/mod.REVIEW_BASE/f"Run2Run3/M{mass}"
            raw.mkdir(parents=True)
            workspace = raw/f"workspace_Run2Run3_M{mass}.root"
            workspace.write_bytes(b"original toys workspace")
            summary = raw/f"bias/bias_summary_Run2Run3_M{mass}.json"
            mod.dump(summary, [{"old": i} for i in range(3)])
            result = self.base/name
            mod.dump(result/"validation.json", {"passed": True, "errors": [],
                                                "n_verified_repairs": {20: 32, 70: 4}[mass]})
            toys = {str(i): {"interval": [-1, 0, 1], "source": str(workspace)} for i in range(1, 501)}
            mod.dump(result/"merged_toys.json", {f"r{i}": toys for i in range(3)})
            mod.dump(result/f"bias_summary_Run2Run3_M{mass}.json", [{"final": i} for i in range(3)])
            mod.dump(result/"input_manifest.json", {"source_summary": str(summary),
                "input_sha256": {str(workspace): mod.sha(workspace), str(summary): mod.sha(summary)}})
        card = self.base/"datacards_NIsoMuon/alpha/blind/combined/Run2Run3/datacard_M-20_Run2Run3.txt"
        mod.dump(self.base/mod.REVIEW_BASE/"run_manifest.json", {
            "cards": {"20": {"path": str(card), "sha256": mod.sha(card)}},
            "limit_json": str(self.base/"limit_outputs/limits.json"),
            "limit_sha256": mod.sha(self.base/"limit_outputs/limits.json")})
        mod.dump(self.base/"review_outputs/unique.json", {"independent": 1})
        # A previous run's workspace at the old canonical review path may differ
        # from the updated workspace. A retained run must keep that exact input.
        old_input = self.base/"review_outputs/Run2Run3/M20/workspace_Run2Run3_M20.root"
        old_input.parent.mkdir(parents=True)
        old_input.write_bytes(b"previous model workspace")
        (self.base/"bias_M20_verified.hul4v2hz").mkdir()
        (self.base/"bias_M20_verified.hul4v2hz/unknown.root").write_bytes(b"preserve unknown")
        mod.dump(self.base/"bias_M20_verified.hul4v2hz/input_manifest.json",
                 {"input_sha256": {str(old_input): mod.sha(old_input)}})
        (self.base/"bias_M20_probe.BmNJspFs").mkdir()
        (self.base/"bias_M20_probe.BmNJspFs/failed.root").write_bytes(b"superseded probe")
        self.temp_run = self.base/"limit_outputs_impacts_M12_negative.fixture"
        self.temp_run.mkdir()
        source = self.base/"limit_outputs_impacts_updated/alpha/blind/Run2Run3/workspace_Run2Run3_alpha_M12.root"
        (self.temp_run/"duplicate.root").write_bytes(source.read_bytes())
        (self.temp_run/"run.log").write_text("keep unique log")

    def plan(self):
        return mod.make_plan(self.base, self.review)

    def test_rejects_wrong_card_before_mutating(self):
        card = self.base/"datacards_NIsoMuon/alpha/blind/combined/Run2/datacard_M-12_Run2.txt"
        card.write_text("changed after review")
        with self.assertRaisesRegex(RuntimeError, "Card changed"):
            self.plan()
        self.assertTrue((self.base/mod.M20).exists())

    def test_rejects_changed_bias_toy_input(self):
        p = self.base/mod.REVIEW_BASE/"Run2Run3/M20/workspace_Run2Run3_M20.root"
        p.write_bytes(b"changed workspace")
        with self.assertRaisesRegex(RuntimeError, "Bias input changed"):
            self.plan()

    def test_rejects_incomplete_or_clipped_impacts(self):
        e = self.entries[0]
        source = Path(e['source_directory'])
        internal = source/"impacts_Run2_alpha_M12_internal.json"
        mod.dump(internal, {"POIs": [{"fit": [0, 0, 1]}]})
        e['internal_json_sha256'] = mod.sha(internal)
        mod.dump(self.review/"preservation/snapshot_manifest.json", {"b_only_impacts": self.entries})
        with self.assertRaisesRegex(RuntimeError, "negative-r"):
            self.plan()

    def test_dry_plan_changes_no_production_files(self):
        before = {str(p.relative_to(self.base)): mod.sha(p) for p in self.base.rglob('*') if p.is_file()}
        plan = self.plan()
        after = {str(p.relative_to(self.base)): mod.sha(p) for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(plan['impacts']), 39)

    def test_full_install_retains_references_inputs_and_unknown_runs(self):
        plan = self.plan()
        stage = self.root/"stage"
        stage.mkdir()
        # A similarly named old ROOT file must be removed, not paired with new JSON.
        old = self.base/"limit_outputs/alpha/blind/Run2/higgsCombine_bad.Impacts_Run2_alpha_M12.root"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"obsolete fitting output")
        inventory = mod.stage_results(self.base, plan, stage)
        backup = self.root/"backup"
        mod.install(self.base, stage, plan, inventory, backup)
        self.assertFalse(old.exists())
        for target in mod.TARGETS:
            self.assertEqual(len(list((self.base/f"limit_outputs/alpha/blind/{target}").glob("impacts_*.pdf"))), 13)
        self.assertEqual((self.base/"limit_outputs/alpha/unblind/keep.txt").read_text(), "untouched")
        self.assertEqual((self.base/"sigfit_inputs/input.root").read_bytes(), b"untouched input")
        self.assertTrue((self.base/"review_outputs/additional_runs/bias_M20_verified.hul4v2hz/unknown.root").is_file())
        unknown = mod.load(self.base/"review_outputs/additional_runs/bias_M20_verified.hul4v2hz/input_manifest.json")
        previous_workspace = Path(next(iter(unknown['input_sha256'])))
        self.assertIn("previous_review_outputs", str(previous_workspace))
        self.assertEqual(previous_workspace.read_bytes(), b"previous model workspace")
        self.assertTrue((self.base/"review_outputs/provenance/previous_review_outputs/unique.json").is_file())
        self.assertFalse((self.base/mod.M20).exists())
        self.assertTrue((backup/mod.M20).is_dir())
        for mass in (20, 70):
            bias = self.base/f"review_outputs/Run2Run3/M{mass}/bias"
            self.assertEqual(mod.load(bias/f"bias_summary_Run2Run3_M{mass}.json")[0], {"final": 0})
            inputs = mod.load(bias/"verified/input_manifest.json")
            self.assertEqual(mod.load(Path(inputs['source_summary']))[0], {"old": 0})
            for path, digest in inputs['input_sha256'].items():
                self.assertEqual(mod.sha(Path(path)), digest)
            merged = mod.load(bias/"verified/merged_toys.json")
            self.assertTrue(Path(merged['r0']['1']['source']).is_file())
        mapping = mod.load(self.base/"review_outputs/relocation_map.json")['path_prefixes']
        duplicate = str(self.temp_run/"duplicate.root")
        self.assertTrue(Path(mapping[duplicate]).is_file())
        self.assertFalse((self.base/plan['moves'][self.temp_run.name]/'duplicate.root').exists())
        self.assertTrue((self.base/plan['moves'][self.temp_run.name]/'run.log').is_file())
        # Exact original manifests remain available, even though live paths changed.
        historical = self.base/f"review_outputs/provenance/original_json/review_outputs/Run2Run3/M20/bias/verified/input_manifest.json"
        self.assertEqual(mod.load(historical)['source_summary'], str(self.base/mod.REVIEW_BASE/"Run2Run3/M20/bias/bias_summary_Run2Run3_M20.json"))

    def test_post_install_failure_rolls_back_originals(self):
        plan = self.plan()
        stage = self.root/"stage"
        stage.mkdir()
        inventory = mod.stage_results(self.base, plan, stage)
        inventory['limit_outputs/limits.json'] = "bad expected digest"
        with self.assertRaisesRegex(RuntimeError, "Installed hash mismatch"):
            mod.install(self.base, stage, plan, inventory, self.root/"backup")
        self.assertTrue((self.base/mod.M20).is_dir())
        self.assertTrue((self.base/mod.REVIEW_BASE).is_dir())
        self.assertEqual(mod.load(self.base/"review_outputs/unique.json"), {"independent": 1})
        self.assertTrue((self.base/"bias_M20_probe.BmNJspFs/failed.root").is_file())

    def test_symlink_in_retired_output_is_not_followed_or_deleted(self):
        (self.base/mod.M20/"linked_input").symlink_to(self.base/"sigfit_inputs")
        with self.assertRaisesRegex(RuntimeError, "symbolic link"):
            self.plan()

    def test_dangling_repair_reference_blocks_cleanup(self):
        source = self.base/"bias_repair_M20.WBgJ4pg8"
        source.mkdir()
        (source/"required.root").write_bytes(b"not disposable")
        mod.dump(self.base/mod.M20/"extra.json", {"source": str(source/"required.root")})
        with self.assertRaisesRegex(RuntimeError, "Unpreserved dependency"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
