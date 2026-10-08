"""Check shared QCD MC statistical parameters and nominal blinded card rates."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w


def channel(era, nominal, local, common, identity="run2", group="Run2"):
    rates = dict(sig=1., QCD=nominal, tt=2., ST=3., DY=4., Others=5.)
    return w.ChannelResult(year=era, mass_label="15", mass=15., window_low=14.8, window_high=15.2,
           observation=0., raw_rates=rates, rates=dict(rates), nuisances={},
           qcd_shape_sigma_down=1., qcd_shape_sigma_up=2.,
           qcd_stat_sigma=math.hypot(local, common), qcd_nf_stat_sigma=local-1., qcd_fit_stat_sigma=1.,
           qcd_local_stat_sigma=local, qcd_mc_transfer_stat_sigma=common,
           qcd_common_transfer_group=group if identity else None, qcd_common_transfer_id=identity)


class CommonQCDCardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = w.build_parser().parse_args(["--stage", "cards", "--parameter", "yield", "--dy-method", "mc"])
        self.args.card_base = self.temp.name

    def card(self, channels, target="Run2"):
        return w.write_datacard(self.args, target, channels, 1., None, None).path

    def test_common_parameter_is_emitted_once_and_shifts_both_eras(self):
        path = self.card([channel("2016preVFP", 100., 10., 20.), channel("2016postVFP", 200., 20., 40.)])
        text = path.read_text()
        w._validate_background_card(self.args, path)
        common = "CMS_NPS26009_QCD_MCTransferStat_Run2"
        self.assertEqual(text.count(common+" param 0 1"), 1)
        self.assertEqual(text.count(common), 3)
        obs, _, bkg = w.counting_card_observation_signal_background(path)
        self.assertEqual(obs, [114., 214.])
        self.assertEqual(obs, bkg)
        shifted = text.replace(common+" param 0 1", common+" param 1 1")
        _, _, shifted_bkg = w.counting_card_observation_signal_background(path, card_text=shifted)
        self.assertEqual(shifted_bkg, [134., 254.])
        # Local and common components appear exactly once, and use different correlations.
        names = {w.nuisance_global_name("QCD_stat", era) for era in ("2016preVFP", "2016postVFP")}
        local_lines = [line.split() for line in text.splitlines() if line.split() and line.split()[0] in names
                       and " param " in line]
        self.assertEqual(sorted(float(row[3]) for row in local_lines), [10., 20.])

    def test_runs_are_independent_and_legacy_formula_remains(self):
        path = self.card([channel("2016preVFP", 100., 10., 20.),
                          channel("2022", 200., 20., 40., identity="run3", group="Run3")], "Run2Run3")
        text = path.read_text()
        self.assertEqual(text.count("MCTransferStat_Run2 param 0 1"), 1)
        self.assertEqual(text.count("MCTransferStat_Run3 param 0 1"), 1)
        legacy = self.card([channel("2016preVFP", 100., 10., 0., identity=None)])
        self.assertIn("max(0.0,@0+@1) ", legacy.read_text())
        self.assertNotIn("MCTransferStat", legacy.read_text())

    def test_mixed_or_stale_common_inputs_are_rejected(self):
        for second in (channel("2016postVFP", 200., 20., 40., identity="different"),
                       channel("2016postVFP", 200., 20., 40., identity=None)):
            with self.assertRaises(w.WorkflowError):
                self.card([channel("2016preVFP", 100., 10., 20.), second])

    def test_formula_parser_rejects_missing_or_unsupported_parameters(self):
        text = "shape param 100 10\nstat param 0 5\nmc param 0 1\ny rateParam bin_2018 QCD max(0.0,@0+@1+20*@2) shape,stat,mc"
        self.assertEqual(w.counting_card_nominal_rate_params(text)[0][2], 100.)
        with self.assertRaises(w.WorkflowError):
            w.counting_card_nominal_rate_params(text.replace("mc param 0 1\n", ""))
        with self.assertRaises(w.WorkflowError):
            w.counting_card_nominal_rate_params(text.replace("20*@2", "20*sqrt(@2)"))

    def test_contract_rejects_missing_shared_constraint(self):
        path = self.card([channel("2016preVFP", 100., 10., 20.)])
        path.write_text(path.read_text().replace("CMS_NPS26009_QCD_MCTransferStat_Run2 param 0 1\n", ""))
        with self.assertRaises(w.WorkflowError): w._validate_background_card(self.args, path)


if __name__ == "__main__":
    unittest.main()
