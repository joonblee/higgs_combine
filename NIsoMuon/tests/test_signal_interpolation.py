"""Signal scan integration checks; synthetic inputs, no ROOT/Combine dependency."""
import contextlib
import copy
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w


def arguments(*extra):
    args = w.build_parser().parse_args([
        "--stage", "cards", "--target", "runs", "--qcd-method", "mc",
        "--dy-method", "mc", "--no-generator-theory", *extra,
    ])
    args.resolution_coefficients = {era: (0., .01 + .001 * i) for i, era in enumerate(w.YEARS)}
    args.resolution_source = Path("synthetic_resolution.json")
    return args


def signal_files(args, masses=(12., 15., 20., 70.)):
    return {m: {era: str(Path(w.root_dir(args, era)) / f"NIsoMuon_Zp_M-{m:g}.root")
                for era in w.YEARS} for m in masses}


class SyntheticReader:
    """Analytic log-linear signal and window-dependent backgrounds."""
    def __init__(self):
        self.calls = []
        self.missing = None
        self.bad_value = None
        self.qcd_calls = []

    def close(self):
        pass

    def integral(self, filename, path, low, high):
        self.calls.append((filename, path, low, high))
        parsed = w.signal_mass_from_filename(filename)
        if parsed:
            if self.missing and self.missing in path:
                return None
            mass = parsed[1]
            era = next(era for era in w.YEARS if era in Path(filename).parts)
            value = 100. * (1 + w.YEARS.index(era)) * math.exp(.03 * mass)
            if "Down" in path:
                value *= math.exp(-.002 * mass)
            if "Up" in path:
                value *= math.exp(.002 * mass)
            if self.bad_value is not None:
                value = self.bad_value
            return w.YieldResult(value, abs(value) * .1, 1, 10)
        if filename.endswith("data.root"):
            return w.YieldResult(123., math.sqrt(123.), 1, 10)
        if path == w.DY_AUX_NF_AMC_PATH:
            return w.YieldResult(.3, .03, 1, 1)
        if path == w.DY_AUX_NF_MG_PATH:
            return w.YieldResult(.4, .02, 1, 1)
        value = (high - low) * (10 + .05 * (high + low))
        if "Down" in path:
            value *= .8
        if "Up" in path:
            value *= 1.25
        return w.YieldResult(value, math.sqrt(value), 1, 10)

    def qcd_statistics(self, filename, path, low, high, era, nominal):
        self.qcd_calls.append((era, low, high))
        d = high - low
        return dict(sigma_stat_bound=math.hypot(2*d, .5*d), sigma_nf_stat=d,
                    sigma_fit_stat=d, sigma_local_stat_bound=2*d,
                    sigma_mc_transfer_stat=.5*d, common_transfer_group=w.run_group(era),
                    common_transfer_id=w.run_group(era), covariance_status=3,
                    covariance_regularised=False, boundary_parameters=[])


class GridTests(unittest.TestCase):
    def test_grid_is_frozen_identical_for_all_targets_and_retains_anchors(self):
        expected = None
        for target in (*w.YEARS, "Run2", "Run3", "Run2+3", "runs"):
            args = arguments("--interpolate-signal", "--target", target)
            w.prepare_mass_scan(args)
            if expected is None:
                expected = args.masses
            self.assertEqual(args.masses, expected)
            self.assertEqual(args.card_base, "datacards_NIsoMuon_interpolated")
        self.assertEqual(len(expected), 229)
        self.assertEqual((expected[0], expected[-1]), (12., 70.))
        self.assertTrue(set([12, 15, *range(20, 71, 5)]).issubset(expected))
        ref = args.mass_grid_payload["reference"]
        for m1, m2 in zip(expected, expected[1:]):
            self.assertLessEqual(m2-m1, .5*m1*(ref["a"]*m1+ref["b"])+1e-9)
            self.assertAlmostEqual(float(w.mass_label(m1).replace("p", ".")), m1)

    def test_default_and_explicit_hypotheses(self):
        args = arguments()
        w.prepare_mass_scan(args)
        self.assertIsNone(args.masses)
        self.assertEqual(args.card_base, "datacards_NIsoMuon")
        args = arguments("--interpolate-signal", "--masses", "20,12.5", "--card-base", "custom")
        w.prepare_mass_scan(args)
        self.assertEqual(args.masses, [12.5, 20.])
        self.assertEqual(args.card_base, "custom")
        with self.assertRaises(w.WorkflowError):
            w.prepare_mass_scan(arguments("--mass-grid-file", "unused.json"))
        with self.assertRaises(w.WorkflowError):
            w.prepare_mass_scan(arguments("--interpolate-signal", "--masses", "11,70"))

    def test_invalid_grids_are_rejected(self):
        for values in ([], [12, 12], [math.nan], [math.inf], [0], [12.000000001, 12.000000002]):
            with self.subTest(values=values), self.assertRaises(w.WorkflowError):
                w.validate_scan_masses(values)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"grid.json"
            path.write_text(json.dumps({"schema_version": 1, "masses": [20, 12]}))
            with self.assertRaises(w.WorkflowError):
                w.prepare_mass_scan(arguments("--interpolate-signal", "--mass-grid-file", str(path)))


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        self.args = arguments("--interpolate-signal", "--masses", "12,15,17.5,20")
        w.prepare_mass_scan(self.args)
        self.reader = SyntheticReader()
        self.model = w.SignalInterpolator(self.reader, self.args, signal_files(self.args), w.YEARS)

    def test_log_yield_and_derivative_error_and_no_extrapolation(self):
        a, b = w.YieldResult(100, 10, 1, 10), w.YieldResult(400, 40, 1, 10)
        result = w.interpolate_signal_yield(17.5, 15, a, 20, b)
        self.assertAlmostEqual(result.value, 200)
        self.assertAlmostEqual(result.error, 200*math.sqrt(.005))
        self.assertIs(w.interpolate_signal_yield(15, 15, a, 15, a), a)
        for mass in (11, 71):
            with self.assertRaises(w.WorkflowError):
                self.model.bracket("2018", mass)
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(w.WorkflowError):
                w.interpolate_signal_yield(17.5, 15, w.YieldResult(value, 1, 1, 10), 20, b)

    def test_exact_sample_channels_equal_original_and_blind_never_reads_data(self):
        files = signal_files(self.args)[15.]
        original, _ = w.build_channels_for_mass(self.reader, self.args, 15., files, w.YEARS)
        proxy, same_files = self.model.reader_for_mass(15.)
        actual, _ = w.build_channels_for_mass(proxy, self.args, 15., same_files, w.YEARS)
        self.assertEqual({era: asdict(ch) for era, ch in original.items()},
                         {era: asdict(ch) for era, ch in actual.items()})
        self.assertFalse(any(call[0].endswith("data.root") for call in self.reader.calls))

    def test_midpoint_signal_systematics_and_era_windows(self):
        self.model.preflight(self.args.masses)
        proxy, files = self.model.reader_for_mass(17.5)
        actual, audit = w.build_channels_for_mass(proxy, self.args, 17.5, files, w.YEARS)
        self.assertTrue(all(e.found == e.expected for e in audit.entries.values()))
        for era, ch in actual.items():
            expected = 100*(1+w.YEARS.index(era))*math.exp(.03*17.5)
            self.assertAlmostEqual(ch.raw_rates["sig"], expected)
            kd, ku = map(float, ch.nuisances["jes"]["sig"].split("/"))
            self.assertAlmostEqual(kd, math.exp(-.002*17.5), places=5)
            self.assertAlmostEqual(ku, math.exp(.002*17.5), places=5)
            self.assertAlmostEqual(ch.raw_rates["tt"],
                                   (ch.window_high-ch.window_low)*(10+.05*(ch.window_high+ch.window_low)))
        self.assertNotEqual(actual["2018"].window_low, actual["2022"].window_low)
        # Signal anchors were integrated at 15/20, never at the intermediate window.
        for filename, path, low, high in self.reader.calls:
            parsed = w.signal_mass_from_filename(filename)
            if parsed:
                era = next(e for e in w.YEARS if e in Path(filename).parts)
                self.assertEqual((low, high), w.mass_window(self.args, era, parsed[1]))

    def test_missing_systematic_and_missing_era_fail_without_strict(self):
        self.reader.missing = "JetEnUp"
        with self.assertRaisesRegex(w.WorkflowError, "Missing signal interpolation anchor"):
            self.model.preflight([17.5])
        files = signal_files(self.args)
        for mass in files:
            files[mass].pop("2023")
        with self.assertRaisesRegex(w.WorkflowError, "2023"):
            w.SignalInterpolator(SyntheticReader(), self.args, files, w.YEARS).preflight([17.5])

    def test_dd_backgrounds_are_reintegrated_and_qcd_statistics_delegated(self):
        self.args.qcd_method = self.args.dy_method = "data-driven"
        proxy, files = self.model.reader_for_mass(17.5)
        # Input schema audits are tested separately; exercise the complete channel
        # and rateParam paths using deterministic auxiliary values here.
        with patch.object(w, "_audit_dy_background_inputs"), patch.object(w, "_audit_qcd_background_central"):
            with contextlib.redirect_stdout(io.StringIO()):
                actual, _ = w.build_channels_for_mass(proxy, self.args, 17.5, files, w.YEARS)
                direct, _ = w.build_channels_for_mass(self.reader, self.args, 17.5,
                                                      signal_files(self.args)[15.], w.YEARS)
        for era in w.YEARS:
            for process in ("QCD", "DY", "tt", "ST", "Others"):
                self.assertEqual(actual[era].raw_rates[process], direct[era].raw_rates[process])
            self.assertEqual(actual[era].qcd_stat_sigma, direct[era].qcd_stat_sigma)
            self.assertEqual(actual[era].dy_lightjet_error, direct[era].dy_lightjet_error)
        self.assertTrue(self.reader.qcd_calls)
        for era, low, high in self.reader.qcd_calls:
            self.assertEqual((low, high), w.mass_window(self.args, era, 17.5))

    def test_closure_and_xsec(self):
        self.model.preflight(self.args.masses)
        with tempfile.TemporaryDirectory() as temp:
            self.model.write_audit(Path(temp), self.args.masses)
            data = json.loads((Path(temp)/"signal_interpolation_audit.json").read_text())
        self.assertEqual(len(data["leave_one_out_closure"]), 8)
        for row in data["leave_one_out_closure"]:
            self.assertAlmostEqual(row["relative_yield_residual"], 0)
            for residual in row["systematic_ratio_relative_residuals"].values():
                self.assertAlmostEqual(residual, 0)
        self.assertAlmostEqual(w.lookup_xsec(self.args, {15: 100., 20: 400.}, 17.5), 200.)
        with self.assertRaises(w.WorkflowError):
            w.lookup_xsec(self.args, {15: 100., 20: 400.}, 12.)

    def test_fractional_cards_and_alpha_normalisation_all_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            self.args.card_base = temp
            with patch.object(w, "RootReader", return_value=self.reader), \
                    patch.object(w, "scan_signal_files", return_value=signal_files(self.args)), \
                    contextlib.redirect_stdout(io.StringIO()):
                cards = w.build_cards(self.args)
            self.assertEqual(len(cards), 12)
            for target in ("Run2", "Run3", "Run2Run3"):
                found = w.cards_for_target(self.args, target)
                self.assertEqual(len(found), 4)
                for card in found:
                    label, mass = w.extract_mass_from_card(card, target)
                    observation, signal, background = w.counting_card_observation_signal_background(card)
                    for observed, expected in zip(observation, background):
                        self.assertAlmostEqual(observed, expected, places=8)
                    self.assertAlmostEqual(sum(signal), 25., places=8)
                    ref = sum(100*(1+w.YEARS.index(era))*math.exp(.03*mass) for era in w.target_years(target))
                    self.assertAlmostEqual(w.alpha_unit_from_card(card), 25./ref)
                    self.assertEqual(float(w.format_mass_for_combine(mass)), mass)
            removed = w.cards_for_target(self.args, "Run2")[0]
            removed.unlink()
            with self.assertRaisesRegex(w.WorkflowError, "Missing 1 scan cards"):
                w.cards_for_target(self.args, "Run2")
            card = w.cards_for_target(self.args, "Run3")[0]
            card.write_text(card.read_text().replace(self.args.mass_grid_sha256, "stale"))
            with self.assertRaisesRegex(w.WorkflowError, "provenance/grid mismatch"):
                w.cards_for_target(self.args, "Run3")

    def test_signal_preflight_failure_preserves_existing_cards(self):
        with tempfile.TemporaryDirectory() as temp:
            self.args.card_base = temp
            old = w.datacard_path(self.args, "Run2", "15")
            old.parent.mkdir(parents=True)
            old.write_text("preserved")
            self.reader.missing = "JetEnUp"
            with patch.object(w, "RootReader", return_value=self.reader), \
                    patch.object(w, "scan_signal_files", return_value=signal_files(self.args)):
                with self.assertRaises(w.WorkflowError):
                    w.build_cards(self.args)
            self.assertEqual(old.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
