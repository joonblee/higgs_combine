"""Regression for ROOT Float_t masses in CollectLimits alpha conversion."""
import ctypes
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w


class RootMassPrecisionTests(unittest.TestCase):
    def test_reported_failure_and_all_frozen_grid_masses(self):
        self.assertEqual(ctypes.c_float(32.553).value, 32.553001403808594)
        self.assertEqual(w.find_value_for_mass({32.553: .1}, 32.553001403808594, "alpha unit"), .1)
        masses = json.loads(Path(w.__file__).with_name("mass_scan_grid.json").read_text())["masses"]
        mapping = {mass: index+1 for index, mass in enumerate(masses)}
        for mass in masses:
            with self.subTest(mass=mass):
                self.assertEqual(w.find_value_for_mass(mapping, mass, "alpha unit"), mapping[mass])
                self.assertEqual(w.find_value_for_mass(mapping, ctypes.c_float(mass).value,
                                                       "alpha unit"), mapping[mass])

    def test_all_grid_quantiles_use_their_own_alpha_unit_and_range(self):
        masses = json.loads(Path(w.__file__).with_name("mass_scan_grid.json").read_text())["masses"]
        units = {mass: (i+1)*1e-4 for i, mass in enumerate(masses)}
        ranges = {mass: 100. for mass in masses}
        payload = {str(ctypes.c_float(mass).value): {key: 2. for key in (*w.EXPECTED_LIMIT_KEYS, "obs")}
                   for mass in masses}
        args = w.build_parser().parse_args(["--mode", "unblind"])
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/"internal.json", Path(temp)/"physical.json"
            source.write_text(json.dumps(payload))
            w.rescale_limit_json(args, source, target, units, ranges)
            result = json.loads(target.read_text())
        self.assertEqual(len(result), 229)
        for mass in masses:
            for value in result[str(ctypes.c_float(mass).value)].values():
                self.assertAlmostEqual(value, 2.*units[mass])

    def test_unrelated_and_ambiguous_masses_fail(self):
        with self.assertRaisesRegex(w.WorkflowError, "No alpha unit"):
            w.find_value_for_mass({32.553: .1}, 32.55302, "alpha unit")
        rounded = ctypes.c_float(32.553).value
        with self.assertRaisesRegex(w.WorkflowError, "Ambiguous alpha unit"):
            w.find_value_for_mass({32.553: .1, 32.5530001: .2}, rounded, "alpha unit")


if __name__ == "__main__":
    unittest.main()
