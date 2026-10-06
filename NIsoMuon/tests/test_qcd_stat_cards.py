from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w
import combine_review as review


class QCDStatCards(unittest.TestCase):
    def test_missing_stat_metadata_does_not_remove_existing_cards(self):
        args = w.build_parser().parse_args(["--stage", "cards", "--target", "2018"])
        reader = NS(_file=lambda filename: None, close=lambda: None)
        with patch.object(w, "RootReader", return_value=reader), patch.object(w, "clean_card_targets") as clean:
            with self.assertRaisesRegex(w.WorkflowError, "existing cards were not removed"):
                w.build_cards(args)
            clean.assert_not_called()

    def test_stat_formula_preserves_asimov_and_review_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            args = w.build_parser().parse_args(["--stage", "cards", "--target", "2018", "--parameter", "yield"])
            args.card_base = directory
            args.dy_method = "mc"
            rates = dict(sig=2.0, QCD=103.403700772, tt=7.0, ST=2.0, DY=3.0, Others=.2)
            ch = w.ChannelResult(year="2018", mass_label="12", mass=12,
                window_low=11.88, window_high=12.12, observation=0,
                raw_rates=rates, rates=rates.copy(), nuisances={},
                qcd_shape_sigma_down=3.468021565, qcd_shape_sigma_up=3.468021565,
                qcd_stat_sigma=24.2100767845, qcd_nf_stat_sigma=18.5318826433,
                qcd_fit_stat_sigma=5.6781941413)
            info = w.write_datacard(args, "2018", [ch], 1, None, None)
            text = info.path.read_text()
            self.assertIn("CMS_NPS26009_stat_QCD_BJetOS_2018 param 0 24.210076785", text)
            self.assertIn("CMS_NPS26009_bckgShape_QCD_BJetOS_2018 param 103.403700772 3.468021565", text)
            self.assertIn("max(0,@0+@1)", text)
            obs, sig, bkg = w.counting_card_observation_signal_background(info.path)
            self.assertAlmostEqual(obs[0], 115.603700772)
            self.assertAlmostEqual(bkg[0], obs[0])
            self.assertEqual(sig, [2])
            bins, names, ids, effective = review.parse_rate_block(info.path)
            self.assertAlmostEqual(effective[names.index("QCD")], rates["QCD"])
            self.assertAlmostEqual(review.safe_negative_rmin(info.path, -100), -obs[0]/2 + 1e-4*obs[0]/2)
            w._validate_background_card(args, info.path)
            info.path.write_text(text.replace("max(0,@0+@1)", "@0*@1"))
            with self.assertRaises(w.WorkflowError):
                w.counting_card_observation_signal_background(info.path)
            with self.assertRaises(w.WorkflowError):
                w._validate_background_card(args, info.path)

    def test_existing_numeric_dy_qcd_modifiers_remain_supported(self):
        card = """bin bin_2018
observation 13.4
bin bin_2018 bin_2018 bin_2018
process Zprime QCD DY
process 0 1 2
rate 2 1 12
shape rateParam bin_2018 QCD 11 [0,50]
shape param 11 3
nf rateParam bin_2018 DY .2 [0,1]
nf param .2 .01
"""
        obs, sig, bkg = w.counting_card_observation_signal_background(Path("unused"), card_text=card)
        self.assertEqual(obs, [13.4])
        self.assertEqual(sig, [2])
        self.assertAlmostEqual(bkg[0], 13.4)


if __name__ == "__main__":
    unittest.main()
