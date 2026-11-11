"""Regression from the Run-3 metadata reported by the analysis on 2026-11-11."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as base


class CommonTransferTests(unittest.TestCase):
    def transfer(self, era='2022'):
        common=json.loads((Path(__file__).parent/'fixtures/qcd_common_transfer_run3.json').read_text())
        return dict(common_transfer=common, windows=copy.deepcopy(common['windows']),
                    mc_double_ratio=common['double_ratio'], double_ratio_variance=common['variance'],
                    primitive=copy.deepcopy(common['records'][era]['primitive']))

    def test_reported_metadata_validates_without_changing_saved_values(self):
        for era in base.GROUP_ERAS['Run3']:
            transfer=self.transfer(era)
            before=copy.deepcopy(transfer)
            result=base.validate_common_transfer(transfer,era)
            self.assertIs(result,transfer['common_transfer'])
            self.assertEqual(transfer,before)

    def test_last_bit_changes_are_accepted_but_real_changes_rejected(self):
        transfer=self.transfer()
        row=transfer['common_transfer']['rows']['2023']
        for key in ('variance','log_variance','error_low','error_high'):
            row[key]=math.nextafter(row[key],math.inf)
        base.validate_common_transfer(transfer,'2022')
        row['variance']*=1.000001
        with self.assertRaisesRegex(ValueError,r'rows.2023.variance.*saved=.*rebuilt='):
            base.validate_common_transfer(transfer,'2022')

    def test_discrete_decisions_structure_and_nonfinite_values_still_fail(self):
        for key,value in (('within_1sigma',True),('within_1sigma',0),
                          ('variance',float('inf')),('variance',float('nan'))):
            with self.subTest(key=key,value=value):
                transfer=self.transfer()
                transfer['common_transfer']['rows']['2023'][key]=value
                with self.assertRaises(ValueError):
                    base.validate_common_transfer(transfer,'2022')
        transfer=self.transfer()
        del transfer['common_transfer']['rows']['2023']['effective_counts']['mc_os_low']
        with self.assertRaisesRegex(ValueError,'keys disagree'):
            base.validate_common_transfer(transfer,'2022')

    def test_input_identity_windows_and_applied_factor_checks_remain_strict(self):
        transfer=self.transfer()
        transfer['common_transfer']['id']='different'
        with self.assertRaisesRegex(ValueError,'identity disagree'):
            base.validate_common_transfer(transfer,'2022')
        transfer=self.transfer()
        transfer['windows']['high'][1]=79.
        with self.assertRaisesRegex(ValueError,'identity disagree'):
            base.validate_common_transfer(transfer,'2022')
        transfer=self.transfer()
        transfer['primitive']['mc_os_low']['value']+=1.
        with self.assertRaisesRegex(ValueError,'local MC primitive'):
            base.validate_common_transfer(transfer,'2022')
        transfer=self.transfer()
        transfer['mc_double_ratio']*=1.01
        with self.assertRaisesRegex(ValueError,'Applied QCD'):
            base.validate_common_transfer(transfer,'2022')


if __name__=='__main__':
    unittest.main()
