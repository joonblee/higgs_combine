"""External covariance must be audited, never accepted solely from covQual=-1."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_spectrum_fit as fs


class Parameter:
    def __init__(self, name): self.name = name
    def GetName(self): return self.name
    def isConstant(self): return False


class Parameters(list):
    def find(self, name): return next((p for p in self if p.GetName() == name), None)


class RobustHesseTest(unittest.TestCase):
    def audit(self, log=None, matrix=None, names=('n',)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'step1.log'
            path.write_text(log or 'Verifying eigenvalues are all positive (iteration 0)\nAll eigenvalues are positive\n')
            config = SimpleNamespace(GetNuisanceParameters=lambda: [Parameter('n')],
                                     GetParametersOfInterest=lambda: [Parameter('r')])
            workspace = SimpleNamespace(obj=lambda name: config)
            fit = SimpleNamespace(floatParsFinal=lambda: Parameters(Parameter(n) for n in names),
                                  covarianceMatrix=lambda: np.array([[1.]]) if matrix is None else matrix)
            return fs.audit_robust_covariance(fit, workspace, path)

    def test_full_external_covariance(self):
        audit = self.audit()
        self.assertTrue(audit['validated'])
        self.assertTrue(fs.fit_quality_ok(dict(status=0, covQual=-1, covariance_audit=audit)))
        self.assertFalse(fs.fit_quality_ok(dict(status=0, covQual=-1)))
        self.assertFalse(fs.fit_quality_ok(dict(status=0, covQual=2)))
        self.assertFalse(fs.fit_quality_ok(dict(status=1, covQual=-1, covariance_audit=audit)))

    def test_reject_parameter_removal_or_bad_covariance(self):
        for log in ('All eigenvalues are positive',
                    'Verifying eigenvalues are all positive (iteration 1)\nAll eigenvalues are positive'):
            with self.assertRaises(ValueError): self.audit(log=log)
        with self.assertRaises(ValueError): self.audit(names=('other',))
        for matrix in (np.array([[0.]]), np.array([[-1.]]), np.array([[float('nan')]])):
            with self.assertRaises(ValueError): self.audit(matrix=matrix)

    def test_opt_in_command_does_not_change_model_signature(self):
        ordinary = fs.make_parser().parse_args(['--masses','70'])
        robust = fs.make_parser().parse_args(['--masses','70','--fit-robust-hesse'])
        self.assertNotIn('--robustHesse', fs.fit_commands(ordinary,70,Path('/tmp/example'),'fit_b')[1])
        self.assertIn('--robustHesse', fs.fit_commands(robust,70,Path('/tmp/example'),'fit_b')[1])
        self.assertEqual(fs.model_directory(ordinary,70),fs.model_directory(robust,70))
        only = fs.make_parser().parse_args(['--fit-components','s-only'])
        self.assertEqual(fs.selected_fits(only),('fit_s',))
        self.assertIn('--skipBOnlyFit',fs.fit_commands(only,70,Path('/tmp/example'))[1])


if __name__ == '__main__': unittest.main()
