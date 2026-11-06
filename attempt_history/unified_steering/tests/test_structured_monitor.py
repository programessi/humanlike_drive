"""New-skill failure must not masquerade as forgetting an accepted skill."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from unified_steering.structured.train import GateCallback


class StructuredMonitorTests(unittest.TestCase):
    def callback(self, output):
        callback = GateCallback(Path(output), "R2", {})
        callback.model = SimpleNamespace(save=lambda *a: None,
                                        logger=SimpleNamespace(record=lambda *a: None, dump=lambda *a: None))
        return callback

    def run_evaluations(self, callback, retained_peak):
        records = [dict(case="hold", support="both", complete=True, peak_error_deg=retained_peak, rmse_deg=.4),
                   dict(case="hold", support="left", complete=True, peak_error_deg=2.4, rmse_deg=.9)]
        records += [dict(case=name, support="left", complete=True, peak_error_deg=2., rmse_deg=1.2)
                    for name in ("positive", "negative", "sequence")]
        with patch("unified_steering.structured.train.stage_suite", return_value=dict(cases=records, passed=False)):
            with redirect_stdout(StringIO()):
                for step in (0, 16384, 32768, 49152):
                    callback.num_timesteps = step
                    callback.check()

    def test_unacquired_single_support_does_not_trigger_forgetting_stop(self):
        with tempfile.TemporaryDirectory() as output:
            callback = self.callback(output)
            self.run_evaluations(callback, .8)
            self.assertEqual(callback.stop_reason, "")
            self.assertEqual(callback.stale_bad, 0)
            self.assertFalse(callback.qualified)

    def test_retained_hold_regression_without_response_progress_stops(self):
        with tempfile.TemporaryDirectory() as output:
            callback = self.callback(output)
            self.run_evaluations(callback, 1.5)
            self.assertEqual(callback.stop_reason, "three_bad_evaluations_requires_diagnosis")
            self.assertFalse(callback.qualified)

    def test_formal_pass_cannot_bypass_declared_precision_margin(self):
        with tempfile.TemporaryDirectory() as output:
            callback=self.callback(output)
            callback.qualification_rmse=.9
            records=[dict(case="hold",support="both",complete=True,peak_error_deg=.4,rmse_deg=.2)]
            records += [dict(case=name,support="left",complete=True,peak_error_deg=1.,rmse_deg=.95)
                        for name in ("positive","negative","sequence")]
            with patch("unified_steering.structured.train.stage_suite",return_value=dict(cases=records,passed=True)):
                with redirect_stdout(StringIO()):
                    for step in (0,16384):
                        callback.num_timesteps=step
                        callback.check()
            self.assertTrue(callback.records[-1]["passed"])
            self.assertFalse(callback.records[-1]["qualification_passed"])
            self.assertFalse(callback.qualified)

    def test_v2_incomplete_reverse_sequence_cannot_win_on_short_prefix_rmse(self):
        with tempfile.TemporaryDirectory() as output:
            callback=self.callback(output)
            callback.include_reverse_sequence=True
            records=[dict(case='hold',support='both',complete=True,peak_error_deg=.4,rmse_deg=.2)]
            records += [dict(case=name,support='left',complete=True,peak_error_deg=1.,rmse_deg=.6)
                        for name in ('positive','negative','sequence')]
            records += [dict(case='sequence_negative',support='right',complete=False,peak_error_deg=.2,rmse_deg=.1)]
            with patch('unified_steering.structured.train.stage_suite',return_value=dict(cases=records,passed=False)):
                with redirect_stdout(StringIO()):
                    callback.num_timesteps=0
                    callback.check()
            self.assertGreater(callback.records[-1]['score'],1000.)
            self.assertEqual(callback.best_score,float('inf'))
            self.assertFalse(callback.qualified)


if __name__ == "__main__":
    unittest.main()
