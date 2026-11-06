"""Exercise curriculum sampling without changing evaluation or physics."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from unified_steering.structured.parking import SupportTeacherEnv
from unified_steering.structured.train import CourseEnv, validate_mode_probabilities
from unified_steering.structured.curriculum import balanced_condition


class StructuredCurriculumTests(unittest.TestCase):
    def env(self):
        env = CourseEnv.__new__(CourseEnv)
        env.config = SimpleNamespace(amplitude_deg=10.)
        env.stage, env.parking, env.hold_fraction = "R2", "nominal", None
        env.mode_probabilities = (.15, .15, .05, .65)
        env._np_random = np.random.default_rng(7)
        return env

    def test_sequence_curriculum_retains_supports_amplitudes_and_explicit_options(self):
        env = self.env()
        modes, supports, amplitudes = Counter(), Counter(), Counter()
        with patch.object(SupportTeacherEnv, "reset", side_effect=lambda **kw: kw["options"]):
            for _ in range(12000):
                selected = env.reset()
                modes[selected["mode"]] += 1
                supports[selected["free_side"]] += 1
                amplitudes[env.config.amplitude_deg] += 1
            explicit = env.reset(seed=1007, options={"mode": "hold", "free_side": -1})
        for name, fraction in zip(("hold", "step", "sine", "sequence"), env.mode_probabilities):
            self.assertAlmostEqual(modes[name]/12000, fraction, delta=.02)
        for side, fraction in zip((-1, 0, 1), (.25, .375, .375)):
            self.assertAlmostEqual(supports[side]/12000, fraction, delta=.02)
        self.assertAlmostEqual(amplitudes[10.]/12000, .5, delta=.02)
        self.assertEqual(explicit, {"mode": "hold", "free_side": -1})

    def test_invalid_or_ambiguous_curricula_rejected(self):
        for values in ((.1, .1), (.2, .2, .2, .2), (0., .2, .3, .5),
                       (np.nan, .2, .3, .5), (-.1, .3, .3, .5)):
            with self.assertRaises(ValueError):
                validate_mode_probabilities(values)
        with self.assertRaises(ValueError):
            validate_mode_probabilities((.15, .15, .05, .65), .25)

    def test_worker_strides_cover_balanced_blocks_and_both_sequence_signs(self):
        for block in (0,1):
            conditions=[balanced_condition(block*64+worker+4*episode,7)
                        for worker in range(4) for episode in range(16)]
            self.assertEqual(len({r['course_slot'] for r in conditions}),64)
            self.assertEqual(Counter(r['free_side'] for r in conditions),{-1:16,1:24,0:24})
            for side,repeats in ((-1,2),(1,3),(0,3)):
                for sign in (-1,1):
                    matches=[r for r in conditions if r['free_side']==side and r['mode']=='sequence' and r['direction']==sign]
                    self.assertEqual(len(matches),repeats)
                    self.assertTrue(all(r['training_amplitude_deg']==10 for r in matches))

    def test_explicit_training_amplitude_is_used_before_parent_reset(self):
        env=self.env()
        with patch.object(SupportTeacherEnv,'reset',side_effect=lambda **kw:kw['options']):
            chosen=env.reset(seed=1007,options={'training_amplitude_deg':10.,'mode':'sequence','direction':-1})
        self.assertEqual(env.config.amplitude_deg,10.)
        self.assertNotIn('training_amplitude_deg',chosen)
        self.assertEqual(chosen['direction'],-1)


if __name__ == "__main__":
    unittest.main()
