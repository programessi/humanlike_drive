"""The revised grid covers reverse sequences without changing the old protocol."""
import unittest
from unittest.mock import patch

from unified_steering.structured.env import TaskConfig
from unified_steering.structured.evaluate import evaluation_cases, run_case
from unified_steering.structured.parking import SupportTeacherEnv


class StructuredEvaluationTests(unittest.TestCase):
    def test_versioned_grid_keeps_old_cases_and_adds_reverse(self):
        old, new = evaluation_cases(), evaluation_cases(True)
        self.assertEqual(len(old), 7)
        self.assertEqual(new[:-1], old)
        self.assertEqual(new[-1], ("sequence_negative", 22., 10.))
        self.assertEqual(len(new)*3*3, 72)

    def test_reverse_case_sets_actual_reference_direction(self):
        config=TaskConfig(hand_action_mode="leaky", hand_leak_alpha=.995)
        directions=[]
        original=SupportTeacherEnv.reset
        def capture(env, *args, **kwargs):
            result=original(env,*args,**kwargs)
            directions.append((env.episode_direction,env.command_angles[-1]-env.command_angles[0]))
            return result
        with patch.object(SupportTeacherEnv,"reset",capture):
            record,_=run_case(None,config,"sequence_negative",1007,.05,10.,"right",
                              fixed=True,env_class=SupportTeacherEnv)
        self.assertEqual(directions[0][0],-1)
        self.assertEqual(record["reset_options"],{})
        self.assertTrue(record["complete"])


if __name__ == "__main__":
    unittest.main()
