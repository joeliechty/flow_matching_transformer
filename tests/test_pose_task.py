"""Task files and the conditions derived from them.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import os
import tempfile
import unittest

import torch

from pose_gen_inference import build_obs_from_actions
from utils.pose_task import load_pose_task, task_conditions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOUR_CORNERS = os.path.join(REPO, 'configs', 'pose_tasks', 'four_corners.yaml')

# The pose trainer's hard-coded defaults before task files existed (pose-easy-v2).
LEGACY_START = {'mu': [[0, 0, 0, 0, 0, 0]], 'sigma': [[1, 1, 1, 1, 1, 1]]}
LEGACY_GOAL = {
    'mu': [[[5, 5, 5, 0, 0, 1.5708]], [[5, 5, -5, 0, 0, -1.5708]],
           [[5, -5, 5, 0, 0, 3.14159]], [[5, -5, -5, 0, 0, 3.14159]]],
    'sigma': [[[0.1] * 6]] * 4,
}
_T, _B, _R, _L = [0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0]
LEGACY_ACTIONS = {'mu': [[_T, _R], [_B, _R], [_T, _L], [_B, _L]], 'sigma': [[[0.0] * 6, [0.0] * 6]] * 4}
# The evaluation's old hard-coded conditions.
LEGACY_FULL = [("top", "right"), ("bottom", "right"), ("top", "left"), ("bottom", "left")]
LEGACY_PARTIAL = [(("top", "right"), 1, [0, 2]), (("bottom", "right"), 1, [1, 3]),
                  (("top", "right"), 0, [0, 1]), (("top", "left"), 0, [2, 3])]


def write_task(text):
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    f.write(text)
    f.close()
    return f.name


class FourCornersTaskTest(unittest.TestCase):
    def test_task_file_reproduces_the_legacy_trainer_defaults(self):
        task = load_pose_task(FOUR_CORNERS)
        self.assertEqual(task['start_dist_params'], LEGACY_START)
        self.assertEqual(task['goal_dist_params'], LEGACY_GOAL)
        self.assertEqual(task['action_dist_params'], LEGACY_ACTIONS)
        self.assertEqual(task['obs_dim'], 6)
        self.assertEqual(task['mode_names'], ['top_right', 'bottom_right', 'top_left', 'bottom_left'])

    def test_conditions_match_the_legacy_evaluation(self):
        full, partial = task_conditions(LEGACY_ACTIONS)
        self.assertEqual([c.valid_modes for c in full], [[0], [1], [2], [3]])
        for cond, pair in zip(full, LEGACY_FULL):
            self.assertTrue(torch.equal(cond.obs(8, 'cpu'), build_obs_from_actions(list(pair), 8, 'cpu')))
            self.assertIsNone(cond.obs_mask(8, 'cpu'))
        self.assertEqual(len(partial), len(LEGACY_PARTIAL))
        for cond, (pair, masked, valid) in zip(partial, LEGACY_PARTIAL):
            self.assertEqual(cond.valid_modes, valid)
            self.assertTrue(torch.equal(cond.obs(8, 'cpu'), build_obs_from_actions(list(pair), 8, 'cpu')))
            expected_mask = torch.zeros(8, 2, dtype=torch.bool)
            expected_mask[:, masked] = True
            self.assertTrue(torch.equal(cond.obs_mask(8, 'cpu'), expected_mask))


class CornersTwoOrientationsTaskTest(unittest.TestCase):
    def test_corners_are_bimodal_in_rotation_only(self):
        from utils.eval_utils import _rotvec, goal_mode_poses_from_config
        from utils.tf_utils import _quat_to_rot_mat
        task = load_pose_task(os.path.join(REPO, 'configs', 'pose_tasks', 'corners_two_orientations.yaml'))
        base = load_pose_task(FOUR_CORNERS)
        full, partial = task_conditions(task['action_dist_params'])
        self.assertEqual([c.valid_modes for c in full], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual([c.valid_modes for c in partial],
                         [[0, 1, 4, 5], [2, 3, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
        poses = goal_mode_poses_from_config(task['goal_dist_params'])
        corners = goal_mode_poses_from_config(base['goal_dist_params'])
        for k in range(8):
            corner = corners[k // 2]
            self.assertTrue(torch.allclose(poses[k, :3], corner[:3]))  # same position as the corner
            rel = _quat_to_rot_mat(corner[None, 3:])[0].T @ _quat_to_rot_mat(poses[k, None, 3:])[0]
            expected = 0.25 if k % 2 == 0 else -0.25                    # ccw, then cw, about z
            self.assertTrue(torch.allclose(_rotvec(rel[None])[0], torch.tensor([0., 0., expected]), atol=1e-4))


class GeneralTaskTest(unittest.TestCase):
    def test_modes_sharing_tokens_form_one_multimodal_condition(self):
        path = write_task("""
name: shared
start: {mu: [0, 0, 0, 0, 0, 0], sigma: 1}
tokens: {a: [1, 0], b: [0, 1]}
modes:
  - {name: m0, tokens: [a], mu: [1, 0, 0, 0, 0, 0], sigma: 0.1}
  - {name: m1, tokens: [a], mu: [-1, 0, 0, 0, 0, 0], sigma: 0.1}
  - {name: m2, tokens: [b], mu: [0, 1, 0, 0, 0, 0], sigma: 0.2}
""")
        task = load_pose_task(path)
        self.assertEqual(task['obs_dim'], 2)
        self.assertEqual(task['start_dist_params']['sigma'], [[1] * 6])
        self.assertEqual(task['goal_dist_params']['sigma'][2], [[0.2] * 6])
        full, partial = task_conditions(task['action_dist_params'])
        self.assertEqual([c.valid_modes for c in full], [[0, 1], [2]])
        self.assertEqual(partial, [])  # a single token per mode has no one-token subsets

    def test_unknown_tokens_and_ragged_token_lists_are_rejected(self):
        unknown = write_task("""
name: bad
start: {mu: [0, 0, 0, 0, 0, 0], sigma: 1}
tokens: {a: [1]}
modes: [{name: m0, tokens: [z], mu: [0, 0, 0, 0, 0, 0], sigma: 0.1}]
""")
        ragged = write_task("""
name: bad
start: {mu: [0, 0, 0, 0, 0, 0], sigma: 1}
tokens: {a: [1], b: [2]}
modes:
  - {name: m0, tokens: [a], mu: [0, 0, 0, 0, 0, 0], sigma: 0.1}
  - {name: m1, tokens: [a, b], mu: [1, 0, 0, 0, 0, 0], sigma: 0.1}
""")
        for path in (unknown, ragged):
            with self.assertRaises(ValueError):
                load_pose_task(path)


if __name__ == '__main__':
    unittest.main()
