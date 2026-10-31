"""Math/serialization tests only: these do not simulate robot performance."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spa_control.ppo import PPOAgent, Transition


def rollout(agent, count=24):
    rng = np.random.default_rng(123)
    transitions = []
    for index in range(count):
        obs = rng.normal(size=agent.obs_dim)
        sample = agent.act(obs)
        transitions.append(Transition(
            obs, sample.raw_action, sample.log_prob, sample.value,
            -float(np.sum((sample.action - 0.2) ** 2)), 0.0, True,
            sample.policy_version,
        ))
    return transitions


class PPOTests(unittest.TestCase):
    def test_zero_initial_deterministic_residual_and_bounds(self):
        agent = PPOAgent(17)
        for observation in (np.zeros(17), np.ones(17), np.arange(17) * 100):
            np.testing.assert_array_equal(agent.act(observation, True).action, np.zeros(4))
        for _ in range(100):
            sample = agent.act(np.zeros(17))
            self.assertTrue(np.all(np.abs(sample.action * 5.0) <= 5.0))
            self.assertTrue(np.isfinite(sample.log_prob))

    def test_squashed_log_probability_matches_change_of_variables(self):
        raw = np.asarray([[0.1, -0.5, 1.3, -2.0]])
        mean = np.asarray([[0.2, 0.1, -0.3, -0.4]])
        log_std = np.log(np.asarray([0.3, 0.5, 0.7, 0.9]))
        normal = -0.5 * ((raw - mean) / np.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        expected = np.sum(normal - np.log(1 - np.tanh(raw) ** 2), axis=-1)
        np.testing.assert_allclose(PPOAgent.squashed_log_prob(raw, mean, log_std), expected, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(PPOAgent.squashed_log_prob(np.array([1000.0, -1000.0]), 0, 0))))

    def test_gae_bootstrap_and_terminal_boundaries(self):
        agent = PPOAgent(2, gamma=0.9, gae_lambda=0.8)
        data = [Transition(np.zeros(2), np.zeros(4), 0, value, reward, next_value, done, 0)
                for value, reward, next_value, done in (
                    (1.0, 2.0, 3.0, False), (3.0, 4.0, 999.0, True),
                    (5.0, 6.0, 7.0, False),
                )]
        advantages, returns = agent._gae(data)
        np.testing.assert_allclose(advantages, [3.7 + 0.9 * 0.8, 1.0, 7.3])
        np.testing.assert_allclose(returns, advantages + [1, 3, 5])

    def test_manual_gradients_match_finite_differences(self):
        agent = PPOAgent(3, action_dim=2, hidden_dim=3)
        rng = np.random.default_rng(10)
        # Nonzero output weights exercise every layer in backpropagation.
        agent.params["actor.w2"][:] = rng.normal(0, 0.1, (3, 2))
        x = rng.normal(size=(5, 3))
        mean = agent._forward(x, "actor")[0]
        raw = rng.normal(mean, 0.15)
        old_log = agent.squashed_log_prob(raw, mean, agent.params["log_std"])
        advantages = np.array([-1.5, 0.3, 2.0, -0.5, 0.7])
        returns = rng.normal(size=5)
        # Shift ratios across both clipping branches, away from their kinks.
        old_log -= np.log(np.asarray([0.6, 1.5, 1.1, 1.1, 0.9]))
        _, gradients = agent._loss_and_grads(x, raw, old_log, advantages, returns)
        epsilon = 1e-6
        for key, parameter in agent.params.items():
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = agent._loss_and_grads(x, raw, old_log, advantages, returns)[0]["loss"]
                parameter[index] = original - epsilon
                minus = agent._loss_and_grads(x, raw, old_log, advantages, returns)[0]["loss"]
                parameter[index] = original
                self.assertAlmostEqual((plus - minus) / (2 * epsilon), gradients[key][index], delta=2e-7,
                                       msg=f"gradient {key}{index}")

    def test_update_consumes_policy_version_and_rejects_replay(self):
        agent = PPOAgent(3, epochs=2)
        data = rollout(agent)
        old_actor = agent.params["actor.w2"].copy()
        metrics = agent.update(data)
        self.assertEqual(agent.policy_version, 1)
        self.assertEqual(agent.total_samples, 24)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(np.array_equal(old_actor, agent.params["actor.w2"]))
        with self.assertRaisesRegex(ValueError, "Stale/off-policy"):
            agent.update(data)

    def test_tampered_rollout_rejected_before_mutation(self):
        agent = PPOAgent(3)
        data = rollout(agent)
        data[0] = replace(data[0], log_prob=data[0].log_prob + 0.1)
        with self.assertRaisesRegex(ValueError, "log probabilities"):
            agent.update(data)
        self.assertEqual(agent.policy_version, 0)
        data = rollout(agent)
        data[0] = replace(data[0], reward=float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            agent.update(data)

    def test_deterministic_evaluation_is_rejected(self):
        agent = PPOAgent(3)
        obs = np.zeros(3)
        sample = agent.act(obs, deterministic=True)
        transition = Transition(obs, sample.raw_action, sample.log_prob, sample.value, 0, 0, True, 0)
        with self.assertRaisesRegex(ValueError, "Deterministic"):
            agent.update([transition, transition])

    def test_normalization_is_fixed_and_invalidates_old_rollouts(self):
        agent = PPOAgent(3)
        data = rollout(agent)
        agent.set_normalization(np.array([1, 2, 3]), np.array([2, 3, 4]))
        np.testing.assert_allclose(agent._observations(np.array([3, 5, 7])), [[1, 1, 1]])
        with self.assertRaisesRegex(ValueError, "Stale/off-policy"):
            agent.update(data)
        before = agent.normalization_mean.copy()
        agent.update(rollout(agent))
        np.testing.assert_array_equal(agent.normalization_mean, before)

    def test_checkpoint_resumes_rng_optimizer_normalization_and_updates(self):
        agent = PPOAgent(3, epochs=2)
        agent.set_normalization(np.array([0.5, 0.2, 0.3]), np.array([1, 2, 3]))
        agent.update(rollout(agent))
        with tempfile.TemporaryDirectory() as directory:
            path = agent.save(Path(directory) / "policy.npz")
            restored = PPOAgent.load(path)
        self.assertEqual(agent.policy_version, restored.policy_version)
        self.assertEqual(agent.total_samples, restored.total_samples)
        np.testing.assert_array_equal(agent.normalization_mean, restored.normalization_mean)
        for _ in range(5):
            expected, actual = agent.act(np.ones(3)), restored.act(np.ones(3))
            np.testing.assert_array_equal(expected.action, actual.action)
            self.assertEqual(expected.log_prob, actual.log_prob)
        data = rollout(agent)
        restored_data = rollout(restored)
        for expected, actual in zip(data, restored_data):
            np.testing.assert_array_equal(expected.raw_action, actual.raw_action)
        expected_metrics, actual_metrics = agent.update(data), restored.update(restored_data)
        self.assertEqual(expected_metrics, actual_metrics)
        for key in agent.params:
            np.testing.assert_array_equal(agent.params[key], restored.params[key])


if __name__ == "__main__":
    unittest.main()
