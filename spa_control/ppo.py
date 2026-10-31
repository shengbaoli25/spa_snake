"""Small, hardware-independent NumPy PPO for bounded residual control.

The actor returns dimensionless tanh actions. The controller must multiply them
by its configured residual limit and apply its independent motor safety limits.
Only fresh stochastic rollouts may be passed to update(). Baseline and
deterministic evaluation trajectories are never training data.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ActionSample:
    raw_action: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    policy_version: int


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    raw_action: np.ndarray
    log_prob: float
    value: float
    reward: float
    next_value: float
    done: bool
    policy_version: int


class PPOAgent:
    """Two separate 32 x 32 tanh MLPs, a Gaussian actor, GAE and clipped PPO.

    Normalization is fixed during sampling and optimization. Call
    set_normalization() before a fresh rollout, or scale observations by fixed
    engineering ranges in the controller. Checkpoints include normalization,
    optimizer moments and RNG state, so sampling/training can resume exactly.
    """

    FORMAT_VERSION = 1

    def __init__(
        self, obs_dim: int, action_dim: int = 4, seed: int = 0, *,
        hidden_dim: int = 32, learning_rate: float = 3e-4,
        gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_ratio: float = 0.2, epochs: int = 6, minibatch_size: int = 64,
        initial_std: float = 0.15, value_coef: float = 0.5,
        max_grad_norm: float = 0.5, target_kl: float = 0.02,
    ) -> None:
        if min(obs_dim, action_dim, hidden_dim, epochs, minibatch_size) < 1:
            raise ValueError("Network dimensions and batch/epoch counts must be positive")
        if not 0 < initial_std <= 1 or not 0 < learning_rate < 1:
            raise ValueError("Invalid initial_std or learning_rate")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not 0 < clip_ratio < 1 or min(value_coef, max_grad_norm, target_kl) <= 0:
            raise ValueError("Invalid PPO clipping or loss parameters")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)
        self.rng = np.random.default_rng(seed)
        self.policy_version = 0
        self.total_updates = 0
        self.total_samples = 0
        self.optimizer_steps = 0
        self.normalization_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.normalization_std = np.ones(self.obs_dim, dtype=np.float64)
        self.params: dict[str, np.ndarray] = {}
        for prefix, output_dim in (("actor", action_dim), ("critic", 1)):
            widths = (obs_dim, hidden_dim, hidden_dim, output_dim)
            for layer, (nin, nout) in enumerate(zip(widths[:-1], widths[1:])):
                bound = np.sqrt(6.0 / (nin + nout))
                self.params[f"{prefix}.w{layer}"] = self.rng.uniform(-bound, bound, (nin, nout))
                self.params[f"{prefix}.b{layer}"] = np.zeros(nout)
        # The initial deterministic policy is exactly the nominal controller.
        self.params["actor.w2"].fill(0.0)
        self.params["actor.b2"].fill(0.0)
        self.params["log_std"] = np.full(action_dim, np.log(initial_std))
        self._adam_m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self._adam_v = {key: np.zeros_like(value) for key, value in self.params.items()}

    def _observations(self, obs: np.ndarray) -> np.ndarray:
        result = np.asarray(obs, dtype=np.float64)
        if result.ndim == 1:
            result = result[None, :]
        if result.ndim != 2 or result.shape[1] != self.obs_dim or not np.all(np.isfinite(result)):
            raise ValueError(f"Expected finite observations of shape (N, {self.obs_dim})")
        result = (result - self.normalization_mean) / self.normalization_std
        if not np.all(np.isfinite(result)):
            raise ValueError("Observation normalization overflow")
        return result

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set fixed input statistics; invalidates all previously sampled data."""
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        if mean.shape != (self.obs_dim,) or std.shape != (self.obs_dim,):
            raise ValueError("Normalization must match obs_dim")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Normalization mean/std must be finite, with positive std")
        self.normalization_mean = mean.copy()
        self.normalization_std = std.copy()
        self.policy_version += 1
        self._adam_m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self._adam_v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.optimizer_steps = 0

    def _forward(self, x: np.ndarray, prefix: str) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        h1 = np.tanh(x @ self.params[f"{prefix}.w0"] + self.params[f"{prefix}.b0"])
        h2 = np.tanh(h1 @ self.params[f"{prefix}.w1"] + self.params[f"{prefix}.b1"])
        output = h2 @ self.params[f"{prefix}.w2"] + self.params[f"{prefix}.b2"]
        return output, (x, h1, h2)

    def _backward(self, output_grad: np.ndarray, cache: tuple[np.ndarray, ...], prefix: str) -> dict[str, np.ndarray]:
        x, h1, h2 = cache
        grad_h2 = (output_grad @ self.params[f"{prefix}.w2"].T) * (1.0 - h2 * h2)
        grad_h1 = (grad_h2 @ self.params[f"{prefix}.w1"].T) * (1.0 - h1 * h1)
        return {
            f"{prefix}.w2": h2.T @ output_grad,
            f"{prefix}.b2": output_grad.sum(axis=0),
            f"{prefix}.w1": h1.T @ grad_h2,
            f"{prefix}.b1": grad_h2.sum(axis=0),
            f"{prefix}.w0": x.T @ grad_h1,
            f"{prefix}.b0": grad_h1.sum(axis=0),
        }

    @staticmethod
    def squashed_log_prob(raw_action: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        """Exact tanh change of variables, stable even for saturated actions."""
        raw_action = np.asarray(raw_action, dtype=np.float64)
        normal = -0.5 * ((raw_action - mean) / np.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        log_jacobian = 2.0 * (np.log(2.0) - raw_action - np.logaddexp(0.0, -2.0 * raw_action))
        return (normal - log_jacobian).sum(axis=-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActionSample:
        x = self._observations(obs)
        if len(x) != 1:
            raise ValueError("act() accepts exactly one observation")
        mean, _ = self._forward(x, "actor")
        value, _ = self._forward(x, "critic")
        raw = mean[0].copy() if deterministic else self.rng.normal(mean[0], np.exp(self.params["log_std"]))
        log_prob = self.squashed_log_prob(raw, mean[0], self.params["log_std"])
        return ActionSample(raw.copy(), np.tanh(raw), float(log_prob), float(value[0, 0]), self.policy_version)

    def value(self, obs: np.ndarray) -> float:
        x = self._observations(obs)
        if len(x) != 1:
            raise ValueError("value() accepts exactly one observation")
        return float(self._forward(x, "critic")[0][0, 0])

    def _gae(self, transitions: Sequence[Transition]) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(transitions), dtype=np.float64)
        running = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            transition = transitions[index]
            nonterminal = 0.0 if transition.done else 1.0
            delta = transition.reward + self.gamma * nonterminal * transition.next_value - transition.value
            running = delta + self.gamma * self.gae_lambda * nonterminal * running
            advantages[index] = running
        return advantages, advantages + np.asarray([transition.value for transition in transitions])

    def _loss_and_grads(
        self, x: np.ndarray, raw: np.ndarray, old_log_probs: np.ndarray,
        advantages: np.ndarray, returns: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        mean, actor_cache = self._forward(x, "actor")
        value, critic_cache = self._forward(x, "critic")
        log_std = self.params["log_std"]
        log_probs = self.squashed_log_prob(raw, mean, log_std)
        log_ratio = log_probs - old_log_probs
        ratio = np.exp(np.clip(log_ratio, -60.0, 60.0))
        clipped_ratio = np.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        surrogate = np.minimum(ratio * advantages, clipped_ratio * advantages)
        active = ~(((advantages > 0) & (ratio > 1.0 + self.clip_ratio)) |
                   ((advantages < 0) & (ratio < 1.0 - self.clip_ratio)))
        active &= np.abs(log_ratio) < 60.0
        log_prob_grad = -(advantages * ratio * active) / len(x)
        standardized_square = ((raw - mean) / np.exp(log_std)) ** 2
        mean_grad = log_prob_grad[:, None] * (raw - mean) / np.exp(2.0 * log_std)
        grads = self._backward(mean_grad, actor_cache, "actor")
        grads["log_std"] = (log_prob_grad[:, None] * (standardized_square - 1.0)).sum(axis=0)
        value_error = value[:, 0] - returns
        grads.update(self._backward(self.value_coef * value_error[:, None] / len(x), critic_cache, "critic"))
        policy_loss = -float(np.mean(surrogate))
        value_loss = 0.5 * float(np.mean(value_error ** 2))
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "loss": policy_loss + self.value_coef * value_loss,
            "approx_kl": float(np.mean((ratio - 1.0) - log_ratio)),
            "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > self.clip_ratio)),
            "action_std": float(np.exp(log_std).mean()),
        }
        return metrics, grads

    def _optimizer_step(self, grads: dict[str, np.ndarray]) -> None:
        if any(not np.all(np.isfinite(grad)) for grad in grads.values()):
            raise FloatingPointError("Non-finite PPO gradient; update aborted")
        grad_norm = np.sqrt(sum(float(np.sum(grad ** 2)) for grad in grads.values()))
        scale = min(1.0, self.max_grad_norm / (grad_norm + 1e-12))
        self.optimizer_steps += 1
        for key, grad in grads.items():
            grad = grad * scale
            self._adam_m[key] *= 0.9
            self._adam_m[key] += 0.1 * grad
            self._adam_v[key] *= 0.999
            self._adam_v[key] += 0.001 * grad * grad
            first = self._adam_m[key] / (1.0 - 0.9 ** self.optimizer_steps)
            second = self._adam_v[key] / (1.0 - 0.999 ** self.optimizer_steps)
            self.params[key] -= self.learning_rate * first / (np.sqrt(second) + 1e-8)
        np.clip(self.params["log_std"], -5.0, 0.0, out=self.params["log_std"])

    def update(self, transitions: Sequence[Transition]) -> dict[str, float]:
        """Consume one batch from the current policy; reusing it is an error.

        Each trial must end with done=True to keep GAE from joining trials.
        For a nonterminal batch cutoff, provide a valid next_value on the last
        transition. Do not call while the hardware is moving or collecting.
        """
        if len(transitions) < 2:
            raise ValueError("PPO needs at least two fresh stochastic transitions")
        if any(transition.policy_version != self.policy_version for transition in transitions):
            raise ValueError("Stale/off-policy batch: policy_version does not match current policy")
        observations = np.stack([transition.obs for transition in transitions])
        x = self._observations(observations)
        raw = np.asarray([transition.raw_action for transition in transitions], dtype=np.float64)
        if raw.shape != (len(transitions), self.action_dim) or not np.all(np.isfinite(raw)):
            raise ValueError("Invalid raw actions")
        scalars = np.asarray([[t.log_prob, t.value, t.reward, t.next_value] for t in transitions], dtype=np.float64)
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Transition contains non-finite scalar data")
        mean, _ = self._forward(x, "actor")
        expected_log_probs = self.squashed_log_prob(raw, mean, self.params["log_std"])
        expected_values = self._forward(x, "critic")[0][:, 0]
        if not np.allclose(expected_log_probs, scalars[:, 0], rtol=1e-6, atol=1e-7):
            raise ValueError("Rollout log probabilities differ from the current policy")
        if not np.allclose(expected_values, scalars[:, 1], rtol=1e-6, atol=1e-7):
            raise ValueError("Rollout values differ from the current policy")
        # A deterministic evaluation batch has no stochastic behavior density.
        if np.array_equal(raw, mean):
            raise ValueError("Deterministic evaluation trajectories cannot train PPO")
        advantages, returns = self._gae(transitions)
        advantage_std = float(advantages.std())
        if advantage_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)
        # Invalidate the consumed version even if a later numeric error occurs.
        self.policy_version += 1
        epochs_completed = 0
        for _ in range(self.epochs):
            indices = self.rng.permutation(len(transitions))
            for start in range(0, len(indices), self.minibatch_size):
                take = indices[start:start + self.minibatch_size]
                _, grads = self._loss_and_grads(x[take], raw[take], scalars[take, 0], advantages[take], returns[take])
                self._optimizer_step(grads)
            epochs_completed += 1
            metrics, _ = self._loss_and_grads(x, raw, scalars[:, 0], advantages, returns)
            if metrics["approx_kl"] > 1.5 * self.target_kl:
                break
        self.total_updates += 1
        self.total_samples += len(transitions)
        metrics.update({
            "samples": float(len(transitions)), "epochs": float(epochs_completed),
            "mean_reward": float(scalars[:, 2].mean()), "policy_version": float(self.policy_version),
            "total_updates": float(self.total_updates),
        })
        return metrics

    def save(self, path: str | Path) -> Path:
        """Atomic .npz checkpoint; accepts any filename without adding a suffix."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        config = {key: getattr(self, key) for key in (
            "obs_dim", "action_dim", "hidden_dim", "learning_rate", "gamma", "gae_lambda",
            "clip_ratio", "epochs", "minibatch_size", "value_coef", "max_grad_norm", "target_kl",
        )}
        metadata = {
            "format_version": self.FORMAT_VERSION, "config": config,
            "policy_version": self.policy_version, "total_updates": self.total_updates,
            "total_samples": self.total_samples, "optimizer_steps": self.optimizer_steps,
            "rng_state": self.rng.bit_generator.state,
        }
        arrays = {"metadata": np.asarray(json.dumps(metadata)),
                  "normalization_mean": self.normalization_mean, "normalization_std": self.normalization_std}
        for prefix, values in (("param", self.params), ("adam_m", self._adam_m), ("adam_v", self._adam_v)):
            arrays.update({f"{prefix}:{key}": value for key, value in values.items()})
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"]))
            if metadata.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("Unsupported PPO checkpoint format")
            agent = cls(**metadata["config"])
            for prefix, values in (("param", agent.params), ("adam_m", agent._adam_m), ("adam_v", agent._adam_v)):
                for key, current in values.items():
                    saved = np.asarray(checkpoint[f"{prefix}:{key}"], dtype=np.float64)
                    if saved.shape != current.shape or not np.all(np.isfinite(saved)):
                        raise ValueError(f"Invalid checkpoint tensor: {prefix}:{key}")
                    values[key] = saved.copy()
            agent.set_normalization(checkpoint["normalization_mean"], checkpoint["normalization_std"])
            # set_normalization resets moments; restore the saved optimizer.
            for prefix, values in (("adam_m", agent._adam_m), ("adam_v", agent._adam_v)):
                for key in values:
                    values[key] = np.asarray(checkpoint[f"{prefix}:{key}"], dtype=np.float64).copy()
            for key in ("policy_version", "total_updates", "total_samples", "optimizer_steps"):
                setattr(agent, key, int(metadata[key]))
            agent.rng.bit_generator.state = metadata["rng_state"]
        return agent
