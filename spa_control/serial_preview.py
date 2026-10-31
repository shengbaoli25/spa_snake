"""Offline components of the 2026-10-05 SPA manuscript.

This module does not connect to a robot. It contains no trained vision encoder,
history encoder or predictor: applications must supply their frozen features and
model functions, along with calibrated, measured rewards. A random policy is not
a validated controller. Host receipt times and camera clocks are not assumed to
be synchronized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import math
import numpy as np

ACTIONS = (-5, 0, 5)


@dataclass(frozen=True)
class Packet:
    """A complete front-drive sample, timestamped on receipt by the host."""
    received_s: float
    values: object


def causal_windows(packets: Sequence[Packet], cutoff: float, *, recent_s: float = .5,
                   width_s: float = .5, count: int = 8) -> list[list[Packet]]:
    """H_k = (cutoff-recent-k*width, cutoff-recent-(k-1)*width]."""
    if not math.isfinite(cutoff) or recent_s <= 0 or width_s <= 0 or count < 1:
        raise ValueError("invalid timing configuration")
    if any(not math.isfinite(p.received_s) for p in packets):
        raise ValueError("packet receipt times must be finite")
    return [[p for p in packets if cutoff - recent_s - k * width_s < p.received_s
             <= cutoff - recent_s - (k - 1) * width_s]
            for k in range(1, count + 1)]


def anchored_preview(null_z: np.ndarray, window_z: Sequence[np.ndarray], valid: Sequence[bool],
                     scores: Sequence[float], null_score: float, recent: np.ndarray,
                     candidates: Sequence[np.ndarray], reference: Callable,
                     correction: Callable) -> tuple[np.ndarray, float, list[np.ndarray], np.ndarray]:
    """Compose model-provided deterministic M and F; null is always available.

    reference(candidate) -> response vector; correction(z, recent, candidate) ->
    same shape. scores contain one value per window, shared across actions.
    Caller provides frozen trained encoders/selector. No quantile interpretation
    is invented here; supplied response representation is retained unchanged.
    """
    z0 = np.asarray(null_z, dtype=float)
    zs = [np.asarray(z, dtype=float) for z in window_z]
    valid = np.asarray(valid, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if len(zs) != len(valid) or scores.shape != (len(zs),) or any(z.shape != z0.shape for z in zs):
        raise ValueError("window shapes do not match")
    if not np.isfinite(null_score) or not np.isfinite(scores[valid]).all() or not np.isfinite(z0).all():
        raise ValueError("invalid selector input")
    if any(not np.isfinite(z).all() for z, good in zip(zs, valid) if good):
        raise ValueError("invalid window representation")
    logits = np.r_[null_score, scores[valid]]
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    alpha = np.zeros(len(zs), dtype=float)
    alpha[valid] = weights[1:]
    zbar = np.zeros_like(z0)
    for z, weight, good in zip(zs, alpha, valid):
        if good:
            zbar += weight * (z - z0)
    predictions = []
    for action in candidates:
        base = np.asarray(reference(action), dtype=float)
        if not np.isfinite(base).all():
            raise ValueError("nonfinite base prediction")
        result = base.copy()
        if valid.any():
            baseline = np.asarray(correction(z0, recent, action), dtype=float)
            if baseline.shape != base.shape or not np.isfinite(baseline).all():
                raise ValueError("invalid anchor")
            for z, weight, good in zip(zs, alpha, valid):
                if good:
                    term = np.asarray(correction(z, recent, action), dtype=float)
                    if term.shape != base.shape or not np.isfinite(term).all():
                        raise ValueError("invalid response correction")
                    result += weight * (term - baseline)
        predictions.append(result)
    return zbar, float(weights[0]), predictions, alpha


def response_features(raw: Sequence[float]) -> np.ndarray:
    """Convert composed location/width logits into ordered quantile features.

    Input is three (location, lower-width logit, upper-width logit) triples,
    followed by intervention logit. Call *after* anchored raw composition.
    """
    x = np.asarray(raw, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("ten finite response logits required")
    out = []
    for i in range(3):
        mu, lower, upper = x[3*i:3*i+3]
        values = np.array([mu - np.logaddexp(0., lower), mu, mu + np.logaddexp(0., upper)])
        if i == 1:
            values = np.logaddexp(0., values)
        elif i == 2:
            values = np.exp(-np.logaddexp(0., -values))
        out.extend(values.tolist())
    out.append(float(np.exp(-np.logaddexp(0., -x[9]))))
    return np.asarray(out)


def measured_load_costs(currents: np.ndarray, fresh: np.ndarray, limits: Sequence[float],
                        warning_ratio: float) -> tuple[float, float, float]:
    """Calculate costs on a caller-built causal as-of grid of active channels.

    Each row is accepted only when *all* active channels are fresh and finite;
    the caller must form rows from original packets and retain packet counts.
    Returns mean squared excess, exceedance occupancy and grid coverage.
    """
    values = np.asarray(currents, dtype=float)
    fresh = np.asarray(fresh, dtype=bool)
    limits = np.asarray(limits, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or fresh.shape != values.shape or limits.shape != (values.shape[1],):
        raise ValueError("invalid feedback grid")
    if not np.isfinite(limits).all() or np.any(limits <= 0) or not np.isfinite(warning_ratio) or warning_ratio < 0:
        raise ValueError("invalid current limits")
    accepted = np.all(fresh & np.isfinite(values), axis=1)
    if not accepted.any():
        raise ValueError("no fresh current feedback in interval")
    ratio = np.abs(values[accepted]) / limits
    current_cost = float(np.clip(np.mean(np.maximum(ratio - warning_ratio, 0.) ** 2), 0., 1.))
    occupancy = float(np.mean(np.any(ratio > warning_ratio, axis=1)))
    return current_cost, occupancy, float(accepted.mean())


def project_candidates(nominal: Sequence[float], signs: Sequence[int], rear: Sequence[bool],
                       active: Sequence[bool], q_min: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Project and deduplicate motor vectors in fixed action order {-5,0,+5}."""
    q = np.asarray(nominal, dtype=float)
    s = np.asarray(signs)
    rear = np.asarray(rear, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if q.shape != (4,) or s.shape != (4,) or rear.shape != (4,) or active.shape != (4,):
        raise ValueError("four structural channels required")
    if not np.isfinite(q).all() or not np.isfinite(q_min) or not 0 <= q_min <= 500 or not np.isin(s, [-1, 1]).all():
        raise ValueError("invalid nominal or directions")
    if np.any(q[active] < q_min) or np.any(q[active] > 500) or np.any(q[active] != np.round(q[active])):
        raise ValueError("active nominal commands must be integers within the configured limits")
    # Use symmetric half-away-from-zero rounding; projected magnitudes are nonnegative.
    projected = []
    for action in ACTIONS:
        magnitudes = np.floor(np.clip(q + rear * action, q_min, 500) + .5).astype(int)
        projected.append(np.where(active, s * magnitudes, 0))
    vectors = np.stack(projected)
    support = np.array([not any(np.array_equal(vectors[i], vectors[j]) for j in range(i))
                        for i in range(len(ACTIONS))], dtype=bool)
    return vectors, support


@dataclass(frozen=True)
class RewardWeights:
    current: float
    occupancy: float
    action: float
    smoothness: float
    stop: float
    sigma: float
    min_speed: float


def measured_reward(velocity: float | None, target: float, current_cost: float | None,
                    occupancy: float | None, residuals: Sequence[float],
                    previous_residuals: Sequence[float], stopped: bool,
                    weights: RewardWeights) -> float:
    """Reward requires measured interval velocity and valid current costs.

    Costs must be derived from fresh actual feedback, not model predictions.
    Missing values are collection faults, never silently zero-filled.
    """
    vals = (velocity, target, current_cost, occupancy, weights.sigma, weights.min_speed,
            weights.current, weights.occupancy, weights.action, weights.smoothness, weights.stop)
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        raise ValueError("missing or invalid measured reward")
    if weights.sigma <= 0 or weights.min_speed <= 0 or min(weights.current, weights.occupancy,
            weights.action, weights.smoothness, weights.stop) < 0:
        raise ValueError("invalid reward parameters")
    if not 0 <= current_cost <= 1 or not 0 <= occupancy <= 1:
        raise ValueError("current costs must be in [0,1]")
    a = np.asarray(residuals, dtype=float)
    previous = np.asarray(previous_residuals, dtype=float)
    if a.ndim != 1 or not a.size or a.shape != previous.shape or not np.isfinite(a).all() or not np.isfinite(previous).all() or np.any(abs(a) > 5) or np.any(abs(previous) > 5):
        raise ValueError("invalid projected residuals")
    e = (velocity - target) / max(abs(target), weights.min_speed)
    return float(math.exp(-e * e / (2 * weights.sigma ** 2)) - weights.current * current_cost
                 - weights.occupancy * occupancy - weights.action * np.mean((a / 5) ** 2)
                 - weights.smoothness * np.mean(((a - previous) / 10) ** 2)
                 - weights.stop * bool(stopped))


def failure_tail(remaining_steps: int, gamma: float, weights: RewardWeights) -> float:
    """Discounted absorbing failure slots following the intervention transition."""
    if remaining_steps < 0 or not 0 <= gamma <= 1:
        raise ValueError("invalid tail parameters")
    b = weights.current + weights.occupancy + weights.action + weights.smoothness
    return float(-b * sum(gamma ** i for i in range(1, remaining_steps + 1)))


@dataclass(frozen=True)
class Sample:
    action_index: int
    log_prob: float
    value: float
    version: int


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    support: np.ndarray
    action_index: int | None
    old_log_prob: float | None
    value: float
    reward: float
    next_value: float
    done: bool
    version: int
    actor_weight: float = 1.
    # At a nonterminal chunk boundary bootstrap next_value but do not cross into
    # an unrelated trajectory during reverse GAE.
    trace_cut: bool = False
    # Only new stochastic real rollouts or an explicitly labeled simulation are accepted.
    source: str = "real_on_policy"


class CategoricalPPO:
    """Separate 64x64 actor and critic, actual masked PPO with NumPy Adam.

    A batch belongs to one behavior version and is invalidated after update.
    Caller logs projected/sent/interrupted commands and must only update when
    stopped. Explicit simulation has no physical validation value.
    """
    def __init__(self, obs_dim: int, seed: int = 0, *, hidden: int = 64,
                 learning_rate: float = 3e-4, gamma: float = .99, gae_lambda: float = .95,
                 clip: float = .2, entropy: float = .01, value_coef: float = .5,
                 epochs: int = 4, batch_size: int = 64, max_grad: float = .5):
        if obs_dim < 1 or hidden < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("invalid dimensions")
        if not (0 < learning_rate < 1 and 0 <= gamma <= 1 and 0 <= gae_lambda <= 1
                and 0 < clip < 1 and entropy >= 0 and value_coef >= 0 and max_grad > 0):
            raise ValueError("invalid PPO parameters")
        self.obs_dim, self.lr, self.gamma, self.lam, self.clip = obs_dim, learning_rate, gamma, gae_lambda, clip
        self.entropy, self.value_coef, self.epochs, self.batch_size, self.max_grad = entropy, value_coef, epochs, batch_size, max_grad
        self.rng = np.random.default_rng(seed)
        self.version = 0
        self.params = {}
        for name, output in (("a", 3), ("v", 1)):
            for i, (ni, no) in enumerate(zip((obs_dim, hidden, hidden), (hidden, hidden, output))):
                self.params[f"{name}w{i}"] = self.rng.normal(0, (2 / (ni + no)) ** .5, (ni, no))
                self.params[f"{name}b{i}"] = np.zeros(no)
        self.params["aw2"][:] = 0
        self.params["ab2"][:] = np.log([.1, .8, .1])
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.steps = 0

    def _forward(self, x, prefix):
        h1 = np.tanh(x @ self.params[prefix + "w0"] + self.params[prefix + "b0"])
        h2 = np.tanh(h1 @ self.params[prefix + "w1"] + self.params[prefix + "b1"])
        return h2 @ self.params[prefix + "w2"] + self.params[prefix + "b2"], (x, h1, h2)

    def _backward(self, grad, cache, prefix):
        x, h1, h2 = cache
        g2 = (grad @ self.params[prefix + "w2"].T) * (1 - h2 * h2)
        g1 = (g2 @ self.params[prefix + "w1"].T) * (1 - h1 * h1)
        return {prefix + "w2": h2.T @ grad, prefix + "b2": grad.sum(0),
                prefix + "w1": h1.T @ g2, prefix + "b1": g2.sum(0),
                prefix + "w0": x.T @ g1, prefix + "b0": g1.sum(0)}

    def _probabilities(self, obs, mask):
        obs = np.asarray(obs, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or mask.shape != (len(obs), 3):
            raise ValueError("invalid observation/support shape")
        if not np.isfinite(obs).all() or not np.all(mask.any(axis=1)):
            raise ValueError("invalid observation or empty action support")
        logits, cache = self._forward(obs, "a")
        logits = np.where(mask, logits, -np.inf)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True), cache

    def act(self, observation, support, deterministic=False):
        x = np.asarray(observation, dtype=float).reshape(1, -1)
        p, _ = self._probabilities(x, np.asarray(support, dtype=bool).reshape(1, -1))
        i = int(np.argmax(p[0])) if deterministic else int(self.rng.choice(3, p=p[0]))
        value = float(self._forward(x, "v")[0][0, 0])
        return Sample(i, float(np.log(p[0, i])), value, self.version)

    def update(self, transitions: Sequence[Transition], *, allow_simulation: bool = False):
        """One fresh behavior batch. A failed or missing reward cannot be zero-filled."""
        ts = list(transitions)
        if len(ts) < 2:
            raise ValueError("at least two transitions required")
        if any(t.version != self.version for t in ts):
            raise ValueError("stale or already-consumed policy batch")
        allowed = {"real_on_policy"} | ({"simulation"} if allow_simulation else set())
        if any(t.source not in allowed for t in ts):
            raise ValueError("historical/evaluation data are not on-policy training")
        if len({t.source for t in ts}) != 1:
            raise ValueError("real and simulated trajectories cannot share a PPO batch")
        x = np.stack([np.asarray(t.observation, float) for t in ts])
        mask = np.stack([np.asarray(t.support, bool) for t in ts])
        p, _ = self._probabilities(x, mask)
        sampled = np.asarray([t.action_index is not None for t in ts], dtype=bool)
        indices = np.asarray([t.action_index if t.action_index is not None else int(np.flatnonzero(mask[i])[0])
                              for i, t in enumerate(ts)], dtype=int)
        if np.any(indices < 0) or np.any(indices >= 3) or not np.all(mask[np.arange(len(ts)), indices]):
            raise ValueError("stored action is unsupported")
        if any((t.action_index is None and (t.old_log_prob is not None or t.actor_weight != 0)) or
               (t.action_index is not None and (t.old_log_prob is None or t.actor_weight != 1)) for t in ts):
            raise ValueError("forced steps have no sampled action and zero actor weight")
        data = np.asarray([[t.old_log_prob if t.old_log_prob is not None else 0., t.value, t.reward,
                            t.next_value, t.actor_weight] for t in ts], float)
        if not np.isfinite(data).all() or np.any((data[:, 4] < 0) | (data[:, 4] > 1)):
            raise ValueError("invalid rollout data")
        actual_values = self._forward(x, "v")[0][:, 0]
        if not np.allclose(np.log(p[np.arange(len(ts)), indices])[sampled], data[sampled, 0], rtol=1e-6, atol=1e-7):
            raise ValueError("stored log probability or support differs from behavior policy")
        if not np.allclose(actual_values, data[:, 1], rtol=1e-6, atol=1e-7):
            raise ValueError("stored values differ from behavior policy")
        if any(t.done and t.next_value != 0 for t in ts):
            raise ValueError("true terminals must have zero bootstrap")
        advantages = np.zeros(len(ts))
        run = 0.
        for i in range(len(ts) - 1, -1, -1):
            nonterm = 0. if ts[i].done else 1.
            delta = ts[i].reward + self.gamma * nonterm * ts[i].next_value - ts[i].value
            run = delta + self.gamma * self.lam * nonterm * (not ts[i].trace_cut) * run
            advantages[i] = run
        returns = advantages + data[:, 1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        last_loss = 0.
        for _ in range(self.epochs):
            order = self.rng.permutation(len(ts))
            for start in range(0, len(ts), self.batch_size):
                ix = order[start:start + self.batch_size]
                probs, acache = self._probabilities(x[ix], mask[ix])
                values, vcache = self._forward(x[ix], "v")
                n = len(ix)
                selected = probs[np.arange(n), indices[ix]]
                ratio = np.exp(np.clip(np.log(selected) - data[ix, 0], -30, 30))
                adv = advantages[ix]
                unclipped = ratio * adv
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
                weight = data[ix, 4]
                active = ~(((adv > 0) & (ratio > 1 + self.clip)) | ((adv < 0) & (ratio < 1 - self.clip)))
                onehot = np.eye(3)[indices[ix]]
                grad = (-weight * adv * ratio * active / n)[:, None] * (onehot - probs)
                safe_logp = np.log(np.maximum(probs, 1e-300))
                ent = -(probs * safe_logp).sum(axis=1)
                grad += self.entropy * weight[:, None] * probs * (safe_logp + ent[:, None]) / n
                grads = self._backward(grad, acache, "a")
                err = values[:, 0] - returns[ix]
                grads.update(self._backward((self.value_coef * err / n)[:, None], vcache, "v"))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
                if not math.isfinite(norm):
                    raise FloatingPointError("non-finite PPO gradient")
                scale = min(1., self.max_grad / (norm + 1e-12))
                self.steps += 1
                for key, g in grads.items():
                    g *= scale
                    self.m[key] = .9 * self.m[key] + .1 * g
                    self.v[key] = .999 * self.v[key] + .001 * g * g
                    mh = self.m[key] / (1 - .9 ** self.steps)
                    vh = self.v[key] / (1 - .999 ** self.steps)
                    self.params[key] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
                last_loss = float(-np.mean(weight * np.minimum(unclipped, clipped)) + .5 * self.value_coef * np.mean(err * err) - self.entropy * np.mean(weight * ent))
        self.version += 1
        return {"loss": last_loss, "transitions": len(ts), "policy_version": self.version}


class DemoExperiment:
    """Synthetic-only rollout and PPO demonstration, with no motor I/O.

    Its toy plant, candidate predictions and rewards are not physical data;
    training here does not produce a policy for use on a real robot.
    """
    def __init__(self, seed: int = 0):
        self.reset(seed)

    def reset(self, seed: int = 0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.policy = CategoricalPPO(16, seed=self.seed)
        self.train_batches = 0
        self.train_samples = 0
        self.last_loss = None
        self._reset_episode()
        return self.status()

    def _reset_episode(self):
        self.steps = 0
        self.position = 0.
        self.velocity = .04
        self.previous = 0
        self.packets = []
        self.history = []
        self.last = None

    def _observation(self):
        """Build an as-of synthetic observation without consuming the policy RNG."""
        t = self.steps * .25
        packets = [*self.packets, Packet(t, self.velocity)]
        windows = causal_windows(packets, t)
        valid = [bool(w) for w in windows]
        zs = [np.array([float(np.mean([p.values for p in w])), len(w) / 2])
              if w else np.zeros(2) for w in windows]
        candidates, support = project_candidates([300]*4, [-1, 1, -1, 1],
                                                 [False, False, True, True], [True]*4)
        zbar, null, predictions, _ = anchored_preview(
            np.zeros(2), zs, valid, [0.]*8, 0., np.zeros(1), candidates,
            lambda c: np.array([float(c[3]) / 500., 0.]),
            lambda z, recent, c: np.array([z[0] * .1, z[1] * .1]))
        remaining = max(0., 48 - self.steps) / 48.
        obs = np.r_[[self.velocity, self.position, self.previous / 5., remaining], zbar, null,
                    np.asarray(predictions).reshape(-1), support.astype(float)]
        return obs, support

    def _step(self, *, terminal=False):
        obs, support = self._observation()
        sample = self.policy.act(obs, support)
        action = ACTIONS[sample.action_index]
        self.packets.append(Packet(self.steps * .25, self.velocity))
        self.velocity = max(0., .04 + .0003 * action + float(self.rng.normal(0, .001)))
        self.position += self.velocity * .25
        reward = measured_reward(self.velocity, .04, .1, 0., [action, action],
                                 [self.previous, self.previous], False,
                                 RewardWeights(.1, .1, .01, .01, 1., .3, .01))
        self.previous = action
        self.steps += 1
        self.last = {"observation": obs.tolist(), "reward": reward, "action": action}
        self.history.append({"step": self.steps, "action": action, "reward": reward,
                             "velocity": self.velocity})
        self.history = self.history[-100:]
        if terminal:
            next_value = 0.
        else:
            next_obs, next_support = self._observation()
            next_value = self.policy.act(next_obs, next_support, deterministic=True).value
        return Transition(obs.copy(), support.copy(), sample.action_index, sample.log_prob,
                          sample.value, reward, next_value, terminal, sample.version,
                          source="simulation")

    def step(self):
        self._step()
        return self.status()

    def train_batch(self):
        """Train only on five freshly sampled 48-step synthetic episodes."""
        transitions = []
        for _ in range(5):
            self._reset_episode()
            for index in range(48):
                transitions.append(self._step(terminal=index == 47))
        result = self.policy.update(transitions, allow_simulation=True)
        self.train_batches += 1
        self.train_samples += len(transitions)
        self.last_loss = result["loss"]
        return self.status()

    def status(self):
        return {"step": self.steps, "seed": self.seed,
                "observation": self.last["observation"] if self.last else None,
                "reward": self.last["reward"] if self.last else None,
                "action": self.last["action"] if self.last else None,
                "history": list(self.history), "synthetic": True, "trained": False,
                "policy_version": self.policy.version, "train_batches": self.train_batches,
                "train_samples": self.train_samples, "last_loss": self.last_loss,
                "notice": "合成演示采用简化特征：未接电机，预测器和策略未经真实数据训练"}
