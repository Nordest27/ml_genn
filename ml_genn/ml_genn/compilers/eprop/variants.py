"""Choices that select the e-prop variant.

Structural choices (how feedback reaches hidden neurons, which policy head, which output rule) are
enums. The hidden-layer rule of the RL / TD(lambda) variant is a small frozen config,
:class:`HiddenRuleConfig`, from which :mod:`.hidden_rule` generates a compact GeNN model that contains
only the terms the configuration uses (no commented-out alternatives, no ``0.0 *`` switches).
Named presets for the variants in the paper are in :data:`PRESETS`.

Decomposition used throughout (see the paper, Sec. 4.2 and Appendix D). The score of a synapse is
``kappa * gate * pi * xi / sigma^2`` with ``pi = ZFilter - Beta * epsilonA`` (e-prop's presynaptic factor),
``xi`` the synapse's noise trace and ``kappa = 1 - Alpha^2``. With ``psibar`` a running average of the
pseudo-derivative ``psi``:

* drift part      ``P = (psi - psibar) * pi * xi / sigma^2``   (perturbation-induced drift, Hebbian-like)
* gradient part   ``G = -psibar * pi * xi / sigma^2``          (perturbation gradient, REINFORCE sign)

The optimiser descends on ``DeltaG``; a rule adds ``delta * (c_P * trace(P) + c_G * trace(G))``.
``c_P = 1, c_G = -1`` is the original rule (gate ``psi``), ``c_P = c_G = 1`` the proposed rule
(gate ``psi - 2 psibar``).
"""

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Optional, Union


class FeedbackType(Enum):
    """How error/learning-signal feedback reaches hidden neurons."""
    SYMMETRIC = "symmetric"   # feedback weights tied to forward readout weights
    RANDOM = "random"         # fixed random feedback weights
    ADAPTIVE = "adaptive"     # learned feedback weights


class OutputRule(Enum):
    """Which rule an output (readout) synapse uses."""
    SUPERVISED_SYMMETRIC = auto()
    SUPERVISED_RANDOM = auto()
    SUPERVISED_ADAPTIVE_FEEDBACK = auto()

    VALUE_SYMMETRIC = auto()
    VALUE_RANDOM = auto()
    VALUE_ADAPTIVE_FEEDBACK = auto()

    POLICY_SYMMETRIC = auto()
    POLICY_RANDOM = auto()
    POLICY_ADAPTIVE_FEEDBACK = auto()


class FeedbackRule(Enum):
    """Which rule a pure feedback connection (conn.is_feedback) uses."""
    RANDOM_ERROR = auto()          # random feedback weights, targets E
    RANDOM_PLAIN = auto()          # random feedback weights, no post var ref needed
    POLICY_ADAPTIVE = auto()       # learned feedback into a policy head
    VALUE_ADAPTIVE = auto()        # learned feedback into the value head


class PolicyType(Enum):
    """Policy head output distribution (same values as the monolithic compiler's PolicyTypes)."""
    CATEGORICAL = "categorical"
    GAUSSIAN_TRACE = "gaussian_trace"
    GENERIC = "generic"


class HiddenRule(Enum):
    """Which hidden-layer model family a synapse uses (selected by the compiler from the neuron type
    and whether the RL variant is active)."""
    LIF = auto()             # LIF postsynaptic neuron (supervised, with a simple perturbation term)
    ALIF = auto()            # supervised ALIF e-prop
    ALIF_TD_LAMBDA = auto()  # RL / TD(lambda) ALIF rule, generated from a HiddenRuleConfig


# ----------------------------------------------------------------------------------------------
# Hidden RL rule configuration
# ----------------------------------------------------------------------------------------------

class NoisePlacement(Enum):
    """Where the exploration noise enters, and which trace the score uses."""
    NONE = auto()                # no noise (e-prop and baselines)
    WEIGHT_SHARED = auto()       # "weight dist": one sample per postsynaptic neuron and step,
                                 # delivered by every transmitting synapse; per-synapse noise trace
    WEIGHT_INDEPENDENT = auto()  # "weight dist ind noise": independent per-synapse noise (hash of the
                                 # pre/post noise samples, synapse indices and time); per-synapse trace
    NODE = auto()                # "node dist": noise added to the membrane; neuron-level trace


class Estimator(Enum):
    """How the noise enters the score."""
    TRACE = auto()   # gate x presynaptic factor x membrane-filtered noise trace (all rules in the paper)
    EXACT = auto()   # unbiased likelihood-ratio estimator: fresh noise x presynaptic spike, no gate, no
                     # filter; zero drift by construction (used with gradient > 0 only)


class LocalRoute(Enum):
    """How the neuron-local objectives (|dV|, voltage regulariser, firing-rate reward) are credited."""
    GRADIENT_INSTANT = auto()  # instantaneous gradient part G with a per-neuron running-mean baseline
                               # (the setting that achieved the objectives in the analyses)
    GRADIENT_TRACE = auto()    # lambda-trace of the gradient part
    WHOLE_TRACE = auto()       # the rule's whole trace (the original implementation; legacy)


class EpropScope(Enum):
    """Which hidden populations use the e-prop term."""
    ALL = auto()                # every hidden synapse combines e-prop with the perturbation terms (eprop_plus_*)
    SIGNAL_RECIPIENTS = auto()  # populations that receive an e-prop learning signal (a forward connection to a policy
                                # or value head under symmetric feedback, or a feedback connection to one) learn with
                                # e-prop only and without noise; all other hidden populations with the perturbation
                                # rule only (no e-prop)


class DVMode(Enum):
    """Definition of the |dV| term."""
    RESET = auto()     # |PrevV - V| with PrevV stored before the reset: the reset jump (original code)
    CORRECT = auto()   # |V - previous post-reset voltage|: the membrane change excluding the reset


@dataclass(frozen=True)
class LocalObjectives:
    route: LocalRoute = LocalRoute.GRADIENT_INSTANT
    dv: float = 0.01                 # weight of -|dV|
    vreg: Optional[float] = None     # weight of -VReg (None: the compiler's c_reg)
    rate: Optional[float] = None     # weight of the firing-rate reward (None: the compiler's c_reg)
    dv_mode: DVMode = DVMode.CORRECT
    baseline_decay: float = 0.999    # per-neuron running mean subtracted (GRADIENT_* routes only)
    decorr: float = 0.0              # weight of the decorrelation reward -(z - zbar)(I_inh - Ibar_inh) / Ibar_inh:
                                     # penalises spiking in step with the inhibition the neuron receives (redundancy
                                     # with the cells that drive that inhibition). Neuron-local: its own spikes and
                                     # its own total inhibitory input (GRADIENT_* routes only)
    decorr_tau: float = 1000.0       # ... time constant (ms) of the running means zbar and Ibar_inh
    cancel: float = 0.0              # weight of the cancellation reward -|dV| / (|dI_exc| + cancel_eps * mean|dI_exc|):
                                     # the neuron's own membrane change relative to the change of its excitatory
                                     # drive. Small when inhibition / recurrence cancel input changes in time (a
                                     # predictive-coding-like balance); scaling the input weights down does not help
                                     # (dV and dI_exc shrink together), unlike the plain dV reward. Needs the separate
                                     # inhibitory channel (switched on automatically) and a GRADIENT_* route
    cancel_tau: float = 1000.0       # ... time constant (ms) of the running mean of |dI_exc|
    cancel_eps: float = 0.1          # ... floor of the denominator, as a fraction of that mean
    contrast: float = 0.0            # weight of the contrast reward, a TD-gated local reward:
                                     # r_i = (delta - mean delta) * trace[(zpost - mean zpost)^2] (zpost the
                                     # postsynaptic spike trace, its mean a running average with the rule's
                                     # hebb_decay; delta centred with the rule's center_drift). After a positive TD
                                     # error a neuron is rewarded for having been far from its usual activity, in
                                     # either direction; in expectation dW ~ delta (zpost - mean) psi pre (active
                                     # neurons up, near-threshold inactive ones down, nothing where psi = 0), plus
                                     # the gradient part weighted by the squared deviation. Credited like the other
                                     # local rewards (part of the same per-neuron reward and baseline), unless
                                     # contrast_route says otherwise
    contrast_route: Optional[LocalRoute] = None   # None: the contrast reward follows `route`; GRADIENT_INSTANT /
                                     # GRADIENT_TRACE: credited on its own (own running-mean baseline) through the
                                     # instantaneous score or the gradient score's lambda-trace, whatever `route` the
                                     # other local rewards use (GRADIENT_TRACE reproduces the runs of 2026-11-11)
    contrast_baseline: str = "mean"  # baseline of the contrast reward on its own route (contrast_route), whose reward
                                     # is delta * ContrD: "mean": its running mean (ContrR - ContrMean);
                                     # "gated_mean": delta * (ContrD - running mean of ContrD); "gated_synapse":
                                     # delta * (ContrD - (CCritA + CCritU * ZFilter)), every synapse a linear critic of
                                     # the neuron's ContrD on its own presynaptic trace, learned by the reward-free
                                     # rule u += contrast_critic_eta * (ContrD - prediction) * x / (1 + x^2). The gated
                                     # baselines cut the variance of the drift-like part delta * dContrD/dW (6-8x in
                                     # the open-loop replay of 2026-11-13) but weight the task-gradient part
                                     # d delta/dW by the residual (ContrD - prediction) instead of ContrD: the
                                     # direction is then left to G / local.td
    contrast_critic_eta: float = 1e-3   # learning rate of the "gated_synapse" critics
    sensitivity: float = 0.0         # weight of the sensitivity reward, the drift as a TD-gated local reward:
                                     # r_i = -(delta - mean delta) * trace[psi - psibar] (delta centred with the rule's
                                     # center_drift): after a better-than-expected outcome a neuron is rewarded for
                                     # having been decisive (less time near threshold than usual), after a worse one
                                     # for having been sensitive. In expectation the gradient of the neuron's recent
                                     # sensitivity, as the drift, credited like the other local rewards (or on its own
                                     # sensitivity_route); the centred gate keeps its gradient part from pointing
                                     # against G
    sensitivity_route: Optional[LocalRoute] = None   # as contrast_route, for the sensitivity reward
    td: float = 0.0                  # weight of the TD error itself as a term of the local reward (delta, uncentred:
                                     # the shared baseline removes its mean). Credited through the noise score like
                                     # the other terms, it is the gradient part G on the local route: with every
                                     # term shared (no own routes), the neuron's whole update is one reward,
                                     # td * delta + delta * (contrast - sensitivity terms) + decorr ..., one baseline
    td_route: Optional[LocalRoute] = None   # two channels: the TD-dependent terms (td, contrast, sensitivity) form
                                     # a second reward with its own baseline, credited through this route (e.g.
                                     # GRADIENT_TRACE: the TD error judges the past window), while the terms of the
                                     # moment (dv, vreg, rate, decorr, cancel) stay on `route` with theirs (e.g.
                                     # GRADIENT_INSTANT). contrast_route / sensitivity_route must then be None
    critic_eta: float = 0.0          # > 0 (with td_route): every synapse is an actor-critic agent on the TD channel's
                                     # reward r_i (LocR2). Its critic V_ij = CritA + CritU * x_j (x_j its presynaptic
                                     # trace, which the neuron's own noise does not affect, so the estimate stays
                                     # unbiased) gives a local TD error d_ij = r_i + critic_gamma V_ij(t) - V_ij(t-1),
                                     # which replaces (LocR2 - Loc2Mean) in the update; the critic learns by TD(lambda)
                                     # with rate critic_eta (CritA += eta d, CritU += eta d CritC / (1 + CritC^2):
                                     # normalised so a large input trace cannot make it overshoot; CritC a
                                     # critic_lambda-trace of x_j). Slow critics predict only the state-level
                                     # expectation, so the trial-to-trial variation the selection learns from stays
    critic_level: str = "synapse"    # "synapse": the per-synapse critics above; "neuron": one linear critic per
                                     # neuron on slowly filtered features of its own state (bias, voltage relative to
                                     # the threshold, adaptation, the spike-trace mean, the inhibitory input's mean when
                                     # decorr is on), V_i = sum_f w_f phi_f, local TD error
                                     # d_i = r_i + critic_gamma V_i(t) - V_i(t-1) for all its synapses; TD(lambda) with
                                     # a step normalised by 1 + |eligibility|^2. The reward belongs to the neuron, so
                                     # its baseline does too; the credit stays in each synapse's noise-score trace
    critic_feature_decay: float = 0.99   # filter of the voltage and adaptation features (~100 ms): they then depend
                                     # only weakly on the current noise sample (a baseline must not track it)
    critic_gamma: float = 0.977      # per step (0.5 per 30-step move)
    critic_lambda: float = 0.977
    decorr_norm: str = "covariance"  # "covariance": the reward above (its size also shrinks when the neuron fires
                                     # less, so going quiet lowers the penalty); "correlation": divided by the
                                     # neuron's spiking spread sqrt(zbar (1 - zbar)) and its inhibition's s.d.:
                                     # only how aligned the firing is with the inhibition counts, not how much it fires


@dataclass(frozen=True)
class PlasticInhibitionConfig:
    """Anti-Hebbian inhibitory plasticity on I -> E synapses (cf. Vogels et al. 2011; King, Zylberberg & DeWeese
    2013): |w| += eta * x_I * (y_E - y0), x / y the pre / post spike traces (time constant tau, +1 per spike) and y0
    the post trace at the target rate. Co-active I and E cells (E above target) strengthen the inhibition, so E
    neurons that share drive come to inhibit each other through the I population (decorrelation). Local: pre and post
    spike traces only, no TD error; the weight is updated directly (own rate, no optimiser) and stays inhibitory."""
    eta: float = 1e-5
    rate: float = 10.0               # target postsynaptic rate (Hz) ("rate" mode)
    tau: float = 20.0                # trace time constant (ms)
    mode: str = "rate"               # "rate": |w| += eta * x * (y - y0), inhibition also pins the E rate at `rate`;
                                     # "covariance": |w| += eta * (x - xbar) * (y - ybar), xbar / ybar slow running
                                     # means of each neuron's own trace: only co-activity beyond what the two rates
                                     # predict strengthens inhibition (Foldiak decorrelation, no rate target);
                                     # "psi": |w| += eta * x * (psibar - psi_target), the hidden rule's homeostat
                                     # applied deterministically to the inhibitory synapses (psibar = running mean
                                     # of the postsynaptic pseudo-derivative): inhibition grows onto neurons that
                                     # sit near threshold more than the target and shrinks onto neurons held far
                                     # below it (psibar ~ 0), so it cannot drive V into a deep silent regime. Mostly
                                     # a homeostat: psibar averages over ~100 steps, so it barely sees co-activity;
                                     # "psi_fast": |w| += eta * x * (psi - psi_target) with the instantaneous psi:
                                     # the same homeostasis plus a co-activity term (I cells active while the E
                                     # neuron is near threshold gain weight), Vogels' rule with "near threshold" in
                                     # place of "just spiked"
    tau_mean: float = 1000.0         # running-mean time constant (ms) of the covariance mode
    psi_target: float = 0.1          # "psi" mode: target psibar (the homeostat's psi_target)
    psibar_decay: float = 0.99       # "psi" mode: per-step decay of psibar (as the hidden rule)
    w_max: Optional[float] = None    # cap on |w| (current-based inhibition can otherwise push V without limit; real
                                     # inhibition stops at its reversal potential)
    w_max_rel: Optional[float] = None    # ... the cap as a multiple of the connection's initial mean |w|
    normalise: bool = False          # keep each postsynaptic neuron's total inhibitory input (sum of |w| over its
                                     # incoming synapses of this connection) at its initial value: the rule then only
                                     # redistributes inhibition between I cells, it cannot raise or lower the total
                                     # (heterosynaptic normalisation; applied by the training loop, see
                                     # examples/eprop/inhibition_normaliser.py)
    normalise_every: int = 10        # ... every this many environment moves
    scaling: float = 0.0             # local alternative to `normalise` (synaptic scaling): every synapse multiplies
                                     # its weight by 1 - scaling * (InhAvg - InhTarget) / InhTarget, where InhAvg is
                                     # the postsynaptic neuron's running average of the inhibitory current it receives
                                     # through these synapses and InhTarget that average captured once after a
                                     # warm-up. Uses only the synapse's own weight and two variables of its neuron;
                                     # the plasticity rule then mostly redistributes inhibition. 0 = off
    scaling_tau: float = 1000.0      # ... time constant (ms) of InhAvg
    scaling_warmup: float = 3000.0   # ... InhTarget is captured at this time (ms); scaling acts from then on

    def __post_init__(self):
        if self.mode not in ("rate", "covariance", "psi", "psi_fast"):
            raise ValueError("plastic inhibition mode must be 'rate', 'covariance', 'psi' or 'psi_fast', "
                             f"not {self.mode!r}")


def plastic_inhibition_from(value):
    if isinstance(value, PlasticInhibitionConfig):
        return value
    if value is None or value is True:
        return PlasticInhibitionConfig()
    return PlasticInhibitionConfig(**value)


@dataclass(frozen=True)
class FactorNorm:
    """Normalise each factor of the update by its own running RMS, kept where the factor lives (all local):
    td (the TD error: a running RMS per postsynaptic neuron of the broadcast signal), post (the postsynaptic
    neuron's psi: its closeness to threshold), pre (the presynaptic trace ZFilter, from the presynaptic neuron's
    firing) and noise (the synapse's normalised noise trace, which also depends on the presynaptic firing).
    The TD terms use all four, the local-objective terms post, pre and noise, the firing-rate gradient post and
    pre, the voltage-regulariser gradient pre. Meant for an optimiser without its own per-weight normalisation
    (SGD with momentum). The running means are bias-corrected (they start at t = 0)."""
    td: bool = True
    post: bool = True
    pre: bool = True
    noise: bool = True
    decay: float = 0.9999            # per step (~10 s at dt = 1 ms)
    floor: float = 1e-3              # added in quadrature to each RMS (bounds the gain of near-silent factors)


@dataclass(frozen=True)
class HiddenRuleConfig:
    """Hidden-layer rule of the RL / TD(lambda) variant. Zero coefficients remove their terms."""
    noise: NoisePlacement = NoisePlacement.WEIGHT_SHARED
    estimator: Estimator = Estimator.TRACE
    drift: float = 1.0               # c_P: TD error x lambda-trace of the drift part
    gradient: float = 1.0            # c_G: TD error x lambda-trace of the gradient part
    center_drift: bool = False       # drift modulated by delta minus its running mean (critic bias removed)
    split_optimiser: bool = False    # the gradient part (and the local objectives, credited through the same noise
                                     # score) accumulate in their own gradient variable DeltaGG with its own
                                     # optimiser state; the drift-trace terms stay in DeltaG. Both steps go to the same
                                     # weight, each normalised by its own statistics (one term's noise no longer sets
                                     # the other's step size). Needs the split traces (drift and gradient both used)
    center_eligibility: Optional[float] = None   # decay d of each synapse's running mean of its drift-eligibility
                                     # increment (psi - psibar) * pre * noise, subtracted before it enters the drift
                                     # trace: the part of the drift shared by all states (the state-independent
                                     # direction of the synapse) is removed, the per-state freeze is kept
                                     # (None: off; d should span many states, e.g. 0.9999)
    drift_td: float = 1.0            # weight of the TD error in the drift's modulator (0: no outcome sign)
    surprise_drift: float = 0.0      # c_S: adds c_S * (|delta| - mean|delta|) to the drift's modulator: the drift then
                                     # freezes responses around surprising moments (good or bad) and keeps routine
                                     # ones plastic (salience-gated consolidation); drift_td=0 leaves surprise only
    revive: float = 0.0              # c_R: deterministic revival of neurons outside the psi range, on worse-than-expected
                                     # outcomes: DeltaG -= c_R * max(0, -delta) * lambda-trace(d * ZFilter), with d the
                                     # signed distance outside the range where psi > 0 (in Vthresh, capped at +-1):
                                     # > 0 below it (inputs pushed up), < 0 above it (saturated: pushed down), 0 inside
                                     # and while refractory; dV/dW = ZFilter needs no psi, so it acts where the drift
                                     # and every psi-gated term vanish
    hebb: float = 0.0                # c_B: the potentiating half of the freeze, a TD-gated covariance Hebbian term:
                                     # dW += lr * c_B * delta * trace[(zpost - mean zpost) * ZFilter], zpost the
                                     # postsynaptic spike trace (decay alpha), its mean a running average per neuron
                                     # (hebb_decay); delta is centred with center_drift. A positive TD error
                                     # strengthens the active inputs of the neurons that fired above their average
                                     # (the drift only lowers the near-threshold ones); a negative one reverses it.
                                     # Needs no noise. Stays with the drift under split_optimiser
    hebb_decay: float = 0.9999       # decay of the running mean of the postsynaptic spike trace
    hebb_gate: str = "none"          # "none": the trace of (zpost - mean zpost) * ZFilter; "psi": times the
                                     # postsynaptic pseudo-derivative psi, trace[(zpost - mean) * psi * ZFilter]: the
                                     # contrast reward's expected local update computed directly (the derivative of
                                     # the squared deviation, d(dev^2)/dW ~ 2 dev * psi * pre, with e-prop's
                                     # eligibility psi * ZFilter in place of the noise-measured sensitivity), with
                                     # no noise and no gradient part
    contrast_drift: float = 0.0      # c_K: the contrast leg, a drift with the squared deviation as its gate:
                                     # dW = lr * c_K * delta * trace[(dev^2 - mean dev^2) * pre * xi / sigma^2],
                                     # dev = zpost - mean zpost (the postsynaptic spike trace and its running mean,
                                     # hebb_decay), mean dev^2 a running mean with the same decay; delta centred with
                                     # center_drift. By Stein's lemma its expectation is delta x the weight
                                     # derivative of the squared deviation through the true dynamics,
                                     # 2 dev x (firing sensitivity) x pre: after a positive TD error the response
                                     # sharpens (active neurons up, near-threshold quiet ones down), after a negative
                                     # one it relaxes towards the usual activity. The centred gate cancels the
                                     # REINFORCE part on average, as for the drift. The clean form of
                                     # local.contrast (one trace of the product instead of two traces multiplied).
                                     # Stays with the drift under split_optimiser
    contrast_drift_centre: float = 0.0   # kappa: the gate is dev^2 - kappa * mean dev^2. 0 (default): it keeps a
                                     # REINFORCE part weighted by the neuron's mean squared deviation, with the same
                                     # sign as G (both ascend the return; the drift instead descends its gate, which
                                     # is why its gate needs psi - psibar and G's -psibar has the opposite sign);
                                     # 1: centred, the REINFORCE part cancels on average (the pure local derivative)
    contrast_drift_gate: str = "covariance"   # "covariance": the gate dev^2 (its derivative has a positive mean: firing
                                     # more means more variance, so the mean TD error pushes every rate one way);
                                     # "rate_neutral": dev^2 - F * zpost, F = mean dev^2 / mean zpost (the trace's
                                     # Fano-like ratio, from the running means): a pure rate change leaves it
                                     # unchanged (Poisson-like firing), so the mean TD error has no rate to push and
                                     # the gate is centred by itself (use contrast_drift_centre 0). Expected update:
                                     # delta * (2 dev - F) * sensitivity * pre, a BCM-like threshold at
                                     # mean z + F / 2
    homeostat: float = 0.0           # c_H: c_H * mean|delta| * (psibar - psi_target) on the drift trace
    psi_target: float = 0.1
    psi_target_warmup: Optional[float] = None   # per-neuron homeostat target: each neuron captures its own psibar at
                                     # this time (ms) and the homeostat defends that value from then on (off before);
                                     # neuron-local, unlike a target taken from the population
    homeostat_floor: bool = False    # one-sided homeostat: c_H * mean|delta| * min(0, psibar - psi_target), i.e. it
                                     # only raises the sensitivity of neurons below psi_target (dying: silent or
                                     # saturated) and is zero for the rest, so it does not compete with local rewards
                                     # such as the decorrelation reward over the healthy range (use a low psi_target,
                                     # e.g. 0.02)
    psi_target_spread: float = 0.0   # heterogeneous homeostat targets: psi_target * exp(spread * z_i), z_i a fixed
                                     # standard-uniform-in-log draw per neuron (from its index, so every synapse
                                     # group targeting the neuron agrees); neurons settle at different sparseness
    eprop: float = 0.0               # e-prop learning signal: lambda-trace of eFiltered * (PG - eprop_value * VE)
    eprop_value: float = 0.1
    eprop_broadcast: bool = False    # e-prop without feedback: the learning signal is the broadcast TD error for every
                                     # neuron, i.e. reward x lambda-trace of eFiltered (a three-factor rule, no noise)
    eprop_center_value: bool = False     # the critic part of e-prop's update uses delta minus its running mean: a
                                         # biased critic (E[delta] != 0) otherwise pushes every neuron's incoming
                                         # weights coherently (the eligibility is >= 0), by the sign of its value
                                         # feedback weight; the policy part keeps delta (its score has zero mean)
    eprop_center_td: bool = False        # the whole e-prop update uses delta minus its running mean: the critic part
                                         # as above, and for the policy part a baseline (no change in expectation,
                                         # lower variance, since the score has zero mean)
    eprop_scope: "EpropScope" = None     # None = EpropScope.ALL
    diagnose_gradients: bool = False     # diagnostic: synapses with an e-prop term (and noise) also accumulate
                                         # AccE = sum delta * e-prop trace, AccG = sum delta * gradient-part trace
                                         # (gate -PsiBar) and AccG1 (gate -1), without using G or G1 to learn
    diagnose_parts: bool = False         # diagnostic: synapses with drift and gradient parts also keep each part's
                                         # own lambda-trace (same lambda, normalisation and episodic reset as the
                                         # learning trace) and accumulate AccP = sum delta * drift-part trace and
                                         # AccGp = sum delta * gradient-part trace, the two parts' contributions to
                                         # the TD-driven update (unscaled by factor_norm); learning is unchanged
    recipient_homeostat: bool = False    # SIGNAL_RECIPIENTS: the e-prop populations keep the noise and the homeostat
                                         # (keeps them alive; their learning signal stays e-prop only)
    backprop: float = 0.0            # c_B: TD error x lambda-trace of eFiltered * L, where L is the backward projection
                                     # of the downstream hidden neurons' gradient scores through the forward weights
    backprop_signal: str = "noise"   # what each hidden neuron sends back through its incoming weights: "noise", its
                                     # gradient score -psibar * xi / sigma^2 (symmetric e-prop with perturbation-
                                     # estimated learning signals); "activity", its spike-trace deviation from its
                                     # running mean (zpost - mean zpost, as for hebb): the upstream neuron then gets
                                     # dW = lr * c_B * delta * trace[e * sum_k w_kj (z_k - mean z_k)] (a positive TD
                                     # error makes it drive the targets that fired above their average more); no
                                     # noise needed; delta centred with center_drift
    local: Optional[LocalObjectives] = None
    fire_rate_gradient: bool = True  # e-prop-style firing-rate regulariser c_reg (F - F*) eFiltered
    vreg_gradient: float = 0.0       # e-prop-style voltage regulariser (gradient, not noise-credited): the derivative of
                                     # the squared hinge 1/2 [max(0, V - th)^2 + max(0, -V - th)^2], th = Vthresh +
                                     # Beta A (dV/dW = ZFilter, dA/dW = epsA): pulls saturated neurons' input
                                     # weights down and strongly hyperpolarised ones' up
    factor_norm: Optional[FactorNorm] = None   # normalise every factor of the update locally (see FactorNorm)
    noise_proxy: bool = False        # WEIGHT_SHARED: the noise is still injected, but the score does not read it; each
                                     # neuron estimates it from its own voltage (the per-step change with the reset
                                     # undone, minus its running mean, over its running RMS) and every synapse that
                                     # transmitted adds that estimate to its noise trace (scaled by SynSig)
    noise_proxy_delay: int = 1       # synapse-dynamics steps between a presynaptic spike and the voltage change that
                                     # carries the noise it delivered (1 or 2)
    noise_proxy_decay: float = 0.999 # running mean / variance of the voltage change
    inject_noise: bool = True        # False (noise_proxy only): no noise is injected; the proxy then sees only the
                                     # network's own variability (Poisson inputs, spiking); SynSig only sets its scale
    score_adaptation: Union[bool, str] = False  # presynaptic factor of the noise scores: ZFilter alone (default) or,
                                     # True, ZFilter - Beta epsA (e-prop's ALIF factor; the rule until 2026-10-28);
                                     # "drift" / "gradient": only that part keeps epsA. Stein's lemma already
                                     # differentiates the gate through the adaptation; in the drift, epsA added the
                                     # weight derivative of the adaptation trace, which lowered firing rates by
                                     # ~10-15 Hz without a clear purpose (the regularisers keep epsA)
    kappa: bool = True               # multiply the score by 1 - Alpha^2
    log_syn_sigma: float = 0.0       # initial LogSynSig; SynSig = exp(LogSynSig - 5) (weight noise)
    noise_site: str = "synapse"      # WEIGHT_SHARED: where the neuron's noise sample Noise1 enters the membrane.
                                     # "synapse": every transmitting synapse delivers SynSig * Noise1 (the injected
                                     # noise grows with the number of inputs firing in a step). "soma": the neuron adds
                                     # sigma * Noise1 every step (uniform with activity; sigma = exp(log_syn_sigma - 5)
                                     # * soma_noise_gain), and the synapses only keep the bookkeeping: a synapse adds
                                     # sigma * Noise1 to its noise trace when it transmits (the steps at which its
                                     # weight enters the membrane, where Stein's identity needs the noise), so the
                                     # scores keep their per-synapse form. Same sample and step as the synapse path
    soma_noise_gain: float = 1.0     # "soma": the somatic amplitude in units of exp(log_syn_sigma - 5) (e.g. the mean
                                     # number of inputs firing per step, to match the synapse path's total noise)
    psibar_decay: float = 0.99
    coactivity_decay: Optional[float] = None   # per-synapse co-activity trace CoAct = d CoAct + (1 - d) ZFilter Psi
                                     # (presynaptic activity while the postsynaptic neuron is eligible) and an age
                                     # counter, for activity-dependent rewiring (None: off)
    td_stats_decay: float = 0.999    # running mean and mean |.| of the TD error (centring, homeostat)
    trace_norm: bool = False         # lambda-traces as running averages, X = lambda X + (1 - lambda) score, instead of
                                     # X = lambda X + score: the same geometric weighting of past steps (the same credit),
                                     # but a step size independent of lambda, so the learning rate and the balance with
                                     # the instantaneous local rewards do not change when lambda does

    def __post_init__(self):
        if self.split_optimiser and self.estimator is Estimator.EXACT:
            raise ValueError("split_optimiser: the exact estimator has no drift part")
        uses_noise = (self.drift != 0 or self.gradient != 0 or self.homeostat != 0
                      or (self.local is not None))
        if uses_noise and self.noise is NoisePlacement.NONE:
            raise ValueError("drift, gradient, homeostat and local objectives need noise "
                             "(noise=NoisePlacement.NONE)")
        if self.estimator is Estimator.EXACT:
            if self.drift != 0 or self.homeostat != 0 or self.center_drift or self.center_eligibility is not None:
                raise ValueError("the exact estimator has no drift: use drift=0, homeostat=0")
            if self.noise is NoisePlacement.NODE:
                raise ValueError("the exact estimator is implemented for weight noise only")
        if self.eprop_scope is EpropScope.SIGNAL_RECIPIENTS and (
                self.eprop == 0 or (self.drift == 0 and self.gradient == 0)):
            raise ValueError("eprop_scope=SIGNAL_RECIPIENTS splits the network between e-prop and the perturbation "
                             "rule: it needs eprop != 0 and a drift or gradient part")
        if self.noise_proxy and (self.noise is not NoisePlacement.WEIGHT_SHARED or self.estimator is Estimator.EXACT
                                 or self.noise_proxy_delay not in (1, 2)):
            raise ValueError("noise_proxy needs WEIGHT_SHARED noise, the trace estimator and noise_proxy_delay 1 or 2")
        if self.eprop_broadcast and (self.eprop == 0 or self.eprop_center_value):
            raise ValueError("eprop_broadcast needs eprop != 0 and no eprop_center_value (there is no critic part)")
        if self.score_adaptation not in (True, False, "drift", "gradient"):
            raise ValueError("score_adaptation must be true, false, 'drift' or 'gradient'")
        if self.noise_site not in ("synapse", "soma"):
            raise ValueError(f"noise_site must be 'synapse' or 'soma', not {self.noise_site!r}")
        if self.noise_site == "soma" and (self.noise is not NoisePlacement.WEIGHT_SHARED or self.noise_proxy
                                          or not self.inject_noise or self.estimator is Estimator.EXACT):
            raise ValueError("noise_site 'soma' needs WEIGHT_SHARED noise, the trace estimator, no noise_proxy and "
                             "inject_noise")
        if self.soma_noise_gain <= 0:
            raise ValueError("soma_noise_gain must be > 0")
        if not self.inject_noise and not self.noise_proxy:
            raise ValueError("inject_noise=False needs noise_proxy (the score would read noise that is never injected)")
        if self.backprop != 0 and (self.noise is NoisePlacement.NONE or self.estimator is Estimator.EXACT):
            raise ValueError("backprop propagates the trace estimator's gradient score: it needs noise and "
                             "estimator=TRACE")
        if self.backprop_signal not in ("noise", "activity"):
            raise ValueError(f"backprop_signal must be 'noise' or 'activity', not {self.backprop_signal!r}")
        has_contrast = self.local is not None and self.local.contrast != 0
        if self.hebb_gate not in ("none", "psi"):
            raise ValueError(f"hebb_gate must be 'none' or 'psi', not {self.hebb_gate!r}")
        if self.contrast_drift_gate not in ("covariance", "rate_neutral"):
            raise ValueError(f"contrast_drift_gate must be 'covariance' or 'rate_neutral', not {self.contrast_drift_gate!r}")
        if not 0.0 <= self.contrast_drift_centre <= 1.0:
            raise ValueError("contrast_drift_centre must be in [0, 1]")
        if self.contrast_drift != 0 and (self.noise is NoisePlacement.NONE or self.estimator is Estimator.EXACT):
            raise ValueError("contrast_drift needs noise and estimator=TRACE")
        if (self.center_drift and self.drift == 0 and self.hebb == 0 and not has_contrast and self.contrast_drift == 0
                and not (self.local is not None and self.local.sensitivity != 0)
                and not (self.backprop != 0 and self.backprop_signal == "activity")):
            raise ValueError("center_drift needs drift != 0 (or hebb / contrast != 0)")
        if self.local is not None and self.local.critic_level not in ("synapse", "neuron"):
            raise ValueError("critic_level must be 'synapse' or 'neuron'")
        if self.local is not None and self.local.critic_eta != 0 and self.local.td_route is None:
            raise ValueError("critic_eta needs td_route (the critics act on the TD channel)")
        if self.local is not None and self.local.td_route is not None:
            if self.local.td_route is LocalRoute.WHOLE_TRACE or self.local.route is LocalRoute.WHOLE_TRACE:
                raise ValueError("td_route needs GRADIENT_* routes")
            if self.local.contrast_route is not None or self.local.sensitivity_route is not None:
                raise ValueError("with td_route, contrast_route and sensitivity_route must be None (one TD channel)")
        if self.local is not None and self.local.contrast_baseline not in ("mean", "gated_mean", "gated_synapse"):
            raise ValueError("contrast_baseline must be 'mean', 'gated_mean' or 'gated_synapse', not "
                             f"{self.local.contrast_baseline!r}")
        if self.local is not None and self.local.contrast_baseline != "mean" and not (
                has_contrast and self.local.contrast_route is not None
                and self.local.contrast_route is not self.local.route):
            raise ValueError("a gated contrast_baseline needs the contrast reward on its own route (contrast != 0 and "
                             "contrast_route set, different from route)")
        has_sens = self.local is not None and self.local.sensitivity != 0
        if has_sens and self.estimator is Estimator.EXACT:
            raise ValueError("the sensitivity reward is credited through the gradient score: it needs estimator=TRACE")
        if has_sens and (self.local.sensitivity_route is LocalRoute.WHOLE_TRACE or
                         (self.local.sensitivity_route is None and self.local.route is LocalRoute.WHOLE_TRACE)):
            raise ValueError("the sensitivity reward needs a GRADIENT_* route (or its own sensitivity_route)")
        if has_contrast and self.estimator is Estimator.EXACT:
            raise ValueError("the contrast reward is credited through the gradient score: it needs estimator=TRACE")
        if has_contrast and self.local.contrast_route is LocalRoute.WHOLE_TRACE:
            raise ValueError("contrast_route must be None, GRADIENT_INSTANT or GRADIENT_TRACE")
        if has_contrast and self.local.contrast_route is None and self.local.route is LocalRoute.WHOLE_TRACE:
            raise ValueError("the contrast reward needs a GRADIENT_* route (or its own contrast_route)")
        if self.factor_norm is not None and (self.eprop != 0 or self.backprop != 0 or self.diagnose_gradients
                                             or self.estimator is Estimator.EXACT):
            raise ValueError("factor_norm is implemented for the perturbation rule (trace estimator, no e-prop, "
                             "backprop or gradient diagnostics)")


PRESETS = {
    # original, empirically found rule: gate psi (= drift - gradient), local rewards through the
    # whole trace, |dV| as implemented
    "original": HiddenRuleConfig(drift=1.0, gradient=-1.0, score_adaptation=True,
                                 local=LocalObjectives(route=LocalRoute.WHOLE_TRACE, dv_mode=DVMode.RESET)),
    "proposed": HiddenRuleConfig(drift=1.0, gradient=1.0),
    "proposed_homeostat": HiddenRuleConfig(drift=1.0, gradient=1.0, homeostat=5.0),
    "proposed_full": HiddenRuleConfig(drift=1.0, gradient=1.0, homeostat=5.0,
                                      local=LocalObjectives(route=LocalRoute.GRADIENT_INSTANT)),
    "drift_only": HiddenRuleConfig(drift=1.0, gradient=0.0),
    "gradient_only": HiddenRuleConfig(drift=0.0, gradient=1.0),
    "gradient_only_homeostat": HiddenRuleConfig(drift=0.0, gradient=1.0, homeostat=5.0),
    "unbiased": HiddenRuleConfig(estimator=Estimator.EXACT, drift=0.0, gradient=1.0),
    "node_proposed": HiddenRuleConfig(noise=NoisePlacement.NODE, drift=1.0, gradient=1.0),
    "ind_noise_proposed": HiddenRuleConfig(noise=NoisePlacement.WEIGHT_INDEPENDENT, drift=1.0, gradient=1.0),
    # e-prop (feedback learning signal, no noise) and its combination with the perturbation gradient
    "eprop": HiddenRuleConfig(noise=NoisePlacement.NONE, drift=0.0, gradient=0.0, eprop=1.0),
    "eprop_plus_gradient": HiddenRuleConfig(drift=0.0, gradient=1.0, eprop=1.0),
    # backward propagation of the perturbation gradient through the forward weights (hidden -> hidden)
    "proposed_backprop": HiddenRuleConfig(drift=1.0, gradient=1.0, backprop=1.0),
    # symmetric e-prop where the readout gradient reaches (populations connected to the heads), the proposed rule
    # with the homeostat (centred) everywhere else; compile with feedback_type="symmetric" and no explicit feedback
    # connections, so that the forward readout weights are the only e-prop signal
    "symmetric_hybrid": HiddenRuleConfig(drift=1.0, gradient=1.0, homeostat=1.0, center_drift=True, eprop=1.0,
                                         eprop_value=1.0, eprop_scope=EpropScope.SIGNAL_RECIPIENTS),
    "symmetric_hybrid_homeostat": HiddenRuleConfig(drift=1.0, gradient=1.0, homeostat=1.0, center_drift=True,
                                                   eprop=1.0, eprop_value=1.0, recipient_homeostat=True,
                                                   eprop_scope=EpropScope.SIGNAL_RECIPIENTS),
    "proposed_homeostat_backprop": HiddenRuleConfig(drift=1.0, gradient=1.0, homeostat=1.0, center_drift=True,
                                                    backprop=1.0),
    "eprop_plus_drift": HiddenRuleConfig(drift=1.0, gradient=0.0, eprop=1.0),
    "eprop_plus_proposed": HiddenRuleConfig(drift=1.0, gradient=1.0, eprop=1.0),
    "eprop_plus_proposed_homeostat": HiddenRuleConfig(drift=1.0, gradient=1.0, homeostat=5.0, eprop=1.0),
    "eprop_plus_homeostat": HiddenRuleConfig(drift=0.0, gradient=0.0, homeostat=5.0, eprop=1.0),
    # hidden layer learns only the firing-rate regulariser (the "baseline"; set f_target in the compiler)
    "baseline": HiddenRuleConfig(noise=NoisePlacement.NONE, drift=0.0, gradient=0.0),
    # reservoir: no hidden learning at all (the weights stay at their initial values); only the readout learns
    "reservoir": HiddenRuleConfig(noise=NoisePlacement.NONE, drift=0.0, gradient=0.0, fire_rate_gradient=False),
}


def receives_eprop_signal(pop, policy_heads, value_head) -> bool:
    """Whether a hidden population receives an e-prop learning signal: it has a connection (forward or feedback,
    not the TD-error transport) to a policy or value head."""
    heads = set(policy_heads or ()) | ({value_head} if value_head is not None else set())
    for ref in pop.outgoing_connections:
        conn = ref()
        if conn is None or conn.target() not in heads:
            continue
        if conn.is_feedback and not (conn.name.endswith("policy_feedback") or conn.name.endswith("value_feedback")):
            continue
        return True
    return False


def rule_for_population(rule: HiddenRuleConfig, pop, policy_heads, value_head) -> HiddenRuleConfig:
    """The rule used by the synapses targeting `pop` (EpropScope.SIGNAL_RECIPIENTS splits the network)."""
    if rule.eprop_scope is not EpropScope.SIGNAL_RECIPIENTS:
        return rule
    if receives_eprop_signal(pop, policy_heads, value_head):
        if rule.diagnose_gradients:       # keep the noise so that both estimators can be compared
            return replace(rule, drift=0.0, gradient=0.0, homeostat=0.0, center_drift=False, backprop=0.0,
                           local=None, eprop_scope=None, recipient_homeostat=False)
        if rule.recipient_homeostat and rule.homeostat != 0:
            return replace(rule, drift=0.0, gradient=0.0, center_drift=False, backprop=0.0, local=None,
                           eprop_scope=None, recipient_homeostat=False)
        return replace(rule, noise=NoisePlacement.NONE, drift=0.0, gradient=0.0, homeostat=0.0,
                       center_drift=False, backprop=0.0, local=None, eprop_scope=None, recipient_homeostat=False)
    return replace(rule, eprop=0.0, eprop_scope=None, diagnose_gradients=False)


def get_hidden_rule(rule) -> HiddenRuleConfig:
    """Accept a HiddenRuleConfig, a preset name or a dict (see :func:`hidden_rule_from_dict`)."""
    if isinstance(rule, HiddenRuleConfig):
        return rule
    if isinstance(rule, dict):
        return hidden_rule_from_dict(rule)
    try:
        return PRESETS[rule]
    except KeyError:
        raise ValueError(f"unknown hidden rule preset '{rule}'; available: {sorted(PRESETS)}")


_ENUM_FIELDS = {"noise": NoisePlacement, "estimator": Estimator, "route": LocalRoute, "contrast_route": LocalRoute, "sensitivity_route": LocalRoute, "td_route": LocalRoute,
                "dv_mode": DVMode,
                "eprop_scope": EpropScope}


def _from_plain(cls, d):
    kw = {}
    for k, v in d.items():
        if k not in cls.__dataclass_fields__:
            raise ValueError(f"{cls.__name__} has no field '{k}'")
        kw[k] = _ENUM_FIELDS[k][v] if k in _ENUM_FIELDS and isinstance(v, str) else v
    return kw


def hidden_rule_from_dict(d: dict) -> HiddenRuleConfig:
    """Build a rule from plain JSON-like data: ``{"preset": "proposed", "homeostat": 5.0,
    "noise": "NODE", "local": {"route": "GRADIENT_INSTANT", "dv": 0.01}}``. Fields override the preset
    (default: the HiddenRuleConfig defaults); enums are given by member name; ``"local": null`` removes
    the local objectives."""
    d = dict(d)
    base = get_hidden_rule(d.pop("preset")) if "preset" in d else HiddenRuleConfig()
    kw = _from_plain(HiddenRuleConfig, d)
    if isinstance(kw.get("local"), dict):
        kw["local"] = LocalObjectives(**_from_plain(LocalObjectives, kw["local"]))
    if kw.get("factor_norm") is True:
        kw["factor_norm"] = FactorNorm()
    elif isinstance(kw.get("factor_norm"), dict):
        kw["factor_norm"] = FactorNorm(**_from_plain(FactorNorm, kw["factor_norm"]))
    return replace(base, **kw)


def hidden_rule_to_dict(rule: HiddenRuleConfig) -> dict:
    """Inverse of :func:`hidden_rule_from_dict` (all fields, enums by name)."""
    plain = lambda v: v.name if isinstance(v, Enum) else v
    out = {k: plain(getattr(rule, k)) for k in rule.__dataclass_fields__}
    if rule.local is not None:
        out["local"] = {k: plain(getattr(rule.local, k)) for k in rule.local.__dataclass_fields__}
    if rule.factor_norm is not None:
        out["factor_norm"] = {k: getattr(rule.factor_norm, k) for k in rule.factor_norm.__dataclass_fields__}
    return out


def as_policy_type(value) -> PolicyType:
    """Accept PolicyType, the monolithic compiler's PolicyTypes, or their string values."""
    if isinstance(value, PolicyType):
        return value
    return PolicyType(getattr(value, "value", value))
