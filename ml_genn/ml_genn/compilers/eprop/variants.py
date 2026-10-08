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
from typing import Optional


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


@dataclass(frozen=True)
class HiddenRuleConfig:
    """Hidden-layer rule of the RL / TD(lambda) variant. Zero coefficients remove their terms."""
    noise: NoisePlacement = NoisePlacement.WEIGHT_SHARED
    estimator: Estimator = Estimator.TRACE
    drift: float = 1.0               # c_P: TD error x lambda-trace of the drift part
    gradient: float = 1.0            # c_G: TD error x lambda-trace of the gradient part
    center_drift: bool = False       # drift modulated by delta minus its running mean (critic bias removed)
    homeostat: float = 0.0           # c_H: c_H * mean|delta| * (psibar - psi_target) on the drift trace
    psi_target: float = 0.1
    eprop: float = 0.0               # e-prop learning signal: lambda-trace of eFiltered * (PG - eprop_value * VE)
    eprop_value: float = 0.1
    local: Optional[LocalObjectives] = None
    fire_rate_gradient: bool = True  # e-prop-style firing-rate regulariser c_reg (F - F*) eFiltered
    kappa: bool = True               # multiply the score by 1 - Alpha^2
    log_syn_sigma: float = 0.0       # initial LogSynSig; SynSig = exp(LogSynSig - 5) (weight noise)
    psibar_decay: float = 0.99
    td_stats_decay: float = 0.999    # running mean and mean |.| of the TD error (centring, homeostat)

    def __post_init__(self):
        uses_noise = (self.drift != 0 or self.gradient != 0 or self.homeostat != 0
                      or (self.local is not None))
        if uses_noise and self.noise is NoisePlacement.NONE:
            raise ValueError("drift, gradient, homeostat and local objectives need noise "
                             "(noise=NoisePlacement.NONE)")
        if self.estimator is Estimator.EXACT:
            if self.drift != 0 or self.homeostat != 0 or self.center_drift:
                raise ValueError("the exact estimator has no drift: use drift=0, homeostat=0")
            if self.noise is NoisePlacement.NODE:
                raise ValueError("the exact estimator is implemented for weight noise only")
        if self.center_drift and self.drift == 0:
            raise ValueError("center_drift needs drift != 0")


PRESETS = {
    # original, empirically found rule: gate psi (= drift - gradient), local rewards through the
    # whole trace, |dV| as implemented
    "original": HiddenRuleConfig(drift=1.0, gradient=-1.0,
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
    # hidden layer learns only the firing-rate regulariser (the "baseline"; set f_target in the compiler)
    "baseline": HiddenRuleConfig(noise=NoisePlacement.NONE, drift=0.0, gradient=0.0),
}


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


_ENUM_FIELDS = {"noise": NoisePlacement, "estimator": Estimator, "route": LocalRoute, "dv_mode": DVMode}


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
    return replace(base, **kw)


def hidden_rule_to_dict(rule: HiddenRuleConfig) -> dict:
    """Inverse of :func:`hidden_rule_from_dict` (all fields, enums by name)."""
    plain = lambda v: v.name if isinstance(v, Enum) else v
    out = {k: plain(getattr(rule, k)) for k in rule.__dataclass_fields__}
    if rule.local is not None:
        out["local"] = {k: plain(getattr(rule.local, k)) for k in rule.local.__dataclass_fields__}
    return out


def as_policy_type(value) -> PolicyType:
    """Accept PolicyType, the monolithic compiler's PolicyTypes, or their string values."""
    if isinstance(value, PolicyType):
        return value
    return PolicyType(getattr(value, "value", value))
