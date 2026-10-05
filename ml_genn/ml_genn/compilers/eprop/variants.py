"""Enumerations describing which e-prop weight-update variant to use.

These enums replace the old "comment/uncomment a block of code" workflow.
Pick a member of each enum that's relevant to your connection/population
and pass it through to the compiler; `models.get_model(...)` will look up
the right GeNN model dict for you.
"""

from enum import Enum, auto


class FeedbackType(Enum):
    """How error/learning-signal feedback reaches hidden neurons."""
    SYMMETRIC = "symmetric"   # feedback weights tied to forward readout weights
    RANDOM = "random"         # fixed random feedback weights
    ADAPTIVE = "adaptive"     # learned feedback weights


class HiddenRule(Enum):
    """Which local eligibility-trace rule a hidden-layer synapse uses."""
    LIF = auto()          # LIF postsynaptic neuron, perturbation variant
    ALIF = auto()         # vanilla ALIF e-prop
    ALIF_TD_LAMBDA = auto()  # RL / TD(lambda) ALIF variant


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
    """Policy head output distribution."""
    CATEGORICAL = "categorical"
    GAUSSIAN_TRACE = "gaussian_trace"
    GENERIC = "generic"


class NoiseSource(Enum):
    """Where distributed e-prop's perturbation noise is drawn from."""
    NODE = auto()                # single Noise1, shared across all synapses of neuron j ("node dist")
    SYNAPTIC_ROTATING = auto()   # selector cycles Noise1..Noise4 per synapse (the commented-out `selector` block)


class PseudoDerivativeGate(Enum):
    """Whether the e-prop pseudo-derivative gates the perturbation trace."""
    ENABLED = auto()    # standard, variance-reducing gate (default / recommended)
    DISABLED = auto()   # "weight dist no heuristic" ablation — kept only for reproducing Fig. snake-comparison


class RoutingMode(Enum):
    """Optional structured routing of the local policy-gradient estimator
    through the recurrent weights (Appendix: Structured Routing)."""
    NONE = auto()            # broadcast-only (Eq. distributed-update) — used throughout main text
    ADDITIVE = auto()        # Eq. routed-update, additive term
    MULTIPLICATIVE = auto()  # (1 + L^t_j) multiplicative variant