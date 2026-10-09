"""Snake's network, built from a declarative spec instead of code in snake.py.

    NetworkSpec  -> build_network(spec) -> Built (network, input, layers, fields, heads, labels, population_rules)

The default spec (spec_from_hparams with the default hyperparameters) reproduces the network snake.py used to build:
input -> stacked excitatory/inhibitory (EI) recurrent layers -> policy and value fields -> policy and value heads,
with TD-error transport connections into the value head and, optionally, explicit random feedback connections.

Config ("network" key of SNAKE_CONFIG), any subset, e.g.
    {"layers": [{"e": 30}, {"e": 20}, {"e": 10, "i": 8, "rule": {"preset": "eprop"}}],
     "fields": {"shape": 6}, "input": {"fan_in": 200}}
Shapes given as an int n mean (n, n, channels). A layer or the fields may carry their own learning rule ("rule"), passed
to the compiler as population_rules (the layer split: e.g. perturbation-based in wide layers, e-prop in narrow ones).
"""
from dataclasses import dataclass, field, fields as dc_fields, replace
from typing import Optional, Tuple, Union

import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.connectivity import Dense, FixedProbability, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import AdaptiveLeakyIntegrateFire, LeakyIntegrate, PoissonInput

Shape = Tuple[int, int, int]


@dataclass
class ConnSpec:
    """How one population connects to another (sign given by the source: E +1, I -1, readout None)."""
    type: str = "toroidal"           # "toroidal" | "fixed"
    fan_in: Optional[float] = None   # toroidal: requested fan-in (None: the layer's default)
    sigma: Optional[float] = None    # toroidal: Gaussian width in units of the grid (None: the layer's sigma)
    p: float = 0.01                  # fixed: connection probability
    mean_scale: float = 0.1          # weight mean = sign * mean_scale / sqrt(fan_in) (x3 for inhibitory, toroidal)
    sd_scale: float = 0.05           # weight s.d. = sd_scale / sqrt(fan_in)  (1 / sqrt(fan_in) without a sign)
    exact: bool = True               # toroidal sampler (ToroidalGaussian2D exact)


@dataclass
class NeuronSpec:
    v_thresh: float = 0.61
    tau_mem: float = 10.0
    tau_refrac: float = 3.0
    tau_adapt: float = 300.0
    beta: float = 0.0174


@dataclass
class LayerSpec:
    e_shape: Shape = (20, 20, 3)
    i_shape: Shape = (15, 15, 3)
    neuron: NeuronSpec = field(default_factory=NeuronSpec)
    sigma: float = 0.05
    fan_in_e: float = 300            # fan-in of connections whose source is an E population (E->E, E->I, E->next)
    fan_in_i: float = 300            # fan-in of connections whose source is an I population
    internal: ConnSpec = field(default_factory=lambda: ConnSpec(p=0.005))   # the four recurrent E/I connections
    to_next: ConnSpec = field(default_factory=ConnSpec)        # this layer -> next layer (E and I sources)
    to_next_i: Optional[ConnSpec] = None                       # ... from the I population only (None: to_next). The
                                                               # inhibitory scale is x3, so with equal rates the
                                                               # feedforward inhibition outweighs the excitation 3:1;
                                                               # mean_scale 0.033 here balances it
    rule: Optional[Union[str, dict]] = None                    # learning rule of this layer (None: the default)


@dataclass
class FieldSpec:
    shape: Shape = (15, 15, 3)       # each field (policy, value)
    neuron: NeuronSpec = field(default_factory=lambda: NeuronSpec(beta=0.0174))
    conn: ConnSpec = field(default_factory=lambda: ConnSpec(p=0.005))   # last layer -> field
    conn_i: Optional[ConnSpec] = None                                   # ... from the I population only (None: conn)
    readout_p: float = 0.99999       # field -> head (fixed probability, weights N(0, 1 / sqrt(fan_in)))
    rule: Optional[Union[str, dict]] = None


@dataclass
class NetworkSpec:
    input_shape: Shape = (20, 20, 3)
    input: ConnSpec = field(default_factory=lambda: ConnSpec(fan_in=300, sigma=0.1))   # input -> first layer
    layers: list = field(default_factory=lambda: [LayerSpec()])
    fields: Optional[FieldSpec] = field(default_factory=FieldSpec)   # None: heads read the last layer directly
    num_actions: int = 4
    head_tau_mem: float = 10.0
    direct_readout_p: float = 0.1    # without fields: last layer -> heads (fixed probability)
    explicit_feedback: bool = True   # random feedback connections from every hidden population to the heads
    feedback_p: float = 1.0          # ... from the EI layers (from the fields: readout_p)
    node_sigma: float = 1e-2         # membrane-noise s.d. of every hidden neuron (used by node-noise rules only)


@dataclass
class Built:
    network: Network
    input: Population
    layers: list          # [(e, i), ...]
    fields: Optional[tuple]   # (policy_field, value_field) or None
    policy: Population
    value: Population
    labels: dict          # Population -> name used in logs (L1_E, L1_I, ..., policy_field, value_field)
    population_rules: dict    # Population -> rule (only populations with their own rule)

    @property
    def hidden(self):
        pops = [p for layer in self.layers for p in layer]
        return pops + list(self.fields or ())


# --------------------------------------------------------------------------------------------------------------
def connectivity(conn: ConnSpec, src_shape, sign, fan_in=None, sigma=None):
    """The connectivity object of one connection (same weight scaling as snake.py's make_connectivity)."""
    mean_scale, sd_scale = conn.mean_scale, conn.sd_scale
    if conn.type == "fixed":
        fan = conn.p * np.prod(src_shape)
        if sign is None:
            sd_scale = 1.0
        return FixedProbability(conn.p, Normal(mean=(sign or 0) * mean_scale / np.sqrt(fan), sd=sd_scale / np.sqrt(fan)))
    if conn.type != "toroidal":
        raise ValueError(f"unknown connectivity type {conn.type}")
    fan = conn.fan_in if conn.fan_in is not None else fan_in
    sig = conn.sigma if conn.sigma is not None else sigma
    if fan is None or sig is None:
        raise ValueError("toroidal connectivity needs fan_in and sigma")
    if sign == -1:
        mean_scale *= 3
    elif sign is None:
        sd_scale = 1.0
    return ToroidalGaussian2D(sigma=sig, fan_in=fan, weight=Normal(mean=(sign or 0) * mean_scale / np.sqrt(fan),
                                                                    sd=sd_scale / np.sqrt(fan)),
                              exact=conn.exact)


def _alif(neuron: NeuronSpec, node_sigma):
    return AdaptiveLeakyIntegrateFire(v_thresh=neuron.v_thresh, tau_mem=neuron.tau_mem, tau_refrac=neuron.tau_refrac,
                                      tau_adapt=neuron.tau_adapt, beta=neuron.beta, perturbation_eps_std=node_sigma)


def build_network(spec: NetworkSpec, default_params) -> Built:
    network = Network(default_params)
    labels, rules = {}, {}
    with network:
        inp = Population(PoissonInput(), spec.input_shape)
        layers = []
        for k, ls in enumerate(spec.layers):
            e = Population(_alif(ls.neuron, spec.node_sigma), ls.e_shape)
            i = Population(_alif(ls.neuron, spec.node_sigma), ls.i_shape)
            for pre, post, src_shape, fan, sign in ((e, e, ls.e_shape, ls.fan_in_e, +1), (e, i, ls.e_shape, ls.fan_in_e, +1),
                                                    (i, e, ls.i_shape, ls.fan_in_i, -1), (i, i, ls.i_shape, ls.fan_in_i, -1)):
                Connection(pre, post, connectivity(ls.internal, src_shape, sign, fan, ls.sigma), exc_inh_sign=sign)
            labels[e], labels[i] = f"L{k + 1}_E", f"L{k + 1}_I"
            if ls.rule is not None:
                rules[e] = rules[i] = ls.rule
            layers.append((e, i))

        fields = None
        if spec.fields is not None:
            fs = spec.fields
            fields = (Population(_alif(fs.neuron, spec.node_sigma), fs.shape),
                      Population(_alif(fs.neuron, spec.node_sigma), fs.shape))
            labels[fields[0]], labels[fields[1]] = "policy_field", "value_field"
            if fs.rule is not None:
                rules[fields[0]] = rules[fields[1]] = fs.rule

        policy = Population(LeakyIntegrate(tau_mem=spec.head_tau_mem, bias=0.0, readout="var"), spec.num_actions)
        value = Population(LeakyIntegrate(tau_mem=spec.head_tau_mem, bias=0.0, readout="var"), 1)

        # input -> first layer (excitatory)
        first = spec.layers[0]
        for target in layers[0]:
            Connection(inp, target, connectivity(spec.input, spec.input_shape, +1, first.fan_in_e, first.sigma),
                       exc_inh_sign=+1)
        # layer k -> layer k+1
        for k in range(len(layers) - 1):
            ls = spec.layers[k]
            for src, src_shape, fan, sign, conn in ((layers[k][0], ls.e_shape, ls.fan_in_e, +1, ls.to_next),
                                                    (layers[k][1], ls.i_shape, ls.fan_in_i, -1, ls.to_next_i or ls.to_next)):
                for target in layers[k + 1]:
                    Connection(src, target, connectivity(conn, src_shape, sign, fan, ls.sigma), exc_inh_sign=sign)
        last, last_spec = layers[-1], spec.layers[-1]
        if fields is not None:
            fs = spec.fields
            for fld in fields:                                   # last layer -> fields
                for src, src_shape, fan, sign, conn in ((last[0], last_spec.e_shape, last_spec.fan_in_e, +1, fs.conn),
                                                        (last[1], last_spec.i_shape, last_spec.fan_in_i, -1,
                                                         fs.conn_i or fs.conn)):
                    Connection(src, fld, connectivity(conn, src_shape, sign, fan, last_spec.sigma), exc_inh_sign=sign)
            for fld, head, name in ((fields[0], policy, "policy_feedback"), (fields[1], value, "value_feedback")):
                Connection(fld, head, connectivity(ConnSpec(type="fixed", p=fs.readout_p), fs.shape, None),
                           exc_inh_sign=None)
                if spec.explicit_feedback:
                    Connection(fld, head, FixedProbability(fs.readout_p, Normal(sd=1.0 / np.sqrt(spec.num_actions))),
                               feedback_name=name, exc_inh_sign=None)
        else:                                                    # heads read the last layer directly
            for src, src_shape in ((last[0], last_spec.e_shape), (last[1], last_spec.i_shape)):
                for head in (policy, value):
                    Connection(src, head, connectivity(ConnSpec(type="fixed", p=spec.direct_readout_p), src_shape, None),
                               exc_inh_sign=None)

        # TD-error transport into the value head: from the policy head and every hidden population
        Connection(policy, value, Dense(weight=1.0), feedback_name="tde_transport")
        for e, i in layers:
            for pop in (e, i):
                Connection(pop, value, Dense(weight=1.0), feedback_name="tde_transport")
        for fld in (fields or ()):
            Connection(fld, value, Dense(weight=1.0), feedback_name="tde_transport")

        # explicit random feedback from the EI layers
        if spec.explicit_feedback:
            for e, i in layers:
                for pop in (e, i):
                    for head, name in ((policy, "policy_feedback"), (value, "value_feedback")):
                        Connection(pop, head, FixedProbability(spec.feedback_p,
                                                               Normal(sd=1.0 / np.sqrt(spec.num_actions))),
                                   feedback_name=name, exc_inh_sign=None)
    return Built(network, inp, layers, fields, policy, value, labels, rules)


# --------------------------------------------------------------------------------------------------------------
def _shape(x, channels):
    if isinstance(x, int):
        return (x, x, channels)
    return tuple(x)


def _merge(obj, d):
    """Recursively update a dataclass from a plain dict (unknown keys are an error)."""
    if d is None:
        return obj
    kw = {}
    names = {f.name for f in dc_fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"{type(obj).__name__} has no field '{k}' (fields: {sorted(names)})")
        cur = getattr(obj, k)
        if cur is None and k == "to_next_i" and isinstance(v, dict):
            cur = obj.to_next                   # inhibitory feedforward: start from the layer's to_next settings
        if cur is None and k == "conn_i" and isinstance(v, dict):
            cur = obj.conn
        kw[k] = _merge(cur, v) if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict) else v
    return replace(obj, **kw)


def spec_from_hparams(hp, input_shape, num_actions, channels=3):
    """The spec of snake.py's network from the hyperparameters; the optional "network" key overrides any part.

    Layer shortcuts in "network": {"layers": [{"e": 30, "i": 20, "rule": ...}, ...]} (sizes as ints) or full LayerSpec
    fields; "fields": null removes the fields; {"fields": {"shape": 6}} resizes them."""
    exact = hp.get("toroidal", "exact") == "exact"
    base_layer = LayerSpec(e_shape=(hp["hid_e"],) * 2 + (channels,), i_shape=(hp["hid_i"],) * 2 + (channels,),
                           fan_in_e=hp["fan_in"], fan_in_i=hp["fan_in"],
                           internal=ConnSpec(exact=exact), to_next=ConnSpec(exact=exact))
    spec = NetworkSpec(input_shape=input_shape, num_actions=num_actions,
                       input=ConnSpec(fan_in=hp["fan_in"], sigma=0.1, exact=exact),
                       layers=[base_layer for _ in range(hp["ei_layers"])],
                       fields=FieldSpec(shape=(hp["hid_i"],) * 2 + (channels,), conn=ConnSpec(p=0.005, exact=exact)),
                       explicit_feedback=hp["explicit_feedback"], node_sigma=hp["node_sigma"])
    net = dict(hp.get("network") or {})
    if "layers" in net:
        layers = []
        for ld in net.pop("layers"):
            ld = dict(ld)
            short = {}
            if "e" in ld:
                short["e_shape"] = _shape(ld.pop("e"), channels)
            if "i" in ld:
                short["i_shape"] = _shape(ld.pop("i"), channels)
            elif "e_shape" in short:      # default inhibitory size: 3/4 of the excitatory side (20 -> 15)
                short["i_shape"] = (max(1, round(short["e_shape"][0] * 0.75)),) * 2 + (channels,)
            for k in ("e_shape", "i_shape"):
                if k in ld:
                    ld[k] = _shape(ld[k], channels)
            layers.append(_merge(replace(base_layer, **short), ld))
        spec = replace(spec, layers=layers)
    if "fields" in net:
        fd = net.pop("fields")
        if fd is None:
            spec = replace(spec, fields=None)
        else:
            fd = dict(fd)
            if "shape" in fd:
                fd["shape"] = _shape(fd["shape"], channels)
            spec = replace(spec, fields=_merge(spec.fields, fd))
    return _merge(spec, net)
