"""Activity-dependent rewiring: synapses chosen on the host (e.g. those whose pre- and postsynaptic neurons are
rarely co-active) are removed and, for the same postsynaptic neuron, a new synapse is created from a different
presynaptic neuron. The host marks synapses to remove by setting their weight to REMOVE (1e30, a value no weight
reaches; NaN tests are unavailable in GeNN code and unsafe under fast-math) and writes the new targets per
presynaptic row (AddTarget, NumAdd); one device pass then removes the marked synapses and adds the planned ones.
The planning (which synapses, which new presynaptic neurons) is up to the caller, so the rule stays local: here it
is examples/eprop/activity_rewiring.py."""
import numpy as np

from ..utils.model import CustomConnectivityUpdateModel

REMOVE = 1e30

def _model(norm_stat):
    """The device pass; with norm_stat (a per-synapse running statistic of factor normalisation, XiSq), a new
    synapse starts from the mean of that statistic over its presynaptic row instead of 0 (which would divide its
    update by the floor alone)."""
    m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _BASE.items()}
    if norm_stat is None:
        return m
    m["var_refs"] = _BASE["var_refs"] + [(norm_stat, "scalar")]
    m["row_update_code"] = f"""
    scalar statSum = 0.0;
    unsigned int statN = 0;
    for_each_synapse {{
        if (g > 1e29) {{
            remove_synapse();
        }}
        else {{
            statSum += {norm_stat};
            statN++;
        }}
    }}
    const scalar statInit = (statN > 0) ? statSum / (scalar)statN : 0.0;
    for (unsigned int k = 0; k < NumAdd; k++) {{
        add_synapse(AddTarget[(id_pre * MaxAdd) + k], (scalar)Sign * 1e-5, statInit);
    }}
    NumAdd = 0;
    """
    return m


_BASE = {
    "params": [("Sign", "int"), ("MaxAdd", "unsigned int")],
    "pre_vars": [("NumAdd", "unsigned int")],
    "var_refs": [("g", "scalar")],
    "extra_global_params": [("AddTarget", "uint32_t*")],
    "row_update_code": """
    for_each_synapse {
        if (g > 1e29) {
            remove_synapse();
        }
    }
    for (unsigned int k = 0; k < NumAdd; k++) {
        add_synapse(AddTarget[(id_pre * MaxAdd) + k], (scalar)Sign * 1e-5);
    }
    NumAdd = 0;
    """}


def add_activity_rewiring(synapse_group, genn_model, compiler, sign, weight_var_ref, max_add=8, headroom=1.5,
                          norm_stat_ref=None, norm_stat="XiSq"):
    """Add the device pass ("ActRewire") to a sparse synapse group and give its rows spare capacity (headroom times
    the current longest row), since rows can grow when synapses move between presynaptic neurons. norm_stat_ref:
    reference to a per-synapse normalisation statistic that new synapses initialise from their row's mean."""
    synapse_group.max_connections = int(np.ceil(synapse_group.max_connections * headroom)) + max_add
    model = CustomConnectivityUpdateModel(
        _model(norm_stat if norm_stat_ref is not None else None),
        param_vals={"Sign": int(sign or 0), "MaxAdd": int(max_add)},
        pre_var_vals={"NumAdd": 0},
        var_refs={"g": weight_var_ref, **({norm_stat: norm_stat_ref} if norm_stat_ref is not None else {})},
        egp_vals={"AddTarget": np.zeros(synapse_group.src.num_neurons * max_add, dtype=np.uint32)})
    return compiler.add_custom_connectivity_update(genn_model, model, synapse_group, "ActRewire",
                                                   "ActRewire_" + synapse_group.name)
