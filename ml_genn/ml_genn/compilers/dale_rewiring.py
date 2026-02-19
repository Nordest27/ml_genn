import numpy as np

from pygenn import VarAccessMode
from pygenn import create_egp_ref, create_pre_var_ref

from ..callbacks import Callback
from ..utils.model import CustomConnectivityUpdateModel


# ================================================================
# Dale-Constrained Rewiring models
# ================================================================

# ------------------------------------------------
# 1. Prune synapses that violate Dale's law
# ------------------------------------------------
dale_prune_model = {
    "params": [
        ("NumRowWords", "unsigned int"),
        ("Sign", "int")  # +1 excitatory, -1 inhibitory
    ],
    "var_refs": [
        ("g", "scalar", VarAccessMode.READ_ONLY)
    ],
    "pre_vars": [
        ("NumPruned", "unsigned int")
    ],
    "extra_global_params": [
        ("Connectivity", "uint32_t*")
    ],

    "row_update_code": """
    NumPruned = 0;
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);

    for_each_synapse {
        const bool violates =
            (Sign > 0 && g < 0.0) ||
            (Sign < 0 && g > 0.0);

        if(violates) {
            NumPruned++;
            rowConnectivity[id_post / 32] &= ~(1 << (id_post % 32));
            remove_synapse();
        }
    }
    """
}


# ------------------------------------------------
# 2. Rewire the same number of synapses
# ------------------------------------------------
dale_rewire_model = {
    "params": [
        ("NumRowWords", "unsigned int"),
        ("Sign", "int")
    ],
    "pre_var_refs": [
        ("NumPruned", "unsigned int")
    ],
    "var_refs": [
        ("g", "scalar")
    ],
    "extra_global_param_refs": [
        ("Connectivity", "uint32_t*")
    ],
    "row_update_code": """
    // Loop through synapses to activate
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);
    
    for(unsigned int i = 0; i < NumPruned; i++) {
        while(true) {
            // Pick a random synapse to activate
            const unsigned int j = gennrand() % num_post;
            
            if(!(rowConnectivity[j / 32] & (1 << (j % 32)))) {
                add_synapse(j, Sign*0.001);
                rowConnectivity[j / 32] |= (1 << (j % 32));
                break;
            }
        }
    }"""
}


# ================================================================
# Optional callback to record number of rewired synapses
# ================================================================
class DaleRewiringRecord(Callback):
    """
    Records the total number of pruned / rewired synapses per batch.
    """
    def __init__(self, prune_ccu, key=None):
        self.num_pruned = prune_ccu.pre_vars["NumPruned"]
        self.key = key

    def set_params(self, data, **kwargs):
        data[self.key] = []
        self._data = data[self.key]

    def on_batch_end(self, batch, metrics):
        # Sum over all presynaptic neurons
        self._data.append(int(np.sum(self.num_pruned.view)))

    def get_data(self):
        return self.key, self._data


# ================================================================
# Public API: add_dale_rewiring
# ================================================================
def add_dale_rewiring(
    synapse_group,
    genn_model,
    compiler,
    sign,
    weight_var_ref
):
    """
    Adds Dale-Constrained Rewiring (DCR) to a synapse group.

    Parameters
    ----------
    synapse_group :
        GeNN synapse population
    genn_model :
        GeNN model
    compiler :
        Compiler / builder object
    sign : int
        +1 for excitatory, -1 for inhibitory
    weight_var_ref :
        Var reference to synaptic weight (e.g. create_wu_var_ref(..., "g"))

    Returns
    -------
    (genn_dale_prune, genn_dale_rewire)
        CustomConnectivityUpdate objects
    """
    # print(synapse_group)
    # Calculate connectivity bitmask size
    num_row_words = (synapse_group.trg.num_neurons + 31) // 32
    num_words = synapse_group.src.num_neurons * num_row_words

    # Shared connectivity bitmask
    connectivity = np.zeros(num_words, dtype=np.uint32)

    # -----------------------------
    # Dale prune step
    # -----------------------------
    dale_prune = CustomConnectivityUpdateModel(
        dale_prune_model,
        param_vals={
            "NumRowWords": num_row_words,
            "Sign": sign
        },
        pre_var_vals={
            "NumPruned": 0
        },
        var_refs={
            "g": weight_var_ref
        },
        egp_vals={
            "Connectivity": connectivity
        }
    )

    genn_dale_prune = compiler.add_custom_connectivity_update(
        genn_model,
        dale_prune,
        synapse_group,
        "DalePrune",
        "DalePrune_" + synapse_group.name
    )

    # -----------------------------
    # Dale rewire step
    # -----------------------------
    dale_rewire = CustomConnectivityUpdateModel(
        dale_rewire_model,
        param_vals={
            "NumRowWords": num_row_words,
            "Sign": sign
        },
        pre_var_refs={
            "NumPruned": create_pre_var_ref(genn_dale_prune, "NumPruned")
        },
        var_refs={
            "g": weight_var_ref
        },
        egp_refs={
            "Connectivity": create_egp_ref(genn_dale_prune, "Connectivity")
        }
    )

    genn_dale_rewire = compiler.add_custom_connectivity_update(
        genn_model,
        dale_rewire,
        synapse_group,
        "DaleRewire",
        "DaleRewire_" + synapse_group.name
    )

    return genn_dale_prune, genn_dale_rewire
