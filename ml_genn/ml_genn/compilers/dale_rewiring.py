import numpy as np

from pygenn import VarAccessMode
from pygenn import create_egp_ref, create_pre_var_ref

from ..callbacks import Callback
from ..utils.model import CustomConnectivityUpdateModel, CustomUpdateModel

# ================================================================
# L1 regularisation model
# ================================================================
dale_r_l1_model = {
    "params": [("alpha", "scalar")],
    "var_refs": [("g", "scalar")],
    "update_code": """
        if (g > 0) g -= alpha;
        else g += alpha;
    """}

# ================================================================
# Dale-Constrained Rewiring models
# ================================================================

# ------------------------------------------------
# 0. Initialisation — builds Connectivity bitmask
#    from the current sparse structure, mirroring
#    deep_r_init_model from Deep-R.
# ------------------------------------------------
dale_init_model = {
    "params": [("NumRowWords", "unsigned int"), ("AllowRecurrent", "int")],
    "var_refs": [("g", "scalar", VarAccessMode.READ_ONLY)],
    "extra_global_param_refs": [("Connectivity", "uint32_t*")],
    "row_update_code": """
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);

    // Zero this row's connectivity
    for (int i = 0; i < NumRowWords; i++) {
        rowConnectivity[i] = 0;
    }
    // If recurrent, mark self-connection as occupied so it is never rewired to
    if (id_pre < num_post && !AllowRecurrent) {
        rowConnectivity[id_pre / 32] |= (1 << (id_pre % 32));
    }
    
    // Set a bit for every existing synapse
    for_each_synapse {
        rowConnectivity[id_post / 32] |= (1 << (id_post % 32));
    }
    """}

# ------------------------------------------------
# 1. Prune synapses that violate Dale's law.
#    - Removes synapse and clears bitmask bit.
#    - Counts pruned synapses per row in NumPruned.
# ------------------------------------------------
dale_prune_model = {
    "params": [
        ("NumRowWords", "unsigned int"),
        ("Sign", "int"),
        ("Resistance", "scalar")
    ],
    "var_refs": [
        ("g", "scalar")
    ],
    "pre_var_refs": [
        ("NumPruned", "unsigned int")
    ],
    "extra_global_param_refs": [
        ("Connectivity", "uint32_t*"),
    ],
    "row_update_code": """
    NumPruned = 0;
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);

    for_each_synapse {
        const bool prune =
            (Sign > 0 && g < 0.0) ||
            (Sign < 0 && g > 0.0);

        if (prune) {
            if ( gennrand_uniform() < Resistance ){
                g = Sign * 0.0001;
            }
            else {
                rowConnectivity[id_post / 32] &= ~(1 << (id_post % 32));
                remove_synapse();
                NumPruned++;
            }
        }
    }
    """
}


dale_rewire_model = {
    "params": [
        ("NumRowWords", "unsigned int"),
        ("Sign", "int"),
    ],
    "pre_vars": [
        ("NumPruned", "unsigned int")
    ],
    "var_refs": [
        ("g", "scalar")
    ],
    "extra_global_params": [
        ("Connectivity", "uint32_t*"),
    ],
    "row_update_code": """
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);

    while (NumPruned > 0) {
        const unsigned int j = gennrand() % num_post;

        // Skip if already connected
        if (rowConnectivity[j / 32] & (1 << (j % 32))) {
            continue;
        }

        add_synapse(j, Sign * 0.0001);
        rowConnectivity[j / 32] |= (1 << (j % 32));
        NumPruned--;
    }
    """
}

# ================================================================
# Optional callback to record rewiring statistics
# ================================================================
class DaleRewiringRecord(Callback):
    """
    Records total rewirings and failed rewirings per batch.
    """
    def __init__(self, rewire_ccu, key=None):
        self.num_rewirings = rewire_ccu.extra_global_params["NumRewirings"]
        self.num_failed_rewirings = rewire_ccu.extra_global_params["NumFailedRewirings"]
        self.key = key

    def set_params(self, data, **kwargs):
        data[self.key] = []
        self._data = data[self.key]

    def on_batch_end(self, batch, metrics):
        self._data.append((
            int(self.num_rewirings.view[0]),
            int(self.num_failed_rewirings.view[0])
        ))

    def get_data(self):
        return self.key, self._data


# ================================================================
# Public API: add_dale_rewiring
# ================================================================
def add_dale_rewiring(
    synapse_group,
    genn_model,
    compiler,
    l1_strength,
    sign,
    weight_var_ref,
    allow_self_connections=False
):
    """
    Adds Dale-Constrained Rewiring (DCR) to a synapse group,
    biasing new connections toward under-connected post neurons
    to preserve fan-in balance at the population level.

    Three passes per update:

      DaleInit (device, run once at start):
        Builds the Connectivity bitmask from the existing sparse
        structure using for_each_synapse, mirroring deep_r_init.

      DalePrune (device):
        Removes synapses whose weight sign violates Dale's law,
        counts them per row in NumPruned, clears bitmask bits.

      DaleRewire (host + device):
        Host pulls NumPruned, row_length, and sparse_post_inds.
        Computes PostFanIn via bincount on sparse_post_inds, then
        samples post neurons weighted inversely by fan-in so that
        under-connected neurons are preferentially targeted.
        Assigns NumActivations per row. Device rejection-samples
        unoccupied post neurons per activation slot (Deep-R style).

    Parameters
    ----------
    synapse_group :
        GeNN synapse population.
    genn_model :
        GeNN model.
    compiler :
        Compiler / builder object.
    l1_strength : float
        L1 regularisation coefficient (0.0 to disable).
    sign : int
        +1 for excitatory, -1 for inhibitory.
    weight_var_ref :
        Var reference to synaptic weight.

    Returns
    -------
    (genn_dale_prune, genn_dale_rewire)
        CustomConnectivityUpdate objects for both passes.
    """
    is_recurrent = synapse_group.src == synapse_group.trg
    num_row_words = (synapse_group.trg.num_neurons + 31) // 32
    num_words = synapse_group.src.num_neurons * num_row_words

    # -------------------------------------------------------
    # DaleRewire — created first as it owns Connectivity,
    # NumPruned and NumActivations (mirrors Deep-R pass 2)
    # -------------------------------------------------------
    dale_rewire = CustomConnectivityUpdateModel(
        dale_rewire_model,
        param_vals={
            "NumRowWords": num_row_words,
            "Sign": sign,
        },
        pre_var_vals={
            "NumPruned": 0,
        },
        var_refs={"g": weight_var_ref},
        egp_vals={
            "Connectivity": np.zeros(num_words, dtype=np.uint32),
        }
    )

    genn_dale_rewire = compiler.add_custom_connectivity_update(
        genn_model, dale_rewire, synapse_group,
        "DaleRewire", "DaleRewire_" + synapse_group.name)

    # -------------------------------------------------------
    # DalePrune — references NumPruned and Connectivity
    # owned by DaleRewire
    # -------------------------------------------------------
    dale_prune = CustomConnectivityUpdateModel(
        dale_prune_model,
        param_vals={
            "NumRowWords": num_row_words,
            "Sign": sign,
            "Resistance": 0.90
        },
        pre_var_refs={
            "NumPruned": create_pre_var_ref(genn_dale_rewire, "NumPruned")
        },
        var_refs={"g": weight_var_ref},
        egp_refs={
            "Connectivity": create_egp_ref(genn_dale_rewire, "Connectivity"),
        }
    )

    genn_dale_prune = compiler.add_custom_connectivity_update(
        genn_model, dale_prune, synapse_group,
        "DalePrune", "DalePrune_" + synapse_group.name)

    # -------------------------------------------------------
    # DaleInit — builds Connectivity bitmask from scratch,
    # must be run once before training begins
    # -------------------------------------------------------
    dale_init = CustomConnectivityUpdateModel(
        dale_init_model,
        param_vals={"NumRowWords": num_row_words, 
                    "AllowRecurrent": not is_recurrent or allow_self_connections },
        var_refs={"g": weight_var_ref},
        egp_refs={
            "Connectivity": create_egp_ref(genn_dale_rewire, "Connectivity"),
        }
    )

    compiler.add_custom_connectivity_update(
        genn_model, dale_init, synapse_group,
        "DaleInit", "DaleInit_" + synapse_group.name)

    # -------------------------------------------------------
    # Optional L1 regularisation
    # -------------------------------------------------------
    if l1_strength > 0.0:
        dale_l1 = CustomUpdateModel(
            dale_r_l1_model,
            param_vals={"alpha": l1_strength},
            var_refs={"g": weight_var_ref}
        )
        compiler.add_custom_update(
            genn_model, dale_l1,
            "DaleRL1", "DaleRL1_" + synapse_group.name)

    return genn_dale_prune, genn_dale_rewire