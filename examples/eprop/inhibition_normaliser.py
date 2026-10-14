"""Heterosynaptic normalisation of plastic inhibitory synapses (PlasticInhibitionConfig.normalise).

Every `every` environment moves, each postsynaptic neuron's incoming weights of a normalised connection are rescaled
so that their summed magnitude returns to its initial value. The plasticity rule then only redistributes inhibition
between presynaptic I cells (towards the ones that co-vary with the neuron, in covariance mode); it cannot raise or
lower the neuron's total inhibition. Local: a neuron only uses its own incoming synapses. Runs on the host, because
GeNN's custom connectivity updates iterate over presynaptic rows while this needs postsynaptic sums.
"""
import numpy as np


def normalise_columns(g, post_ind, target, eps=1e-12):
    """Rescale g so that the summed |g| of each postsynaptic neuron equals target[post] (neurons whose current sum is
    ~0 are left alone). Returns the new weights."""
    current = np.bincount(post_ind, weights=np.abs(g), minlength=len(target))
    scale = np.where(current > eps, target / np.maximum(current, eps), 1.0)
    return g * scale[post_ind]


class InhibitionNormaliser:
    def __init__(self, compiled_net, connections, every):
        """connections: the ml_genn Connections to normalise; every: environment moves between normalisations."""
        self.every, self.moves, self.groups = int(every), 0, []
        for c in connections:
            sg = compiled_net.connection_populations[c]
            sg.pull_connectivity_from_device()
            post = np.asarray(sg.get_sparse_post_inds(), dtype=np.int64)
            sg.vars["g"].pull_from_device()
            g = np.asarray(sg.vars["g"].values, dtype=np.float64)
            n_post = int(np.prod(c.target().shape))
            self.groups.append((sg, post, np.bincount(post, weights=np.abs(g), minlength=n_post)))

    def move(self):
        """Call once per environment move."""
        self.moves += 1
        if self.moves % self.every:
            return
        for sg, post, target in self.groups:
            sg.vars["g"].pull_from_device()
            g = np.asarray(sg.vars["g"].values, dtype=np.float64)
            sg.vars["g"].values = normalise_columns(g, post, target).astype(sg.vars["g"].values.dtype)
            sg.vars["g"].push_to_device()
