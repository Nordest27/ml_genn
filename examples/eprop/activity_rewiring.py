"""Host-side planning of activity-dependent rewiring (the device pass is ml_genn.compilers.activity_rewiring).

Every call (e.g. at episode ends), in every connection with an "ActRewire" pass:
1. candidates: synapses older than min_age steps whose co-activity trace (CoAct, the rule's slow mean of presynaptic
   trace x psi) is below rel_threshold x the median of the old synapses in that connection;
2. at most max_frac of the connection's synapses, the lowest co-activity first, are relocated: the postsynaptic
   neuron is kept (fan-in is preserved) and a new presynaptic neuron is placed at the offset of a randomly drawn
   existing synapse of the connection from its position on the source sheet, so the distance distribution of the
   connection is reproduced exactly (locality is preserved); existing pairs, self-connections and full rows are
   skipped, and the new partner may not be farther from the postsynaptic neuron than the synapse it replaces (plus
   half a cell), so locality cannot degrade; a synapse without such a partner within max_tries draws stays in place;
3. the chosen synapses get g = REMOVE (a sentinel) and the new ones are written per presynaptic row; one "ActRewire" pass applies it.
Everything it reads is local to a synapse (its age, co-activity and weight) or the connection's geometry.
"""
import numpy as np
from pygenn import SynapseMatrixConnectivity

from ml_genn.compilers.activity_rewiring import REMOVE
from ml_genn.compilers.dale_rewiring import _grid


class ActivityRewirer:
    def __init__(self, compiled_net, min_age=100_000, rel_threshold=0.05, max_frac=0.005, max_tries=64, seed=0,
                 log_path=None):
        self.net, self.min_age, self.rel, self.max_frac, self.max_tries = (compiled_net, min_age, rel_threshold,
                                                                           max_frac, max_tries)
        self.rng = np.random.default_rng(seed)
        self.log_path = log_path
        self.groups = []
        ccus = getattr(compiled_net.genn_model, "custom_connectivity_updates", {})
        for c, sg in compiled_net.connection_populations.items():
            ccu = ccus.get("ActRewire_" + sg.name)
            if ccu is None or not (sg.matrix_type & SynapseMatrixConnectivity.SPARSE):
                continue
            sg.pull_connectivity_from_device()
            pre, post = np.asarray(sg.get_sparse_pre_inds()), np.asarray(sg.get_sparse_post_inds())
            geo = dict(pre=_grid(c.source().shape), post=_grid(c.target().shape))
            geo["offsets"] = self._offsets(pre, post, geo)        # the connection's own pre - post offsets
            self.groups.append(dict(conn=c, sg=sg, ccu=ccu, geo=geo, recurrent=c.source() is c.target(),
                                    max_add=int(ccu.extra_global_params["AddTarget"].view.size
                                                // sg.src.num_neurons)))
        if log_path:
            with open(log_path, "w") as f:
                f.write("call,group,synapses,candidates,relocated,failed,median_coact,fanout_cv\n")
        self.calls = 0

    @staticmethod
    def _mapped(j, geo):
        """Postsynaptic neuron j's position on the source sheet (in source cells)."""
        (ph, pw, _), (qh, qw, qc) = geo["pre"], geo["post"]
        cell = j // qc
        return (cell // qw + 0.5) / qh * ph, (cell % qw + 0.5) / qw * pw

    @classmethod
    def _offsets(cls, pre, post, geo):
        """Every synapse's (row, column) offset of its presynaptic cell from its postsynaptic neuron's mapped
        position, toroidally wrapped into [-size/2, size/2)."""
        ph, pw, pc = geo["pre"]
        py, px = cls._mapped(np.asarray(post), geo)
        cell = np.asarray(pre) // pc
        dy = (cell // pw + 0.5) - py; dx = (cell % pw + 0.5) - px
        dy = (dy + ph / 2) % ph - ph / 2; dx = (dx + pw / 2) % pw - pw / 2
        return np.stack([dy, dx], 1)

    def _new_pre(self, j, geo, existing, row_len, planned, row_cap, recurrent, max_dist=np.inf):
        """A presynaptic neuron at the offset of a random existing synapse from postsynaptic neuron j, no farther
        than max_dist (source cells)."""
        ph, pw, pc = geo["pre"]
        py, px = self._mapped(j, geo)
        off = geo["offsets"]
        for _ in range(self.max_tries):
            dy, dx = off[self.rng.integers(len(off))]
            r = int(np.floor(py + dy)) % ph
            c = int(np.floor(px + dx)) % pw
            i = (r * pw + c) * pc + int(self.rng.integers(pc))
            if np.hypot(*self._offsets([i], [j], geo)[0]) > max_dist:
                continue
            if (recurrent and i == j) or (i, j) in existing:
                continue
            if row_len[i] + planned.get(i, 0) >= row_cap or planned.get(i, 0) >= self.max_add_current:
                continue
            return i
        return None

    def __call__(self):
        self.calls += 1
        rows = []
        any_change = False
        for gr in self.groups:
            sg, ccu = gr["sg"], gr["ccu"]
            sg.pull_connectivity_from_device()
            for v in ("g", "CoAct", "Age"):
                sg.vars[v].pull_from_device()
            pre, post = np.asarray(sg.get_sparse_pre_inds()), np.asarray(sg.get_sparse_post_inds())
            g = np.array(sg.vars["g"].values)
            coact, age = np.asarray(sg.vars["CoAct"].values), np.asarray(sg.vars["Age"].values)
            old = age > self.min_age
            n = len(g)
            med = float(np.median(coact[old])) if old.any() else 0.0
            cand = np.flatnonzero(old & (coact < self.rel * med)) if med > 0 else np.array([], dtype=int)
            cand = cand[np.argsort(coact[cand])][:int(self.max_frac * n)]
            existing = set(zip(pre.tolist(), post.tolist()))
            row_len = np.bincount(pre, minlength=sg.src.num_neurons)
            row_cap = sg.max_connections
            self.max_add_current = gr["max_add"]
            planned, targets, remove, failed = {}, {}, [], 0
            for s in cand:
                j = int(post[s])
                d_old = float(np.hypot(*self._offsets([pre[s]], [j], gr["geo"])[0]))
                i = self._new_pre(j, gr["geo"], existing, row_len, planned, row_cap, gr["recurrent"],
                                  max_dist=d_old + 0.5)
                if i is None:
                    failed += 1
                    continue
                remove.append(s); existing.add((i, j))
                k = planned.get(i, 0); planned[i] = k + 1
                targets.setdefault(i, []).append(j)
            if remove:
                any_change = True
                g[remove] = REMOVE
                sg.vars["g"].values = g
                sg.vars["g"].push_to_device()
                add = np.zeros(sg.src.num_neurons * gr["max_add"], dtype=np.uint32)
                num = np.zeros(sg.src.num_neurons, dtype=np.uint32)
                for i, js in targets.items():
                    add[i * gr["max_add"]:i * gr["max_add"] + len(js)] = js
                    num[i] = len(js)
                ccu.extra_global_params["AddTarget"].view[:] = add
                ccu.extra_global_params["AddTarget"].push_to_device()
                ccu.pre_vars["NumAdd"].view[:] = num
                ccu.pre_vars["NumAdd"].push_to_device()
            fan_out = row_len[row_len > 0]
            rows.append((self.calls, sg.name, n, len(cand), len(remove), failed, med,
                         float(fan_out.std() / fan_out.mean()) if len(fan_out) else 0.0))
        if any_change:
            self.net.genn_model.custom_update("ActRewire")
        if self.log_path:
            with open(self.log_path, "a") as f:
                for r in rows:
                    f.write(",".join(str(x) for x in r) + "\n")
        return rows
