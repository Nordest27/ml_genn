"""Is the expected drift of the perturbation e-prop rule the gradient of some function?

A smooth vector field F(w) is the gradient of a scalar function (on a simply connected region) exactly when its
Jacobian J_ab = dF_a/dw_b is symmetric (equality of mixed partials; Poincare lemma). A significantly asymmetric
Jacobian at one point rules out every scalar objective in these coordinates. A positive per-synapse learning rate
d_a (any diagonal metric) only rescales rows, F = D grad f, which keeps J_ab / d_a = J_ba / d_b: entries J_ab and J_ba
of opposite sign rule out every such objective as well.

Model: a tiny recurrent ALIF network with the Snake neuron settings and Poisson inputs. The rule's drift is
    F_a = E[ xi_a * e_a(w + xi) ] / sigma^2 = d_a E[e_a],   e_a = sum_t (Psi_post(t) - PsiBar_post) * pre_a(t),
    pre_a = ZFilter - Beta * epsA,
with static Gaussian weight noise xi per sample (the Gaussian-smoothing view). The common TD-error factor is left
out (a positive constant does not change symmetry).

Its Jacobian is J_ab = d_b d_a E[e_a], so J_ab - J_ba = d_a d_b E[e_a - e_b]: a gradient needs every synapse's
eligibility to share mixed partials with every other's. By the second-order Stein identity,
E[xi_a xi_b f(w + xi)] = sigma^4 d_a d_b E[f] for a != b, so the asymmetry is estimated directly from one set of
simulations, with i.i.d. per-sample terms and valid error bars. Antithetic pairs (+xi, -xi) and the noise-free run
(same input spikes) reduce the variance.

Two versions: PsiBar held fixed at its base-point value ("fixed"), and PsiBar tracking the weights ("track"; the
rule's PsiBar is the neuron's running mean at the current weights and is not differentiated by the noise), which
adds -dPsiBar_post(a)/dw_b * d_a E[sum_t pre_a]. A null estimate with the noise shuffled across samples (expectation
exactly zero) checks the error bars.
"""
import argparse
import numpy as np

VTH, TAU_MEM, TAU_REFRAC, TAU_ADAPT, BETA = 0.61, 10.0, 3.0, 300.0, 0.0174
ALPHA, RHO = np.exp(-1.0 / TAU_MEM), np.exp(-1.0 / TAU_ADAPT)


def simulate(w, xi, inputs, n_in, n_hid):
    """w: (P,) weights; xi: (N, P) noise; inputs: (T, N, n_in) spikes. Returns sum_t Psi*pre and sum_t pre per
    synapse (N, P), the mean Psi per neuron (N, n_hid) and the firing rate per neuron (N, n_hid), per sample."""
    n, p = xi.shape
    we = w[None, :] + xi
    off = ~np.eye(n_hid, dtype=bool)
    w_in = we[:, :n_in * n_hid].reshape(n, n_hid, n_in)
    w_rec = np.zeros((n, n_hid, n_hid))
    w_rec[:, off] = we[:, n_in * n_hid:]
    post = np.concatenate([np.repeat(np.arange(n_hid), n_in), np.nonzero(off)[0]])
    rec_pre = np.nonzero(off)[1]
    v = np.zeros((n, n_hid)); a = np.zeros((n, n_hid)); refrac = np.zeros((n, n_hid)); z = np.zeros((n, n_hid))
    zf_in = np.zeros((n, n_in)); zf_hid = np.zeros((n, n_hid))
    eps_a = np.zeros((n, p)); s_psi_pre = np.zeros((n, p)); s_pre = np.zeros((n, p))
    psi_sum = np.zeros((n, n_hid)); spikes = np.zeros((n, n_hid))
    for t in range(inputs.shape[0]):
        x = inputs[t]
        v = ALPHA * v + np.einsum("nhi,ni->nh", w_in, x) + np.einsum("nhj,nj->nh", w_rec, z)
        a = RHO * a
        refrac = np.maximum(refrac - 1.0, 0.0)
        z = ((v >= VTH + BETA * a) & (refrac <= 0.0)).astype(float)
        v = v - VTH * z; a = a + z; refrac = np.where(z > 0, TAU_REFRAC, refrac)
        psi = np.where(refrac > 0, 0.0,
                       (1.0 / VTH) * 0.3 * np.maximum(0.0, 1.0 - np.abs((v - (VTH + BETA * a)) / VTH)))
        zf_in = ALPHA * zf_in + x
        zf_hid = ALPHA * zf_hid + z
        zf = np.concatenate([np.tile(zf_in, (1, n_hid)), zf_hid[:, rec_pre]], axis=1)
        psi_s = psi[:, post]
        pre = zf - BETA * eps_a
        s_psi_pre += psi_s * pre
        s_pre += pre
        eps_a = psi_s * zf + RHO * eps_a - psi_s * BETA * eps_a
        psi_sum += psi
        spikes += z
    steps = inputs.shape[0]
    return s_psi_pre, s_pre, psi_sum / steps, spikes * 1000.0 / steps


def chunk_estimates(w, seed, args, psibar0):
    """Estimates from one chunk of noise samples: J_ab (fixed and tracking PsiBar) and a shuffled-noise null."""
    rng = np.random.default_rng(seed)
    n, p = args.samples, w.size
    inputs = (rng.random((args.steps, n, args.n_in)) < args.rate / 1000.0).astype(float)
    xi = rng.normal(0.0, args.sigma, (n, p))
    post = np.concatenate([np.repeat(np.arange(args.n_hid), args.n_in),
                           np.nonzero(~np.eye(args.n_hid, dtype=bool))[0]])
    runs = [simulate(w, s * xi, inputs, args.n_in, args.n_hid) for s in (1.0, -1.0, 0.0)]
    e = [sp - psibar0[post] * s for sp, s, _, _ in runs]
    even = 0.5 * (e[0] + e[1]) - e[2]                         # even in xi; noise-free run as control variate
    s2, s4 = args.sigma ** 2, args.sigma ** 4

    def second(noise):                                         # d_b d_a E[e_a] estimated as a (P, P) matrix
        return np.einsum("na,nb,na->ab", noise, noise, even) / (n * s4)

    hess_own = second(xi)                                      # H_ab = d_a d_b E[e_a]  ->  J_fixed_ab = H_ab
    null = second(np.roll(xi, 1, axis=0))                      # noise unrelated to the run: expectation 0
    # tracking correction: -dPsiBar_post(a)/dw_b * d_a E[sum pre_a]
    d_pre = (xi * 0.5 * (runs[0][1] - runs[1][1])).mean(0) / s2                  # (P,)
    d_psibar = np.einsum("nk,nb->kb", 0.5 * (runs[0][2] - runs[1][2]), xi) / (n * s2)   # (n_hid, P)
    track = hess_own - d_psibar[post, :] * d_pre[:, None]
    drift = (xi * 0.5 * (e[0] - e[1])).mean(0) / s2
    return dict(fixed=hess_own, track=track, null=null, drift=drift,
                rate=runs[2][3].mean(0), psibar=runs[2][2].mean(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-in", type=int, default=4)
    ap.add_argument("--n-hid", type=int, default=3)
    ap.add_argument("--rate", type=float, default=40.0, help="input Poisson rate (Hz)")
    ap.add_argument("--steps", type=int, default=500, help="timesteps (ms) per sample")
    ap.add_argument("--samples", type=int, default=5000, help="noise samples per chunk")
    ap.add_argument("--chunks", type=int, default=20, help="independent chunks (error bars from their spread)")
    ap.add_argument("--sigma", type=float, default=0.05, help="weight-noise s.d.")
    ap.add_argument("--w-in", type=float, default=0.25, help="mean input weight")
    ap.add_argument("--w-rec", type=float, default=0.1, help="recurrent weight scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="drift_curl.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_rec = args.n_hid * (args.n_hid - 1)
    w = np.concatenate([args.w_in * (1.0 + 0.5 * rng.standard_normal(args.n_in * args.n_hid)),
                        args.w_rec * rng.standard_normal(n_rec)])
    p = w.size
    # PsiBar at the base point (noise-free run, separate inputs)
    base_inputs = (np.random.default_rng(999).random((args.steps, 2000, args.n_in)) < args.rate / 1000.0)
    _, _, psibar0, rate0 = simulate(w, np.zeros((2000, p)), base_inputs.astype(float), args.n_in, args.n_hid)
    psibar0 = psibar0.mean(0)
    print(f"{p} weights ({args.n_in * args.n_hid} input, {n_rec} recurrent); rates {np.round(rate0.mean(0), 1)} Hz;"
          f" PsiBar {np.round(psibar0, 4)}; {args.chunks} x {args.samples} samples of {args.steps} ms", flush=True)

    est = {k: [] for k in ("fixed", "track", "null", "drift")}
    for c in range(args.chunks):
        r = chunk_estimates(w, 1000 + c, args, psibar0)
        for k in est:
            est[k].append(r[k])
        if (c + 1) % 5 == 0:
            print(f"  chunk {c + 1}/{args.chunks}", flush=True)
    est = {k: np.array(v) for k, v in est.items()}
    k = args.chunks
    drift_z = est["drift"].mean(0) / (est["drift"].std(0, ddof=1) / np.sqrt(k))
    print(f"\ndrift itself: {np.sum(np.abs(drift_z) > 3)}/{p} components with |z| > 3")

    iu = np.triu_indices(p, 1)
    print("\nversion   |J-J^T|/|J+J^T|   pairs with |z(J_ab-J_ba)|>3   opposite-sign pairs (|z|>3 both)   median |z| asym")
    for name in ("null", "fixed", "track"):
        jac = est[name]                                            # (K, P, P)
        anti = jac - jac.transpose(0, 2, 1)
        anti_z = anti.mean(0) / (anti.std(0, ddof=1) / np.sqrt(k) + 1e-30)
        jz = jac.mean(0) / (jac.std(0, ddof=1) / np.sqrt(k) + 1e-30)
        m = jac.mean(0)
        ratio = np.linalg.norm(m - m.T) / (np.linalg.norm(m + m.T) + 1e-30)
        sig = np.abs(anti_z[iu]) > 3
        opp = (np.abs(jz[iu]) > 3) & (np.abs(jz.T[iu]) > 3) & (np.sign(m[iu]) != np.sign(m.T[iu]))
        print(f"{name:8s}  {ratio:14.3f}   {sig.sum():8d} / {len(iu[0])}"
              f"{opp.sum():30d}{np.median(np.abs(anti_z[iu])):30.2f}")
    np.savez(args.out, w=w, psibar0=psibar0, **est)


if __name__ == "__main__":
    main()
