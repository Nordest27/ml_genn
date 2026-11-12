# Rule forms (12 November 2026)

New ways of writing the rule, all as overlays for your working headless config (for example the drift + contrast
reward run that reached a mean of +1.66 over episodes 49-54k). Merge after your base config with
`../gpu_tests_2026-11-03/merge.py`, and check the merged `hidden_rule`: keys your base sets and an overlay does not
stay (nested dicts are merged key by key).

## The idea

Every term of the rule can be written as a TD-gated or a fixed local objective of the neuron, credited through its own
noise (Stein's lemma gives the weight derivative of the objective through the true dynamics):
- the TD error itself (`local.td`): the gradient part G on the local route;
- contrast (`local.contrast`): after a better-than-expected outcome, be far from your usual activity;
- sensitivity (`local.sensitivity`): the drift as a reward, after a better-than-expected outcome, be decisive
  (less time near threshold); r = -(delta - mean) x trace(psi - psibar);
- the terms of the moment: decorr, rate, vreg, dv.

Two channels, two baselines (`local.td_route`): the TD-dependent terms (td, contrast, sensitivity) form one reward
with one baseline, credited on the lambda-trace (the TD error judges the past window); the terms of the moment stay on
`route` (instantaneous) with their own baseline. In open-loop single-neuron tests (the drift-check replay) the
lambda-trace was the right credit for contrast (3x better selectivity than instantaneous) and the instantaneous
route the right one for decorr (reliability 0.81 vs 0.21 in L1, 0.96 vs 0.83 in L5).

Local critics (`local.critic_eta`, `critic_level`, with `td_route`): the TD channel's running-mean baseline is replaced
by a learned critic and a local TD error d = r + gamma V(t) - V(t-1). `"neuron"`: one linear critic per neuron on slow
features of its own state (bias, filtered voltage relative to threshold, filtered adaptation, spike-trace mean, mean
inhibitory input with decorr); `"synapse"`: one critic per synapse on its own presynaptic trace (weak predictor).

## What the CPU showed (small 1-layer mexhat, selection and contingency tasks; 2-4 seeds; not conclusive)

- Contrast as a reward through the noise is the strongest single piece. Computed directly with e-prop's psi and no
  noise (`hebb_gate: "psi"`), it fails with the drift (contingency +0.04 against +0.70 with the noise version).
- Estimator test (open-loop replay, 5-layer network): the reward form and the one-trace form estimate the same
  derivatives (corrected correlation 0.72-0.91); in L5 (sparse inputs) the reward form is much more reliable for
  contrast (0.90 vs 0.42).
- Drift as a reward (`drift_as_reward.json`) did worse than the current drift (alone: contingency +0.02 / -0.31 vs
  +0.24 / +0.19; selection commits on almost everything). Keep the drift as it is.
- Full two-channel reward (`two_channel.json`): selection gaps 0.26 / 0.20 vs 0.42 / 0.22 for drift + contrast reward;
  without its TD term (`two_channel_no_td.json`) 0.41 on the one finished seed: G inside the reward seems to hurt, as
  G with contrast did on your GPU. Contingency runs pending.
- Critics: pending (per-synapse: one seed better, one collapsed so far).

## Overlays

| Overlay | |
|---|---|
| `drift_plus_two_channel_contrast.json` | the drift as it is + the contrast reward on the TD channel (lambda-trace) + decorr on the instantaneous channel: your best GPU recipe written in two channels. The one I would try first |
| `two_channel.json` | everything as rewards: TD term + contrast + sensitivity (no separate drift, no G, no homeostat) |
| `two_channel_no_td.json` | (after `two_channel.json`) the same without the TD term |
| `drift_as_reward.json` | the drift replaced by the sensitivity reward (worse on the CPU) |
| `neuron_critic.json` / `synapse_critic.json` | (after a two-channel overlay) a learned critic in place of the TD channel's baseline |

The sensitivity weight 5 and the critics' rates are untuned. With your uncentred TD error, set `center_drift` as in
your base config; the CPU runs used `true`.
