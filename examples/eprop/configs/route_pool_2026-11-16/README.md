# Headless generalised e-prop: pooled motor credit routed back (16 November 2026)

This is e-prop without a policy head. The motor pools vote; there is no learned per-motor-neuron readout, so the
hidden layers have to encode the actions themselves.

How credit flows:

1. **Motor pools.** Each motor neuron credits its pool's mean noise sample instead of its own (`pool_credit`). Pool
   homeostasis moves each pool's bias toward its fair share of the decisions (`pool_homeo`). The overlay also turns on
   the pools' gradient term (`motor_rule: {"gradient": 1.0}`), so they learn from that pooled noise.
2. **Route links.** Every hidden neuron learns links back along each of its forward connections, without reward. Each
   link regresses the target's input on the hidden neuron's own spikes, so it becomes the per-spike weight it mirrors.
   This is feedback alignment with learned feedback.
3. **Hidden credit.** Each step, the motor neurons send their pool's noise back through the links. Hidden neurons
   relay what they receive further upstream (`route_msg_own: false`: they are relays, not sources), gated by whether
   they fired recently (`route_gate: "activity"`), at the scale of their own perturbation credit (`route_trust:
   "scale"`). Each synapse then learns from TD error × trace(gate × routed credit × presynaptic activity).

Everything is local: the pools' relay synapses, the two ends of each link, the neuron's own spikes, and the broadcast
TD error.

Toy evidence (rate and spiking pick task, flexible choice among 2-6 items, no columns):

- One hidden layer: 0.715 against 0.575 frozen. The hidden neurons' own-noise credit reached at best 0.62, even with
  learned noise regions and a half-window shadow.
- Two hidden layers: 0.63 against 0.50 frozen and 0.525 with own-noise credit.
- With routing, layer-1 credit adds +0.08.
- The old `route_spike.json` (raw noise, psi gate, learned trust) did not help on Snake. The toys needed pooled motor
  credit, an activity gate and fixed-scale trust.

| Overlay | |
|---|---|
| `route_pool.json` | pool credit + pool homeostasis + routed credit, on top of the base rule (drift, contrast etc. stay) |
| `route_pool_only.json` | (after `route_pool.json`) hidden learning from routed credit only: drift, gradient and contrast off, decorr 0.001 kept |
| `pool_only.json` | pool credit + pool homeostasis without routing (attribution) |

Merge onto your headless config, keeping its network:

    python ../gpu_tests_2026-11-03/merge.py narrow_5l_headless.json route_pool.json -o narrow_5l_route_pool.json
    python ../gpu_tests_2026-11-03/merge.py narrow_5l_headless.json route_pool.json route_pool_only.json -o narrow_5l_route_only.json
    python ../gpu_tests_2026-11-03/merge.py narrow_5l_headless.json pool_only.json -o narrow_5l_pool_only.json

Works with or without `network.fields`. Without fields, the motor pools are the last layer's E neurons, which now get
the motor rule.

Suggested order, one at a time (same seed as the base run):

1. `route_pool`
2. `pool_only`
3. `route_only`

What to look at:

- the reward and value curves against the base run;
- the firing rates per layer in `_neurons.csv`. Routed credit raised hidden rates in the toys (to about 145 Hz in the
  last hidden layer), so watch for that. The links do not enter the membrane, so any rise comes from learning;
- the decision shares per pool (homeostasis should keep them near 1/4).
