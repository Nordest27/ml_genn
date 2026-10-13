# Dale's law during continuous training (locally aware rewiring)

**Without enforcement, the populations aren't E and I any more.** Dale's law was never applied during training: the rewiring only ran at batch ends, which continuous training never reaches. After 2M timesteps on the small CPU network, about a third of all learned synapses had changed sign. With plastic inhibition, about half of I -> fields had become excitatory. So "I" was partly a second excitatory pathway into the readout. The new monitor columns `flip|src->dst` show the fraction of wrong-sign synapses per connection group.

**`"dale": {"every_episodes": 1, "local": true}`** runs DalePrune / DaleRewire at episode ends:
- Each synapse whose weight crossed to the wrong sign is either reset to a tiny weight of the right sign (90%) or pruned and replaced.
- With `"local": true` the replacement is drawn around the presynaptic neuron's position on the sheet. It uses a Gaussian with the connection's own realised spread (measured from its synapses at compile time) and toroidal wrapping. So connectivity stays local and its statistics stay stationary.
- The old uniform draw scattered replacements over the whole target layer.
- Test: fewer than 5% of locally rewired synapses land outside the original neighbourhood, against ~46% with uniform rewiring.
- CPU smoke test (small network, 200 moves): wrong-sign synapses drop from 4–5% to ~1%. The rest crossed since the last episode end and are fixed at the next one.

| config | what it is |
|---|---|
| proposed_pi_dale, proposed_pi | proposed + plastic inhibition, with / without Dale |
| proposed_dale | proposed, Dale, no plastic inhibition |
| adaptive_pi_dale, adaptive_pi | adaptive e-prop + plastic inhibition, with / without Dale |
| adaptive_dale | adaptive e-prop, Dale, no plastic inhibition |
| proposed_pi_dale_uniform | as proposed_pi_dale with the old uniform rewiring (does locality matter?) |

How to read them:
- Performance holds with Dale → the plastic-inhibition results stand on real E/I populations, and I's rate and the energy accounting (`python activity_cost.py runs/...`) mean what they say.
- It drops → part of the gain came from the unconstrained sign changes (e.g. I as a second excitatory readout pathway).
- Watch `L1_I_hz` with Dale enforced: if I no longer drives the fields, its rate should no longer be "needed".

## Plastic inhibition on I -> I (`"plastic_inhibition_i"`)

Plain proposed (no controller) drives everything up: E 120 Hz, I 117 Hz, fields 87–95 Hz after 60k moves.
- Plastic inhibition on I -> E brings E to 7 Hz and the fields to ~20 Hz. That's ~3.3× fewer spikes per decision.
- But I stays at ~100 Hz in both runs: the drift pushes I's excitatory inputs up, and nothing regulates I.

`"plastic_inhibition_i": {"eta": 1e-5, "rate": 20.0}` applies the same rule to every layer's I -> I synapses, so I regulates itself.

Full-size CPU smoke test (eta 1e-4, 200 moves):

| version | E | I |
|---|---|---|
| E only | 6–8 Hz | 35 -> 48 Hz, rising |
| E + I | 13–15 Hz | ~20 Hz |

With a calmer I, the I -> E weights grow faster to hold E at 10 Hz. Watch `dW|L1_I->L1_E` and `dW|L1_I->L1_I` level off in long runs.

| config | what it is |
|---|---|
| proposed_pi_pii20, proposed_pi_pii20_dale | proposed + PI on E (10 Hz) and I (20 Hz), without / with Dale |
| adaptive_pi_pii20, adaptive_pi_pii20_dale | the same for adaptive e-prop |

If performance holds with I at 20 Hz, the total drops to ~1,050 spikes per decision, against ~10,400 for plain proposed (`activity_cost.py`).
