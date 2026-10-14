# Local decorrelation reward

`hidden_rule.local.decorr = c` adds a local reward for every neuron:

    r_j = -c (z_j - zbar_j) (I_inh,j - Ibar_inh,j) / Ibar_inh,j

z_j is the neuron's spiking and I_inh,j the total inhibitory input it receives, with running means over ~1 s.
- It penalises spiking in step with the received inhibition, i.e. redundancy with the cells that drive that inhibition.
- Every synapse of the rule learns from it through the local policy-gradient route: it correlates its own noise with the neuron's reward minus a running baseline.
- It is local: the neuron's own spikes and inhibitory input.
- All inhibitory connections go to an `ISynInh` input channel, added to the membrane exactly as before.

**CPU result** (small network, 1 layer, local Dale, 2 seeds), strength 1 against the base:
- score −0.032/−0.009 vs −0.048/−0.026;
- decoding +0.31/+0.34 vs +0.23/+0.31;
- slightly lower neuron-to-neuron correlation.

Promising but within seed noise, hence these configs: more seeds, a strength sweep, and depth.

| config | question |
|---|---|
| d1_proposed, d1_proposed_decorr1 / 3 / 10 | is the gain real, and how strong should the reward be? |
| d1_proposed_decorr1_dale | does it hold with real E/I populations (local Dale rewiring)? |
| d5_homeo, d5_homeo_decorr1 / 3 | at 5 layers (proposed + homeostat, your best): does it help information survive the layers? |

Run at least 4 seeds per config for the 1-layer set, e.g.

    PYTHON=... configs/run_configs.sh "0 1 2 3" configs/decorr_reward_gpu/d1_*.json

All configs use legacy connectivity, as in your runs. If the full-size architecture search finds a better starting network, a variant of the 5-layer configs will follow with that architecture.

What to look at:
- the score;
- `python probe_analyse.py runs/` (redundancy table: mean |r|, saturation; decoding; effective dimension first -> last);
- for 5 layers, `python layer_dims.py runs/<run>` (dimension and decoding per layer).
