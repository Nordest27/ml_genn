# Why does plastic inhibition make the 1-layer proposed rule so strong?

GPU observation (2026-10-12): 1 layer, proposed + homeostat + plastic inhibition, E at ~11 Hz, ~7 reward per episode.
With the defaults (eta 1e-5, target 10 Hz) the "11 Hz" is the plastic inhibition holding E at its target.

Run one at a time, 1–2 seeds each, in this order:

    cd examples/eprop && PYTHON=/path/to/python configs/run_configs.sh "0" configs/plastic_inh_gpu/0[1-5]*.json

Configs 01–14 use `"toroidal": "legacy"`; 15–17 vary the connectivity.
Plastic inhibition acts on every layer's I -> E synapses: `|w| += eta * x_I * (y_E - y0)` (rate mode), from spike traces only.

| explanation | test | what supports it |
|---|---|---|
| it's the plastic inhibition plus the proposed rule | 02 vs 01 | 02 ≫ 01 (reproduces your run) |
| the proposed rule isn't needed: iSTDP alone makes a good reservoir | 03 vs 12, and 03 vs 02 | 03 ≈ 02 → the readout does the work; 03 ≪ 02 → the rule matters |
| it's general, not specific to the proposed rule | 05 vs 04 | 05 ≫ 04 → helps e-prop too (architecture effect); 05 ≈ 04 → specific to perturbation learning |
| decorrelation (less redundancy) | 06 (covariance mode, no rate target) | 06 ≈ 02 → decorrelation; 06 ≈ 01 → it's the rate control |
| rate control (stable sparse E) | 07, 08 (stronger firing-rate regulariser, no plasticity) | 07/08 ≈ 02 → any rate control; ≪ 02 → it's specifically the inhibitory route |
| sensitivity | 09 (eta 1e-6), 10 (5 Hz), 11 (20 Hz) | |

Priority: 02 (second seed), 03, 05/04, 06, then 07/08 and the sweep.

Look at, per run:
- the score;
- `outputs/*_neurons.csv`: L1_E/L1_I Hz over time, dead fractions, and `dW|L1_I->L1_E` (how far the inhibition moved);
- `python probe_analyse.py runs/`: decoding and the redundancy table (mean |r|, subset curve).

My prediction:
- 03 (reservoir + plastic inhibition) clearly below 02;
- 05 also improves e-prop, but less;
- 06 sits between 01 and 02, and 07/08 below 02.

Pinning E with inhibition (shunting the shared drive) should differ from the regulariser, which only pushes each neuron's own weights.
CPU caveat: on the small 6/4 network with a 20 Hz target, plastic inhibition reduced redundancy (mean |r| 0.22 → 0.18; 16-neuron saturation 0.6 → 0.4) but cut decodable information and the score (−0.1 vs 0.0). Full size and the 10 Hz target clearly behave differently, which is worth understanding too.

## Update: mean V around -20 (and voltage regulariser around 20) in the strong run

The "voltage" column is layer-1 E's running mean membrane potential.
- At -20 with a threshold of 0.61, E sits more than 30 thresholds below firing. It still keeps 10 Hz, which means very large balanced currents.
- The synapses are current-based, so growing inhibition can push V down without limit. Real inhibition stops at its reversal potential.
- This matters for the proposed rule: ψ is only non-zero near threshold, so the hidden rule may be almost switched off. Config 03 (reservoir + PI) tests exactly this.

New tools:
- The monitor now logs, per population: `v_mean`, `v_p10` and `v_p90` (percentiles of the per-neuron mean V), and `hyper` (the fraction of neurons with mean V below -5 thresholds). They show whether the network is uniformly hyperpolarised or bimodal.
- `"w_max_rel"` caps |w| of the plastic I -> E synapses at a multiple of their initial mean.
  - 13: cap at 2× the initial mean.
  - 14: cap at 5× the initial mean.
  - If 13/14 keep the score with V in a normal range, the extreme regime isn't needed.
- `dW|L1_I->L1_E` in the monitor shows how far the inhibition grew (a relative norm change).

The small CPU network does not show this regime: its voltage loss is 0.2–0.3 with plastic inhibition.

## Update: plastic inhibition works only with legacy / matched connectivity

Both use the legacy weight scale, 1/sqrt(300), while realising only 45–147 inputs. Their synapses are therefore ~2.3× weaker than "local"'s, and that has two effects:
- the weight noise (SynSig ≈ 0.0067) is as large as the weights themselves;
- the same η moves the small I -> E weights ~3× faster in relative terms.

They also project inhibition into the fields.

| config | what it is |
|---|---|
| 15 | local pattern at the legacy weight scale (`weight_fan_in: 300`) + PI |
| 16 | 15 + inhibitory feedforward into the fields, field gain 0.1 (as legacy-like as the local rule gets) |
| 17 | matched + PI, as a reference |

How to read them:
- 15 works → it's the weight scale (noise/weight ratio, relative plasticity speed).
- Only 16 works → it's the inhibitory input to the fields.
- Neither works → it's the legacy sampling pattern itself.

Controls 01–14 now use `"toroidal": "legacy"`, since that is where the effect exists.

## Update: conductance-like inhibition (`"inhibitory_reversal"`)

Inhibitory synapses act through the driving force: each step, V relaxes towards E_inh by (E_inh − V)(1 − exp(−G)), with G = I_inh / E_inh.
- For small inputs this equals I_inh · (V − E_inh) / −E_inh: unchanged at rest, vanishing at E_inh.
- V can never be pushed below E_inh. Inhibition becomes shunting instead of unlimited subtraction.
- This is the conductance-based setting of the original iSTDP model (Vogels et al. 2011). There, E_inh sits 20 mV below rest with the threshold 10 mV above, i.e. E_inh ≈ −2 thresholds = −1.2 here.

| config | what it is |
|---|---|
| 18 | 02 + E_inh = −1.2 (2 thresholds below rest) |
| 19 | 02 + E_inh = −2.4 (4 thresholds) |
| 20 | 01 + E_inh = −1.2, without plastic inhibition (does the reversal alone change anything?) |

Compare with 02 (your strong run). Check the voltage columns: `v_mean`, `v_p10` and `hyper` in `*_neurons.csv`, and voltage / voltage_loss in the episode CSV.
- 18/19 keep the score → the benefit doesn't need the deep regime.
- 18 loses it but 19 keeps it → it needs some depth, but not unlimited.
- Both lose it → the effect relies on unbounded inhibition (your effective-threshold reading).

Expected: both stay well above 01. If plastic inhibition must hold E at 10 Hz with bounded inhibition, the weights may keep growing; in that case combine with `w_max_rel` (13/14).
