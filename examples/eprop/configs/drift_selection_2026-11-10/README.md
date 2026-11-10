# Drift-as-selection overlays (10 November 2026)

Small JSON overlays to add to your working headless config (the 1-layer mexhat Snake run with `"gradient": 0`,
`"center_drift": true`, λ 0.5, γ 0.5). Merge with `../gpu_tests_2026-11-03/merge.py`:

    python ../gpu_tests_2026-11-03/merge.py your_headless.json lse_decoder.json -o lse.json
    SNAKE_CONFIG=lse.json python run_headless.py

## Why

On the CPU (small mexhat, headless selection: commit or skip, good iff features 0 and 1, random −0.35, oracle 0.91):

- The drift freezes per state. A positive TD error lowers the drive of most neurons in that state (near-threshold
  neurons most) and spares the most active ones. The pattern sharpens and becomes more deterministic. A negative
  TD error sensitises.
- The pool-mean decoder works against this. Freezing narrowed the mean margin between the pools in 12 of 16
  states. Decoders that read each pool's most active neurons agree with it (pool max 81%, top-5 69%).
- With the log-sum-exp decoder, 5 of 7 runs became selective in the right direction. With pool means, 1 of 2
  learned backwards.
- Some seeds lock into one global bias (always skip). Centring the drift eligibility per synapse removed the
  lock-in on the seed that always failed. The homeostat lowered the commits on bad candidates (best run: good
  0.92 / bad 0.34, score −0.07).
- Decorrelation (decorr 1, from the mexhat config) was on in all of these runs. An ablation without it is
  running on the CPU.

The same mechanisms (the decoder disagreeing with the freeze, the shared-bias lock-in) should act in Snake.

## Overlays

| Overlay | What it changes |
|---|---|
| `lse_decoder.json` | headless decoder `lse`, β 1: each motor pool's log-mean-exp of its neurons' spike counts over the move, instead of the pool mean |
| `homeostat.json` | `homeostat` 1 (drift channel, target ψ 0.1) |
| `center_eligibility.json` | `center_eligibility` 0.9999 (+ `center_drift`): each synapse subtracts the running mean of its drift-eligibility increment, removing the part shared by all states |
| `motor_l1.json` | `"fields": null`: the motor pools are the last hidden layer's E neurons, and the value head reads that layer. Not combinable with `motor_log_syn_sigma` / `motor_local` / `motor_optimiser` (they act on the policy field) |
| `selection.json` | the selection task instead of Snake |
| `reach_popvec.json` | continuous reaching (point agent to target, egocentric frame) with the population-vector decoder |
| `move_log.json` | per-decision CSV `<prefix>(<rep>)_moves.csv` (selection / reach only) and drift probes (+ `diagnose_parts`) |

## Suggested runs (one at a time)

1. your headless Snake config + `lse_decoder.json` (the decoder alone: the cleanest test)
2. + `homeostat.json`
3. + `center_eligibility.json`
4. optionally + `motor_l1.json` (the hidden layer as the motor population: its drift stays state-specific, while the
   drift into the fields had no per-state signal at small size)

Check `_policy.csv` (pool shares) for a fixed action bias across states (the Snake version of the lock-in).

## New options (all in `snake.py` / `snake_network.py` headers)

- `headless.decoder`: `max`, `topk` (`k`), `lse` (`beta`), `popvec` (`gain`). The motor neurons' spike counts are
  read on the host once per move (one small device-to-host transfer per decision).
- `hidden_rule.center_eligibility`: see above (`variants.py`, `hidden_rule.py`).
- `"network": {"fields": null}` with headless: the last layer's E neurons are the motor pools.
- `boolean_head` with `"position": "before_fields"` is now allowed with headless.
- `field_ei`: each field becomes an E/I layer with its own inhibitory population (`gain_ee`, `gain_ei`, `gain_i`).
- `reach`: `snake_switch.ReachEnv` (`speed`, `radius`, `progress`, `reach_reward`, `max_moves`, `near`, `noise`,
  `blob`, `walls`, `seed`).
- `selection`: `switch_moves` / `required_after` (the required features change every n decisions); `ctype`.
- `move_log`: `probe_every`, `probe_moves`, `aggregate`, `include`, `activity` (see `snake.py`). Rule variable
  `AccPx` (with `diagnose_parts`): the drift eligibility without the TD error.
