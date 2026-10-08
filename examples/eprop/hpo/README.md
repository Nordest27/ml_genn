# Hyperparameter search for the Snake variants

## Strategy

**What gets tuned:** a few knobs per rule, not everything.
- Architecture, task, γ/λ and time constants stay at their hand-tuned values.
- Each main-comparison rule gets the **same budget** over its learning rate and its own noise scale (or, for e-prop, `c_reg`).
- Additions to the rule (homeostat, local objectives) start from the proposed rule's tuned values and tune only their own coefficients.
- Ablations (`gradient_only`, `drift_only`, `unbiased`, …) inherit the tuned values and re-tune only the learning rate. That way each one still changes a single thing.

**How the search works:** successive halving on short runs.
- Start with 16 configurations. Configuration 0 is always the hand-tuned (or inherited) point, so the search cannot do worse than hand tuning, up to seed noise. The other 15 come from a scrambled Sobol sequence, in log scale for rates and noise.
- Rungs: 1M → 3M → 9M simulation timesteps, with 1 → 2 → 2 seeds. The top third is kept at each rung.
- Seeds are shared across configurations, so they are compared on the same random numbers.
- A run is scored as reward per environment step over the last third of its timesteps.

**Trust the proxy only after checking it.** Before the first search, run `proxy` on your existing full-length CSVs. It shows whether the 1M/3M/9M score ranks runs the same way the 30M score does. If the rank correlation at 1M is low, drop the first rung (`rungs: [3e6, 9e6]`).

**Report fresh seeds.** The winner's tuning score is optimistically biased, because it is the maximum over noisy runs. The paper numbers come from `evaluate`, which uses new seeds (100+) at full length.

## Order (≈ 48 h budget)

```bash
cd examples/eprop
python hpo/sweep.py calibrate --jobs 1          # ~10 min; then try --jobs 2 if memory allows
python hpo/sweep.py proxy ~/old_runs/*.csv      # does 1M predict 30M?
python hpo/sweep.py plan                        # hours per study at the measured speed
python hpo/sweep.py run proposed adaptive_eprop baseline_120hz      # priority 1: the main claim
python hpo/sweep.py run node_proposed ind_noise_proposed proposed_homeostat
python hpo/sweep.py run gradient_only gradient_only_homeostat drift_only unbiased proposed_full
python hpo/sweep.py report proposed
python hpo/sweep.py evaluate proposed --seeds 100-104               # paper runs, full length
```

- All runs at once cost about 530M timesteps, roughly 18 full-length runs. If `plan` says this does not fit, lower `n` to 8 or drop the 9M rung.
- `run` can be resumed: after a crash or Ctrl-C, run the same command again and it skips the runs that already finished.
- Runs are sequential by default (`--jobs 1`). Raise `--jobs` only after checking memory with `calibrate --jobs 2`.

## Using a result

Every run folder has a complete `config.json`, and every study writes its winner to `best.json`:

```bash
SNAKE_CONFIG=hpo/studies/proposed/best.json python run_headless.py   # or python snake.py (with plots)
```

- To tweak a configuration by hand, copy the json and edit it. Only the keys you want to change need to be present (see `snake_hparams.py`).
- Without `SNAKE_CONFIG`, `snake.py` behaves as before.

## Files
- `sweep.py`: the tool (`plan`, `calibrate`, `proxy`, `run`, `report`, `evaluate`).
- `studies.py`: search spaces and schedules. Edit these.
- `studies/<study>/`:
  - `points.json`: the sampled configurations
  - `results.jsonl`: one line per run
  - `runs/<key>/`: the config, log and CSV of each run
  - `best.json`: the winner
- `tests/`: sweep logic against a fake trainer (`python -m pytest -q hpo/tests`).
