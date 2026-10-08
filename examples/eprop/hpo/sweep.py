"""Hyperparameter search for the Snake e-prop variants: successive halving over short runs.

    python sweep.py plan [STUDY ...] [--throughput TS_PER_S] [--jobs N]
    python sweep.py calibrate [--timesteps 300000]           # measures simulation speed
    python sweep.py proxy CSV [CSV ...]                      # does a short run predict a long one?
    python sweep.py run STUDY [STUDY ...] [--jobs N]         # resumable; reruns nothing that finished
    python sweep.py report STUDY
    python sweep.py evaluate STUDY --seeds 100-104           # full-length runs of the study's best config

Each study (studies.py) samples `n` configurations (the inherited/hand-tuned point plus a scrambled Sobol
sequence over `space`), runs all of them for rungs[0] timesteps, keeps the best `keep` fraction, reruns those
for rungs[1] timesteps with more seeds, and so on. Every run is an independent `run_headless.py` process with
its own folder and a complete SNAKE_CONFIG json. The study's winner is written to studies/<study>/best.json:

    SNAKE_CONFIG=hpo/studies/proposed/best.json python run_headless.py

Score of a run: reward per environment step over the last `window` fraction of its timesteps (from the
CSV the Snake script writes; independent of episode length). Runs are compared on common seeds.
"""
import argparse
import atexit
import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
sys.path.insert(0, str(HERE))
from studies import STUDIES, DEFAULT_SCHEDULE  # noqa: E402

WAIT_INC = 30                     # simulation timesteps per environment step (snake.py)
POLL_S = float(os.environ.get("HPO_POLL_S", "20"))
BUILD_JOBS = os.environ.get("BUILD_JOBS", "4")   # parallel compile jobs per GeNN build
CODE_DIR = "snakeEPropCompiler_CODE"              # GeNN build folder of snake.py (model name + compiler)
RUNTIME_KEYS = {"max_timesteps", "min_episodes", "csv_prefix", "repetition", "trace_log", "_meta"}


# ------------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------------

def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_path(cfg, key, value):
    *head, last = key.split(".")
    d = cfg
    for h in head:
        if not isinstance(d.get(h), dict):
            d[h] = {"preset": d[h]} if isinstance(d.get(h), str) else {}
        d = d[h]
    d[last] = value


def get_path(cfg, key):
    d = cfg
    for h in key.split("."):
        if not isinstance(d, dict) or h not in d:
            return None
        d = d[h]
    return d


def study(name):
    if name not in STUDIES:
        raise SystemExit(f"unknown study '{name}'; available: {', '.join(STUDIES)}")
    return {**DEFAULT_SCHEDULE, **STUDIES[name]}


def base_config(name, root):
    """The study's starting point: the inherited study's best config, then the study's own base."""
    s = study(name)
    cfg = {}
    if s.get("inherit"):
        best = root / s["inherit"] / "best.json"
        if not best.exists():
            raise SystemExit(f"study '{name}' inherits from '{s['inherit']}', which has no best.json yet: run it first")
        cfg = json.loads(best.read_text())
        cfg.pop("_meta", None)
        if isinstance(cfg.get("hidden_rule"), dict) and isinstance(s.get("base", {}).get("hidden_rule"), dict):
            cfg["hidden_rule"].pop("preset", None)          # the study's own preset wins
    return deep_merge(cfg, s.get("base", {}))


def sample(space, n, seed):
    """n points: index 0 is None (= the base config), the rest a scrambled Sobol sequence over `space`."""
    keys = list(space)
    if n <= 1 or not keys:
        return [None]
    try:
        from scipy.stats import qmc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = qmc.Sobol(len(keys), scramble=True, seed=seed).random(n - 1)
    except ImportError:
        u = np.random.default_rng(seed).random((n - 1, len(keys)))
    points = [None]
    for row in u:
        p = {}
        for k, x in zip(keys, row):
            kind, *args = space[k]
            if kind == "log":
                v = float(math.exp(math.log(args[0]) + x * (math.log(args[1]) - math.log(args[0]))))
            elif kind == "lin":
                v = float(args[0] + x * (args[1] - args[0]))
            elif kind == "choice":
                v = args[0][min(int(x * len(args[0])), len(args[0]) - 1)]
            else:
                raise ValueError(f"{k}: unknown kind {kind}")
            p[k] = float(f"{v:.4g}") if isinstance(v, float) else v
        points.append(p)
    return points


def run_config(base, point, seed, budget, prefix):
    cfg = copy.deepcopy(base)
    for k, v in (point or {}).items():
        set_path(cfg, k, v)
    cfg.update(seed=seed, max_timesteps=budget, min_episodes=0, csv_prefix=prefix, repetition=0, trace_log=False)
    return cfg


# ------------------------------------------------------------------------------------------------
# metric
# ------------------------------------------------------------------------------------------------

def read_csv(path):
    """(score, ep_steps, hz) per episode from the Snake CSV."""
    try:
        a = np.genfromtxt(path, delimiter=",", names=True)
    except (OSError, ValueError, StopIteration):
        return None
    if a.size == 0 or a.dtype.names is None:
        return None
    a = np.atleast_1d(a)
    return a["score"], a["ep_steps"], 1000.0 * a["frequency"]


def window_rate(score, steps, t_end, frac):
    """Reward per environment step over the episodes ending in (t_end * (1 - frac), t_end]."""
    cum = np.cumsum(steps)
    sel = (cum > t_end * (1 - frac)) & (cum <= t_end)
    if steps[sel].sum() == 0:
        return float("nan")
    return float(WAIT_INC * score[sel].sum() / steps[sel].sum())


# ------------------------------------------------------------------------------------------------
# running
# ------------------------------------------------------------------------------------------------

def trainer_cmd():
    """Command that trains with SNAKE_CONFIG (HPO_TRAINER overrides it, e.g. for tests)."""
    if os.environ.get("HPO_TRAINER"):
        return os.environ["HPO_TRAINER"].split()
    return [os.environ.get("PYTHON", sys.executable), str(EXAMPLES / "run_headless.py")]


class Job:
    def __init__(self, key, run_dir, cfg, budget, gates, meta):
        self.key, self.dir, self.cfg, self.budget, self.gates, self.meta = key, run_dir, cfg, budget, gates, meta
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=1))
        self.csv = self.dir / "outputs" / f"{cfg['csv_prefix']}({cfg['repetition']}).csv"
        if self.csv.exists():
            self.csv.unlink()
        link_build_cache(self.dir, cfg)
        env = dict(os.environ, SNAKE_CONFIG=str(self.dir / "config.json"), MPLBACKEND="Agg", BUILD_JOBS=BUILD_JOBS)
        self.log = open(self.dir / "log.txt", "w")
        self.t0 = time.time()
        # own process group (stopped as a whole) and SIGTERM to the run if the sweep itself dies
        self.proc = subprocess.Popen(trainer_cmd(), cwd=self.dir, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=_die_with_parent)
        RUNNING.add(self.proc)
        self.status = None

    def check(self):
        """None while running; else 'ok' | 'failed' | 'gated: ...' | 'timeout'."""
        data = read_csv(self.csv)
        if data is not None and self.status is None:
            score, steps, hz = data
            t = steps.sum()
            if not np.all(np.isfinite(score)):
                self.status = "gated: non-finite reward"
            elif self.gates.get("max_hz") and t > self.gates.get("hz_after", 0.2) * self.budget \
                    and hz[-1] > self.gates["max_hz"]:
                self.status = f"gated: {hz[-1]:.0f} Hz > {self.gates['max_hz']}"
            elif self.gates.get("min_hz") is not None and t > self.gates.get("hz_after", 0.2) * self.budget \
                    and hz[-1] < self.gates["min_hz"]:
                self.status = f"gated: {hz[-1]:.1f} Hz < {self.gates['min_hz']}"
        if self.status is None and self.gates.get("timeout_s") and time.time() - self.t0 > self.gates["timeout_s"]:
            self.status = "timeout"
        if self.status is not None and self.proc.poll() is None:
            stop(self.proc)
        rc = self.proc.poll()
        if rc is None:
            return None
        RUNNING.discard(self.proc)
        self.log.close()
        if self.status is None:
            self.status = "ok" if rc == 0 else f"failed (exit {rc})"
        return self.status

    def result(self, frac):
        data = read_csv(self.csv)
        timesteps = float(data[1].sum()) if data is not None else 0.0
        score = float("-inf")
        if self.status == "ok" and data is not None:
            # the run exits normally only after max_timesteps; the CSV's ep_steps add up to a few % less than
            # the simulation's timesteps, so the window is placed on the CSV's own timeline
            score = window_rate(data[0], data[1], timesteps, frac)
            score = score if np.isfinite(score) else float("-inf")
        hz = float(data[2][-1]) if data is not None and len(data[2]) else float("nan")
        return {**self.meta, "key": self.key, "score": score, "status": self.status, "timesteps": timesteps,
                "final_hz": hz, "wall_s": round(time.time() - self.t0, 1)}


RUNNING = set()


def _die_with_parent():
    """In the child: receive SIGTERM when the sweep process dies (Linux), so no run is left orphaned."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except OSError:
        pass


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@atexit.register
def _stop_all():
    for p in list(RUNNING):
        if p.poll() is None:
            stop(p)


def _on_signal(signum, frame):
    raise SystemExit(f"stopped by signal {signum}")    # runs atexit -> stops the runs


def code_key(cfg):
    """Configs that differ only in run-time keys generate the same GeNN code and can share a build."""
    code = {k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}
    return hashlib.sha1(json.dumps(code, sort_keys=True).encode()).hexdigest()[:16]


def link_build_cache(run_dir, cfg):
    """Point the run's GeNN build folder at a shared cache entry. A configuration promoted to the next rung
    (same seed) or re-evaluated then reuses its build instead of compiling again. Concurrent runs never share
    an entry: within a rung every run differs in configuration or seed."""
    cache = run_dir.parents[2] / "_build" / code_key(cfg) if run_dir.parent.name == "runs" else None
    if cache is None:
        return
    cache.mkdir(parents=True, exist_ok=True)
    link = run_dir / CODE_DIR
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(cache, target_is_directory=True)


def lock_root(root):
    """Only one sweep per studies folder (runs are meant to go one at a time)."""
    f = open(root / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another sweep is running on {root} (lock {root / '.lock'})")
    return f


def run_jobs(specs, jobs, results_path, frac):
    """specs: list of (key, run_dir, cfg, budget, gates, meta). Runs `jobs` at a time; appends results."""
    pending, running = list(specs), []
    while pending or running:
        while pending and len(running) < jobs:
            running.append(Job(*pending.pop(0)))
            print(f"  start {running[-1].key}", flush=True)
        time.sleep(POLL_S)
        for j in list(running):
            if j.check() is not None:
                running.remove(j)
                r = j.result(frac)
                with open(results_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  done  {r['key']}: {r['status']}, score {r['score']:+.4f}, {r['wall_s'] / 60:.0f} min",
                      flush=True)


def load_results(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def rung_scores(results, rung):
    """{config index: mean score over its seeds} for completed configs at a rung."""
    by = {}
    for r in results.values():
        if r["rung"] == rung:
            by.setdefault(r["config"], []).append(r["score"])
    return {c: (float(np.mean(v)) if all(np.isfinite(v)) else float("-inf")) for c, v in by.items()}


def plan_study(name):
    s = study(name)
    configs = [s["n"]]
    for _ in s["rungs"][1:]:
        configs.append(max(1, math.ceil(configs[-1] * s["keep"])))
    runs = [c * k for c, k in zip(configs, s["seeds"])]
    return configs, runs, sum(r * b for r, b in zip(runs, s["rungs"]))


def run_study(name, root, jobs, retry_failed=False):
    s = study(name)
    sdir = root / name
    sdir.mkdir(parents=True, exist_ok=True)
    base = base_config(name, root)
    points = sample(s["space"], s["n"], s.get("sobol_seed", 0))
    (sdir / "points.json").write_text(json.dumps({"base": base, "points": points}, indent=1))
    res_path = sdir / "results.jsonl"
    alive = list(range(len(points)))
    for rung, (budget, n_seeds) in enumerate(zip(s["rungs"], s["seeds"])):
        results = load_results(res_path)
        if rung > 0:
            prev = rung_scores(results, rung - 1)
            k = max(1, math.ceil(len(alive) * s["keep"]))
            alive = sorted(alive, key=lambda c: -prev.get(c, float("-inf")))[:k]
        specs = []
        for c in alive:
            for seed in range(n_seeds):
                key = f"r{rung}_c{c:02d}_s{seed}"
                done = results.get(key)
                if done and (done["status"] == "ok" or not retry_failed or done["status"].startswith("gated")):
                    continue
                cfg = run_config(base, points[c], seed, budget, key)
                meta = {"study": name, "rung": rung, "config": c, "seed": seed, "point": points[c]}
                specs.append((key, sdir / "runs" / key, cfg, budget, s.get("gates", {}), meta))
        print(f"[{name}] rung {rung}: {len(alive)} configs x {n_seeds} seeds x {budget:.3g} timesteps "
              f"({len(specs)} runs to do)", flush=True)
        run_jobs(specs, jobs, res_path, s["window"])
    final = rung_scores(load_results(res_path), len(s["rungs"]) - 1)
    best = max(alive, key=lambda c: final.get(c, float("-inf")))
    cfg = copy.deepcopy(base)
    for k, v in (points[best] or {}).items():
        set_path(cfg, k, v)
    cfg["_meta"] = {"study": name, "config": best, "point": points[best], "final_rung_score": final.get(best),
                    "rungs": s["rungs"], "seeds": s["seeds"]}
    (sdir / "best.json").write_text(json.dumps(cfg, indent=1))
    print(f"[{name}] best: config {best} {points[best] or '(base)'} -> {sdir / 'best.json'}")
    report(name, root)


def report(name, root):
    s = study(name)
    results = load_results(root / name / "results.jsonl")
    if not results:
        print(f"[{name}] no results yet"); return
    points = json.loads((root / name / "points.json").read_text())["points"]
    keys = list(s["space"])
    print(f"\n[{name}] score = reward per env step, last {s['window']:.2g} of each run; mean over seeds")
    print("  config  " + "  ".join(f"{k.split('.')[-1]:>14s}" for k in keys) + "".join(
        f"  {'rung ' + str(r) + ' (' + format(b, '.2g') + ')':>18s}" for r, b in enumerate(s["rungs"])))
    per_rung = [rung_scores(results, r) for r in range(len(s["rungs"]))]
    order = sorted({r["config"] for r in results.values()},
                   key=lambda c: tuple(-per_rung[r].get(c, float("-inf")) for r in reversed(range(len(s["rungs"])))))
    for c in order:
        p = points[c] or {}
        cells = "  ".join(f"{p[k] if k in p else '(base)':>14}" for k in keys)
        sc = "".join(f"  {per_rung[r][c]:>18.4f}" if c in per_rung[r] else f"  {'':>18s}" for r in range(len(s["rungs"])))
        print(f"  {c:>6d}  {cells}{sc}")
    bad = [r for r in results.values() if r["status"] != "ok"]
    if bad:
        print(f"  not ok: " + ", ".join(f"{r['key']} ({r['status']})" for r in bad))


def evaluate(name, root, seeds, jobs, budget):
    best = root / name / "best.json"
    if not best.exists():
        raise SystemExit(f"{best} not found: run the study first")
    cfg0 = json.loads(best.read_text())
    edir = root / name / "eval"
    res_path = edir / "results.jsonl"
    done = load_results(res_path)
    specs = []
    for seed in seeds:
        key = f"eval_s{seed}"
        if key in done and done[key]["status"] == "ok":
            continue
        cfg = {**cfg0, "seed": seed, "max_timesteps": budget, "min_episodes": 0, "csv_prefix": f"{name}_eval",
               "repetition": seed, "trace_log": True}
        specs.append((key, edir / key, cfg, budget, {}, {"study": name, "rung": -1, "config": -1, "seed": seed}))
    run_jobs(specs, jobs, res_path, study(name)["window"])
    for r in load_results(res_path).values():
        print(f"  {r['key']}: {r['status']}, score {r['score']:+.4f}")


# ------------------------------------------------------------------------------------------------
# calibration and proxy check
# ------------------------------------------------------------------------------------------------

def calibrate(root, timesteps, jobs):
    """Runs `jobs` copies of the default proposed config at once and reports timesteps per second per run."""
    cdir = root / "_calibrate"
    specs = [(f"cal_{i}", cdir / f"cal_{i}", run_config({"hidden_rule": "proposed"}, None, i, timesteps, f"cal_{i}"),
              timesteps, {}, {"study": "_calibrate", "rung": 0, "config": 0, "seed": i}) for i in range(jobs)]
    res_path = cdir / "results.jsonl"
    if res_path.exists():
        res_path.unlink()
    run_jobs(specs, jobs, res_path, 1 / 3)
    rs = list(load_results(res_path).values())
    ok = [r for r in rs if r["status"] == "ok"]
    if not ok:
        raise SystemExit("calibration failed: see " + str(cdir))
    # wall time includes code generation and compilation (~once per run); report it separately
    tps = float(np.mean([r["timesteps"] / r["wall_s"] for r in ok]))
    (root / "throughput.json").write_text(json.dumps({"timesteps_per_s_per_run": tps, "jobs": jobs}))
    print(f"{tps:.0f} timesteps/s per run with {jobs} concurrent run(s) "
          f"(a 30M-timestep run: {30e6 / tps / 3600:.1f} h); saved to {root / 'throughput.json'}")


def proxy(csvs, budgets, final, frac):
    """Spearman rank correlation, across finished runs, between the score at each short budget and at `final`."""
    from scipy.stats import spearmanr
    rows = []
    for p in csvs:
        d = read_csv(p)
        if d is None or d[1].sum() < final * 0.95:
            print(f"  skip {p} (shorter than {final:.3g} timesteps)"); continue
        rows.append([window_rate(d[0], d[1], b, frac) for b in budgets] + [window_rate(d[0], d[1], final, frac)])
    rows = np.array(rows)
    if len(rows) < 5:
        raise SystemExit(f"need at least 5 finished runs, have {len(rows)}")
    print(f"{len(rows)} runs; score = reward per env step over the last {frac:.2g} of the budget")
    print(f"  {'budget':>10s}  {'Spearman vs ' + format(final, '.3g'):>18s}  {'top-1/3 kept the best?':>24s}")
    best = np.argmax(rows[:, -1])
    for i, b in enumerate(budgets):
        rho = spearmanr(rows[:, i], rows[:, -1]).correlation
        k = math.ceil(len(rows) / 3)
        kept = best in np.argsort(-rows[:, i])[:k]
        print(f"  {b:>10.3g}  {rho:>18.2f}  {str(kept):>24s}")


# ------------------------------------------------------------------------------------------------

def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE / "studies", help="where studies are stored")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("studies", nargs="*")
    p.add_argument("--throughput", type=float, help="timesteps/s per run (default: from calibrate)")
    p.add_argument("--jobs", type=int, default=None)
    p = sub.add_parser("calibrate"); p.add_argument("--timesteps", type=float, default=3e5)
    p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("proxy"); p.add_argument("csvs", nargs="+")
    p.add_argument("--budgets", default="1e6,3e6,9e6"); p.add_argument("--final", type=float, default=30e6)
    p.add_argument("--window", type=float, default=1 / 3)
    p = sub.add_parser("run"); p.add_argument("studies", nargs="+"); p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--retry-failed", action="store_true")
    p = sub.add_parser("report"); p.add_argument("studies", nargs="+")
    p = sub.add_parser("evaluate"); p.add_argument("study"); p.add_argument("--seeds", default="100-104")
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--timesteps", type=float, default=30e6)
    a = ap.parse_args(argv)
    a.root.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("run", "evaluate", "calibrate"):
        _lock = lock_root(a.root)  # noqa: F841  (held until exit)
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)

    if a.cmd == "plan":
        cal = a.root / "throughput.json"
        tp = a.throughput or (json.loads(cal.read_text())["timesteps_per_s_per_run"] if cal.exists() else None)
        jobs = a.jobs or (json.loads(cal.read_text())["jobs"] if cal.exists() else 1)
        total = 0
        print(f"  {'study':26s} {'configs per rung':>18s} {'runs':>12s} {'timesteps':>10s}" + ("  hours" if tp else ""))
        for name in a.studies or STUDIES:
            configs, runs, ts = plan_study(name)
            total += ts
            h = f"  {ts / tp / jobs / 3600:5.1f}" if tp else ""
            print(f"  {name:26s} {'/'.join(map(str, configs)):>18s} {'/'.join(map(str, runs)):>12s} {ts / 1e6:9.0f}M{h}")
        h = f", {total / tp / jobs / 3600:.1f} h with {jobs} job(s) at {tp:.0f} timesteps/s each" if tp else \
            " (run `calibrate` or pass --throughput for hours)"
        print(f"  total {total / 1e6:.0f}M timesteps = {total / 30e6:.1f} full-length runs{h}")
    elif a.cmd == "calibrate":
        calibrate(a.root, a.timesteps, a.jobs)
    elif a.cmd == "proxy":
        proxy(a.csvs, [float(b) for b in a.budgets.split(",")], a.final, a.window)
    elif a.cmd == "run":
        for name in a.studies:
            run_study(name, a.root, a.jobs, a.retry_failed)
    elif a.cmd == "report":
        for name in a.studies:
            report(name, a.root)
    elif a.cmd == "evaluate":
        evaluate(a.study, a.root, parse_seeds(a.seeds), a.jobs, a.timesteps)


if __name__ == "__main__":
    main()
