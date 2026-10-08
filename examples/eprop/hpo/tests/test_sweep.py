"""Sweep logic with a fake trainer (no GeNN): python -m pytest -q hpo/tests"""
import json, os, sys
from pathlib import Path

import numpy as np
import pytest

HPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HPO)); sys.path.insert(0, str(HPO.parent))
import sweep          # noqa: E402
import snake_hparams  # noqa: E402

SMALL = {"n": 8, "rungs": [2e5, 6e5], "seeds": [1, 2], "keep": 1 / 4, "window": 1 / 3, "gates": {}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("HPO_TRAINER", f"{sys.executable} {HPO / 'tests' / 'fake_trainer.py'}")
    monkeypatch.setattr(sweep, "POLL_S", 0.05)
    studies = {
        "toy": {**SMALL, "base": {"hidden_rule": {"preset": "proposed"}},
                "space": {"lr": ("log", 2e-6, 5e-5), "hidden_rule.log_syn_sigma": ("lin", -1.5, 1.5)}},
        "toy_child": {**SMALL, "n": 3, "inherit": "toy", "base": {"hidden_rule": {"preset": "drift_only"}},
                      "space": {"lr": ("log", 2e-6, 5e-5)}},
    }
    monkeypatch.setattr(sweep, "STUDIES", studies)
    return tmp_path


def test_plan_counts(env):
    configs, runs, ts = sweep.plan_study("toy")
    assert configs == [8, 2] and runs == [8, 4] and ts == 8 * 2e5 + 4 * 6e5


def test_sampling_is_reproducible_and_in_range():
    space = {"lr": ("log", 2e-6, 5e-5), "x": ("lin", -1, 1), "c": ("choice", ["a", "b"])}
    a, b = sweep.sample(space, 16, 0), sweep.sample(space, 16, 0)
    assert a == b and a[0] is None and len(a) == 16
    assert all(2e-6 <= p["lr"] <= 5e-5 and -1 <= p["x"] <= 1 and p["c"] in "ab" for p in a[1:])


def test_study_runs_resumes_and_writes_a_usable_best(env, monkeypatch):
    sweep.run_study("toy", env, jobs=3)
    res = sweep.load_results(env / "toy" / "results.jsonl")
    assert len(res) == 8 + 2 * 2
    best = json.loads((env / "toy" / "best.json").read_text())
    # the winner is close to the fake optimum (lr 8e-6, sigma 0.3) or at least better than the base point
    r1 = sweep.rung_scores(res, 1)
    assert best["_meta"]["config"] == max(r1, key=r1.get)
    # usable by the Snake script and the compiler
    path = env / "best.json"; path.write_text(json.dumps(best))
    monkeypatch.setenv("SNAKE_CONFIG", str(path))
    hp = snake_hparams.load()
    from ml_genn.compilers.eprop import get_hidden_rule
    rule = get_hidden_rule(hp["hidden_rule"])
    assert rule.drift == 1.0 and rule.gradient == 1.0
    # resuming starts nothing new
    started = []
    monkeypatch.setattr(sweep, "Job", lambda *a: started.append(a) or pytest.fail("job restarted"))
    sweep.run_study("toy", env, jobs=3)
    assert not started


def test_inheritance_keeps_tuned_fields_and_switches_preset(env):
    sweep.run_study("toy", env, jobs=4)
    parent = json.loads((env / "toy" / "best.json").read_text())
    base = sweep.base_config("toy_child", env)
    assert base["hidden_rule"]["preset"] == "drift_only"
    assert base.get("lr") == parent.get("lr")
    if "log_syn_sigma" in parent["hidden_rule"]:
        assert base["hidden_rule"]["log_syn_sigma"] == parent["hidden_rule"]["log_syn_sigma"]


def test_crashing_configs_score_minus_inf(env, monkeypatch):
    sweep.STUDIES["crash"] = {**SMALL, "n": 2, "rungs": [2e5], "seeds": [1],
                              "base": {"lr": 4.9e-5, "hidden_rule": {"preset": "proposed"}}, "space": {}}
    sweep.run_study("crash", env, jobs=1)
    r = list(sweep.load_results(env / "crash" / "results.jsonl").values())
    assert r[0]["status"].startswith("failed") and r[0]["score"] == float("-inf")


def test_window_rate_is_per_env_step():
    score = np.array([1.0, -1.0, 2.0]); steps = np.array([300, 300, 600])
    assert sweep.window_rate(score, steps, 1200, 0.5) == pytest.approx(30 * 2.0 / 600)


def test_promoted_runs_reuse_their_build(env):
    sweep.run_study("toy", env, jobs=2)
    links = {p.resolve() for p in (env / "toy" / "runs").glob(f"*/{sweep.CODE_DIR}")}
    # rung 0: 8 configs x seed 0; rung 1: 2 configs x seeds 0, 1 -> seed 0 reuses rung 0's build
    assert len(links) == 8 + 2
    assert sweep.code_key({"lr": 1, "seed": 0, "max_timesteps": 5}) == sweep.code_key({"lr": 1, "seed": 0})
    assert sweep.code_key({"lr": 1, "seed": 0}) != sweep.code_key({"lr": 1, "seed": 1})


def test_second_sweep_on_same_root_is_refused(env):
    held = sweep.lock_root(env)
    with pytest.raises(SystemExit):
        sweep.lock_root(env)
    held.close()


def test_runs_are_stopped_when_the_sweep_exits(env, monkeypatch):
    import subprocess, time
    monkeypatch.setenv("HPO_TRAINER", f"{sys.executable} -c import\\ time;time.sleep(60)")
    monkeypatch.setattr(sweep, "trainer_cmd", lambda: [sys.executable, "-c", "import time; time.sleep(60)"])
    j = sweep.Job("k", env / "r" / "k", {"csv_prefix": "k", "repetition": 0}, 1e6, {}, {})
    assert j.proc.poll() is None
    sweep._stop_all()
    for _ in range(50):                      # the machine may be busy: allow up to 5 s
        if j.proc.poll() is not None:
            break
        time.sleep(0.1)
    assert j.proc.poll() is not None


def test_switch_runs_are_scored_by_switches(env):
    sweep.STUDIES["sw"] = {**SMALL, "n": 2, "rungs": [2e5], "seeds": [1], "keep": 1.0,
                           "base": {"hidden_rule": {"preset": "proposed"}, "switch": {"criterion": 0.1}},
                           "space": {"lr": ("choice", [3e-5])}}
    sweep.run_study("sw", env, jobs=1)
    r = {x["config"]: x["score"] for x in sweep.load_results(env / "sw" / "results.jsonl").values()}
    assert 2.99 < r[0] < 3.01 and 0.99 < r[1] < 1.01          # base lr 1e-5 -> 3 switches; lr 3e-5 -> 1
    assert json.loads((env / "sw" / "best.json").read_text())["_meta"]["config"] == 0


def test_inherit_from_several_studies_in_order(env):
    for name, cfg in (("pa", {"lr": 1e-5, "hidden_rule": {"preset": "proposed", "log_syn_sigma": 0.4}}),
                      ("pb", {"lr": 7e-6, "optimise_feedback": True, "hidden_rule": {"preset": "eprop"}})):
        (env / name).mkdir(); (env / name / "best.json").write_text(json.dumps(cfg))
    sweep.STUDIES["combo"] = {**SMALL, "inherit": ["pa", "pb"], "base": {"hidden_rule": {"preset": "eprop_plus_proposed"}},
                              "space": {}}
    base = sweep.base_config("combo", env)
    assert base["lr"] == 7e-6 and base["optimise_feedback"] is True                 # later parent wins
    assert base["hidden_rule"] == {"preset": "eprop_plus_proposed", "log_syn_sigma": 0.4}
    from ml_genn.compilers.eprop import get_hidden_rule
    r = get_hidden_rule(base["hidden_rule"])
    assert r.eprop == 1.0 and r.drift == 1.0 and r.gradient == 1.0 and r.log_syn_sigma == 0.4


def test_grid_studies_run_every_combination(env):
    sweep.STUDIES["g"] = {**SMALL, "grid": True, "rungs": [2e5], "seeds": [1], "keep": 1.0,
                          "base": {"hidden_rule": {"preset": "proposed"}},
                          "space": {"gamma_env": ("choice", [0.5, 0.9]), "lr": ("choice", [1e-5, 2e-5])}}
    assert sweep.study("g")["n"] == 4
    sweep.run_study("g", env, jobs=2)
    pts = json.loads((env / "g" / "points.json").read_text())["points"]
    assert sorted((p["gamma_env"], p["lr"]) for p in pts) == [(0.5, 1e-5), (0.5, 2e-5), (0.9, 1e-5), (0.9, 2e-5)]
    assert len(sweep.load_results(env / "g" / "results.jsonl")) == 4


def test_whole_variant_grid(env):
    variants = [{"_meta": {"variant": "a"}, "hidden_rule": {"preset": "proposed"}},
                {"_meta": {"variant": "b"}, "hidden_rule": {"preset": "eprop"}, "optimise_feedback": True}]
    sweep.STUDIES["v"] = {**SMALL, "grid": True, "rungs": [2e5], "seeds": [1], "keep": 1.0, "base": {"lr": 1e-5},
                          "space": {"*": ("choice", variants)}}
    sweep.run_study("v", env, jobs=2)
    cfgs = [json.loads(p.read_text()) for p in sorted((env / "v" / "runs").glob("*/config.json"))]
    assert {c["hidden_rule"]["preset"] for c in cfgs} == {"proposed", "eprop"}
    assert any(c.get("optimise_feedback") for c in cfgs) and all(c["lr"] == 1e-5 for c in cfgs)
    path = env / "cfg.json"; path.write_text(json.dumps(cfgs[0]))
    import os; os.environ["SNAKE_CONFIG"] = str(path)
    try:
        snake_hparams.load()                                 # _meta is accepted by the Snake script
    finally:
        del os.environ["SNAKE_CONFIG"]


def test_switch_sequence_follows_the_run_seed():
    base = {"switch": {"criterion": 0.1, "seed": 99}}
    a, b = sweep.run_config(base, None, 0, 1e5, "a"), sweep.run_config(base, None, 3, 1e5, "b")
    assert a["switch"]["seed"] == 0 and b["switch"]["seed"] == 3 and base["switch"]["seed"] == 99


def test_peek_blocks_are_per_block(env, capsys):
    d = env / "pk" / "runs" / "r0"; (d / "outputs").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"csv_prefix": "p", "repetition": 0}))
    rows = ["episode,score,ep_steps,avg_abs_td_error,reward_rate,voltage,voltage_loss,frequency"]
    rows += [f"{k},{-1 if k < 10 else 1},100,0,0,0,0,0.01" for k in range(20)]       # first half -1, second +1
    (d / "outputs" / "p(0).csv").write_text("\n".join(rows) + "\n")
    sweep.peek("pk", env, block=1000)
    out = capsys.readouterr().out
    assert "-0.300 +0.300" in out                         # 30 * (-10 / 1000) and 30 * (+10 / 1000)
