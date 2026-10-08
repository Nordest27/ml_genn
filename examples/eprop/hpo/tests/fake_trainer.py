"""Stand-in for run_headless.py in the tests: writes a Snake-like CSV whose reward rate rises over time
towards a level that peaks at lr = 8e-6 and LogSynSig = 0.3 (plus seed noise)."""
import json, math, os
import numpy as np

cfg = json.load(open(os.environ["SNAKE_CONFIG"]))
rule = cfg.get("hidden_rule") or {}
sig = rule.get("log_syn_sigma", 0.0) if isinstance(rule, dict) else 0.0
if cfg.get("lr", 1e-5) > 4.5e-5:
    raise SystemExit(3)                                   # a crashing configuration
level = 0.2 - 0.15 * (math.log10(cfg.get("lr", 1e-5)) - math.log10(8e-6)) ** 2 - 0.05 * (sig - 0.3) ** 2
rng = np.random.default_rng(cfg["seed"])
level += 0.01 * rng.standard_normal()
os.makedirs("outputs", exist_ok=True)
t, rows = 0, []
while t < cfg["max_timesteps"]:
    steps = 300
    t += steps
    rate = -0.25 + (level + 0.25) * (1 - math.exp(-t / 1e6)) + 0.03 * rng.standard_normal()
    rows.append(f"{len(rows) + 1},{rate * steps / 30:.6f},{steps},0.01,{rate:.6f},0.5,0.05,0.06")
with open(f"outputs/{cfg['csv_prefix']}({cfg['repetition']}).csv", "w") as f:
    f.write("episode,score,ep_steps,avg_abs_td_error,reward_rate,voltage,voltage_loss,frequency\n")
    f.write("\n".join(rows) + "\n")
if cfg.get("switch"):
    n_switches = 3 if cfg.get("lr", 1e-5) < 2e-5 else 1
    with open(f"outputs/{cfg['csv_prefix']}({cfg['repetition']})_switches.csv", "w") as f:
        f.write("switch,moves,moves_since_last,channels,actions,flip\n")
        f.writelines(f"{k + 1},{(k + 1) * 100},100,0 1 2,0 1 2 3,00\n" for k in range(n_switches))
