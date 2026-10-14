"""Thermal guard: pause running simulations when the machine gets too hot, resume them when it has cooled.

    python thermal_guard.py --gpu --high 85 --low 75
    python thermal_guard.py --lhm http://<windows-host>:8085/data.json --high 90 --low 80     # CPU via Windows

Every --interval seconds it reads the hottest of the selected sensors.
- Above --high: it pauses (SIGSTOP) the matching simulation processes, newest first, one more per check while the
  machine stays hot.
- Below --low: it resumes them (SIGCONT), one per check.
Paused runs keep their state and just take longer, so nothing is lost. The sweep tool only kills runs whose study sets
gates "timeout_s" (none of ours do). On exit (Ctrl-C / SIGTERM) every paused process is resumed.

Sensors:
- --gpu: NVIDIA GPU temperature via nvidia-smi (works inside WSL).
- --lhm URL: CPU temperature from LibreHardwareMonitor's web server on Windows. WSL cannot read the host CPU's sensors
  itself. Run LibreHardwareMonitor as administrator and enable Options -> Remote Web Server (port 8085). From WSL2 the
  Windows host is usually the default gateway (`ip route | grep default`), or localhost with mirrored networking.
  The hottest CPU sensor is used (CPU Package / Core Max / Tctl).
- --perf-min P: CPU throttling, without temperature sensors or admin rights. Windows' processor performance
  (% of nominal frequency, WMI Win32_PerfFormattedData_Counters_ProcessorInformation) runs above 100 under load with
  turbo and drops when the laptop throttles. While the CPU is busy (utility > 50%), a reading below P counts as hot;
  resume needs P + 10.
- --log-only: only print the readings (no pausing), to watch how the machine behaves under load.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def gpu_temperature():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout
    temps = [float(x) for x in out.split() if x.strip()]
    return max(temps) if temps else None


def lhm_cpu_temperature(url):
    """Hottest CPU temperature sensor in LibreHardwareMonitor's data.json."""
    with urllib.request.urlopen(url, timeout=10) as r:
        tree = json.load(r)
    temps = []

    def walk(node, in_cpu=False):
        text = node.get("Text", "")
        in_cpu = in_cpu or any(k in text for k in ("CPU", "Intel", "AMD Ryzen", "Core i", "Core Ultra"))
        value = node.get("Value", "")
        if in_cpu and "°C" in value and not node.get("Children"):
            try:
                temps.append(float(value.replace("°C", "").replace(",", ".").strip()))
            except ValueError:
                pass
        for child in node.get("Children", []):
            walk(child, in_cpu)

    walk(tree)
    return max(temps) if temps else None


def cpu_performance():
    """(% processor performance, % processor utility) of the Windows host, via powershell.exe."""
    cmd = ("Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter \"Name='_Total'\" | "
           "ForEach-Object { \"$($_.PercentProcessorPerformance) $($_.PercentProcessorUtility)\" }")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                         timeout=60).stdout.split()
    return (float(out[0]), float(out[1])) if len(out) >= 2 else (None, None)


def simulations(match):
    """PIDs of running simulation processes (oldest first), excluding this process."""
    me = {os.getpid(), os.getppid()}
    found = []
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in me:
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            start = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[19])
        except (OSError, IndexError, ValueError):
            continue
        if match in cmd and "thermal_guard" not in cmd and "python" in cmd.split(" ", 1)[0]:
            found.append((start, int(p)))
    return [pid for _, pid in sorted(found)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", action="store_true", help="watch the NVIDIA GPU temperature")
    ap.add_argument("--lhm", metavar="URL", help="LibreHardwareMonitor data.json URL (CPU temperature)")
    ap.add_argument("--high", type=float, default=85.0, help="pause above this temperature (°C)")
    ap.add_argument("--low", type=float, default=75.0, help="resume below this temperature (°C)")
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between checks")
    ap.add_argument("--match", default="run_headless.py", help="command-line substring of the processes to pause")
    ap.add_argument("--perf-min", type=float, help="CPU throttling: busy CPU below this %% of nominal counts as hot")
    ap.add_argument("--log-only", action="store_true", help="only print the readings, never pause")
    args = ap.parse_args()
    if not (args.gpu or args.lhm or args.perf_min is not None):
        ap.error("choose at least one sensor: --gpu, --lhm URL and/or --perf-min P")
    if args.low >= args.high:
        ap.error("--low must be below --high")

    paused = []

    def resume_all(*_):
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        print(f"resumed {len(paused)} paused processes; exiting", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, resume_all)
    signal.signal(signal.SIGTERM, resume_all)

    while True:
        readings = {}
        try:
            if args.gpu:
                readings["gpu"] = gpu_temperature()
            if args.lhm:
                readings["cpu"] = lhm_cpu_temperature(args.lhm)
        except Exception as e:                                    # a failed read must not leave runs paused forever
            print(f"sensor read failed: {e}", flush=True)
        hot = cool = None
        temps = [t for t in readings.values() if t is not None]
        if temps:
            hot, cool = max(temps) > args.high, max(temps) < args.low
        if args.perf_min is not None:
            try:
                perf, util = cpu_performance()
                readings["cpu_perf"], readings["cpu_util"] = perf, util
                if perf is not None:
                    throttled = util > 50 and perf < args.perf_min
                    hot = bool(hot) or throttled
                    cool = (cool if cool is not None else True) and not (util > 50 and perf < args.perf_min + 10)
            except Exception as e:
                print(f"performance read failed: {e}", flush=True)
        temp = None if hot is None else 0
        paused = [p for p in paused if os.path.exists(f"/proc/{p}")]   # drop runs that have finished
        running = [p for p in simulations(args.match) if p not in paused]
        stamp = time.strftime("%H:%M:%S")
        if args.log_only:
            print(f"{stamp} {readings} | {len(running)} running", flush=True)
        elif temp is None:
            print(f"{stamp} no reading; leaving {len(running)} running, {len(paused)} paused", flush=True)
        elif hot and running:
            pid = running[-1]                                       # newest first
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
            print(f"{stamp} {readings} hot: paused {pid} ({len(running) - 1} running, {len(paused)} paused)",
                  flush=True)
        elif cool and paused:
            pid = paused.pop()
            os.kill(pid, signal.SIGCONT)
            print(f"{stamp} {readings} cool: resumed {pid} ({len(running) + 1} running, {len(paused)} paused)",
                  flush=True)
        else:
            print(f"{stamp} {readings} | {len(running)} running, {len(paused)} paused", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
