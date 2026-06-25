import multiprocessing as mp
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Manager, Queue


# -----------------------------
# Utilities (generic)
# -----------------------------
def decode_frame(frame):
    if isinstance(frame, bytes):
        arr = np.frombuffer(frame, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# -----------------------------
# Rolling statistics modes
# -----------------------------
STATS_MODES = ["mean+std", "median+iqr", "ema", "mean+std+ema"]

def rolling_mean_std(data, window):
    if len(data) < window:
        return None, None, None
    means, lo, hi = [], [], []
    for i in range(window, len(data) + 1):
        w = data[i - window:i]
        m = np.mean(w)
        s = np.std(w)
        means.append(m)
        lo.append(m - s)
        hi.append(m + s)
    return np.array(means), np.array(lo), np.array(hi)

def rolling_median_iqr(data, window):
    if len(data) < window:
        return None, None, None
    medians, q25, q75 = [], [], []
    for i in range(window, len(data) + 1):
        w = data[i - window:i]
        medians.append(np.median(w))
        q25.append(np.percentile(w, 25))
        q75.append(np.percentile(w, 75))
    return np.array(medians), np.array(q25), np.array(q75)

def compute_ema(data, alpha=0.05):
    if len(data) == 0:
        return np.array([])
    ema = np.empty(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def compute_stats(data, window, mode):
    """
    Returns (center, lo, hi, ema) arrays.
    lo/hi are the band edges (std or IQR depending on mode).
    ema is always computed for reference; may be None if not needed.
    """
    arr = np.array(data)
    ema = compute_ema(arr)

    if mode == "mean+std":
        center, lo, hi = rolling_mean_std(arr, window)
        return center, lo, hi, None

    elif mode == "median+iqr":
        center, lo, hi = rolling_median_iqr(arr, window)
        return center, lo, hi, None

    elif mode == "ema":
        return ema, None, None, None

    elif mode == "mean+std+ema":
        center, lo, hi = rolling_mean_std(arr, window)
        return center, lo, hi, ema

    return None, None, None, None


# -----------------------------
# Plot process
# -----------------------------
def _plot_loop(metrics_q: Queue, stop_event: mp.Event,
               window: int = 25, stats_mode: str = "mean+std"):

    assert stats_mode in STATS_MODES, \
        f"stats_mode must be one of {STATS_MODES}, got '{stats_mode}'"

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)
    ax_reward, ax_value, ax_probs = axes

    # reward chart
    reward_line, = ax_reward.plot([], [], alpha=0.2, linewidth=0.7,
                                  color="steelblue", label="reward")
    center_line, = ax_reward.plot([], [], linewidth=2.0, color="steelblue",
                                  label=_center_label(stats_mode))
    band_fill    = [None]   # mutable container so we can replace the fill
    ema_line,    = ax_reward.plot([], [], linewidth=1.2, linestyle="--",
                                  color="darkorange", alpha=0.8, label="EMA")
    ema_line.set_visible(stats_mode == "mean+std+ema")

    ax_reward.legend(fontsize=8)
    ax_reward.set_title(_chart_title(stats_mode, window))

    # value / reward-trace chart
    value_line,  = ax_value.plot([], [], label="value")
    rt_line,     = ax_value.plot([], [], label="reward trace (centred)")
    ax_value.legend(fontsize=8)
    ax_value.set_title("Value & reward trace (best run)")

    # prob heatmap
    prob_img = ax_probs.imshow(np.zeros((1, 1)), aspect="auto",
                               origin="lower", vmin=0, vmax=1)
    ax_probs.set_title("Policy probabilities (best run)")

    rewards, values, reward_trace, probs = [], [], [], None
    last_draw = 0.0

    while not stop_event.is_set():
        # drain queue
        try:
            while True:
                m = metrics_q.get_nowait()
                if "reward"       in m: rewards.append(m["reward"])
                if "values"       in m: values       = m["values"]
                if "probs"        in m: probs         = m["probs"]
                if "reward_trace" in m: reward_trace  = m["reward_trace"]
        except Exception:
            pass

        now = time.time()
        if now - last_draw > 0.2:

            # ── reward chart ──────────────────────────────────────
            if rewards:
                xr = np.arange(len(rewards))
                reward_line.set_data(xr, rewards)

                center, lo, hi, ema = compute_stats(rewards, window, stats_mode)

                if center is not None:
                    xw = np.arange(len(center)) + (window - 1
                         if stats_mode not in ("ema",) else 0)

                    if stats_mode == "ema":
                        # center IS the ema array (full length)
                        center_line.set_data(xr, center)
                    else:
                        center_line.set_data(xw, center)

                    # band
                    if lo is not None and hi is not None:
                        if band_fill[0] is not None:
                            band_fill[0].remove()
                        band_fill[0] = ax_reward.fill_between(
                            xw, lo, hi,
                            alpha=0.15, color="steelblue"
                        )

                    # optional EMA overlay
                    if ema is not None and stats_mode == "mean+std+ema":
                        ema_line.set_visible(True)
                        ema_line.set_data(xr, ema)

                ax_reward.relim()
                ax_reward.autoscale_view()

            # ── value / rt chart ──────────────────────────────────
            if values:
                xv = np.arange(len(values))
                value_line.set_data(xv, values)
                if len(reward_trace) > 0:
                    rt_arr = np.asarray(reward_trace)
                    xrt = np.arange(len(rt_arr))
                    rt_line.set_data(xrt, rt_arr + np.mean(values))
                ax_value.relim()
                ax_value.autoscale_view()

            # ── prob heatmap ──────────────────────────────────────
            if probs is not None:
                data = np.array(probs).T
                prob_img.set_data(data)
                prob_img.set_extent([0, data.shape[1], 0, data.shape[0]])
                ax_probs.set_xlim(0, data.shape[1])
                ax_probs.set_ylim(0, data.shape[0])

            plt.pause(0.001)
            last_draw = now

        time.sleep(0.03)

    plt.close(fig)


def _center_label(mode):
    return {
        "mean+std":     "mean",
        "median+iqr":   "median",
        "ema":          "EMA",
        "mean+std+ema": "mean",
    }[mode]

def _chart_title(mode, window):
    return {
        "mean+std":     f"Training rewards — mean ± std  (window={window})",
        "median+iqr":   f"Training rewards — median + IQR  (window={window})",
        "ema":          "Training rewards — EMA",
        "mean+std+ema": f"Training rewards — mean ± std + EMA  (window={window})",
    }[mode]


# -----------------------------
# Sequence viewer (OpenCV)
# -----------------------------
def _sequence_loop(best_q: Queue, aux_q: Queue, stop_event: mp.Event):
    best_seq, aux_seq = [], []
    cv2.namedWindow("Best", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Aux",  cv2.WINDOW_NORMAL)

    t = 0
    while not stop_event.is_set():
        try:
            while True:
                seq = best_q.get_nowait()
                if seq is None: stop_event.set(); break
                best_seq = seq
        except Exception:
            pass

        try:
            while True:
                seq = aux_q.get_nowait()
                if seq is None: stop_event.set(); break
                aux_seq = seq
        except Exception:
            pass

        if best_seq:
            frame = decode_frame(best_seq[t % len(best_seq)])
            if frame is not None:
                cv2.imshow("Best", frame)

        if aux_seq:
            frame = decode_frame(aux_seq[t % len(aux_seq)])
            if frame is not None:
                cv2.imshow("Aux", frame)

        if cv2.waitKey(1) == 27:
            stop_event.set()
            break

        t += 1
        time.sleep(1.0)

    cv2.destroyAllWindows()


# -----------------------------
# Public API
# -----------------------------
class PerformanceVisualizer:
    """
    Generic performance visualizer for RL / SNN training.

    Parameters
    ----------
    window : int
        Rolling window size for mean/median statistics.
    stats_mode : str
        One of: "mean+std", "median+iqr", "ema", "mean+std+ema"
    """

    def __init__(self, window: int = 25, stats_mode: str = "mean+std"):
        assert stats_mode in STATS_MODES, \
            f"stats_mode must be one of {STATS_MODES}"

        self.manager    = Manager()
        self.metrics_q  = self.manager.Queue(maxsize=10)
        self.best_q     = self.manager.Queue(maxsize=2)
        self.aux_q      = self.manager.Queue(maxsize=2)
        self.stop_event = self.manager.Event()

        self.plot_proc = mp.Process(
            target=_plot_loop,
            args=(self.metrics_q, self.stop_event, window, stats_mode),
            daemon=True,
        )
        self.seq_proc = mp.Process(
            target=_sequence_loop,
            args=(self.best_q, self.aux_q, self.stop_event),
            daemon=True,
        )
        self.plot_proc.start()
        self.seq_proc.start()

    # ---- sending data ----------------------------------------
    def push_metrics(self, **kwargs):
        if self.metrics_q.full():
            try: self.metrics_q.get_nowait()
            except Exception: pass
        self.metrics_q.put_nowait(kwargs)

    def push_best_sequence(self, frames):
        if self.best_q.full():
            try: self.best_q.get_nowait()
            except Exception: pass
        self.best_q.put_nowait(frames)

    def push_aux_sequence(self, frames):
        if self.aux_q.full():
            try: self.aux_q.get_nowait()
            except Exception: pass
        self.aux_q.put_nowait(frames)

    # ---- lifecycle -------------------------------------------
    def close(self):
        self.stop_event.set()
        time.sleep(0.2)
        if self.plot_proc.is_alive(): self.plot_proc.terminate()
        if self.seq_proc.is_alive():  self.seq_proc.terminate()