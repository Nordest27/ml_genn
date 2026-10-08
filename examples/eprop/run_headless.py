"""Snake without the visualiser windows. Hyperparameters: SNAKE_CONFIG=config.json (see snake_hparams.py).
BUILD_JOBS=N limits the parallel compile jobs of the GeNN build (default: all physical cores)."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
if os.environ.get("BUILD_JOBS"):
    import pygenn.genn_model
    pygenn.genn_model.cpu_count = lambda logical=False: int(os.environ["BUILD_JOBS"])
import snake
try:
    snake.train_snake_agent_with_ipc(metrics_q=None, best_run_q=None, random_run_q=None, sigma_q=None,
                                     compress_frames=True, compress_quality=80)
finally:
    snake.trace_logger.close()
