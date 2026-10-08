"""Snake without the visualiser windows. Hyperparameters: SNAKE_CONFIG=config.json (see snake_hparams.py)."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import snake
try:
    snake.train_snake_agent_with_ipc(metrics_q=None, best_run_q=None, random_run_q=None, sigma_q=None,
                                     compress_frames=True, compress_quality=80)
finally:
    snake.trace_logger.close()
