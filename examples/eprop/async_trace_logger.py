import multiprocessing as mp
import csv
import os


def _trace_writer_process(path, q):
    """Runs in its own process. Pulls chunks (lists of row-tuples) off the
    queue and writes them with plain csv.writerows -- all the string
    formatting and disk I/O happens here, off the training process
    entirely, so it never contends for the GIL with the sim loop."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", buffering=1024 * 1024)
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestep", "value", "reward_trace"])

    while True:
        chunk = q.get()          # blocks; None is the shutdown sentinel
        if chunk is None:
            break
        writer.writerows(chunk)

    f.flush()
    f.close()


class AsyncTraceLogger:
    """
    Background-PROCESS logger for the continuous (timestep, value,
    reward_trace) trace.

    .log() only appends to an in-process Python list -- no IPC, no
    locking, on the hot path. Every `chunk_size` calls, that list is
    handed to the writer process as ONE queue.put (pickled once per
    chunk, not once per row), and the writer process does the actual
    CSV formatting completely independently on its own core.

    No rows are ever dropped: the queue is unbounded, and .close() blocks
    until the writer process has drained everything and exited, so every
    buffered + queued row is guaranteed to be on disk before it returns.
    """
    def __init__(self, path, chunk_size=2000):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._queue = mp.Queue()       # unbounded -- put() never blocks/drops
        self._chunk_size = chunk_size
        self._buffer = []

        self._process = mp.Process(
            target=_trace_writer_process,
            args=(path, self._queue),
            daemon=True,
        )
        self._process.start()

    def log(self, timestep, value, reward_trace):
        """Call every step. No IPC, no locks -- just a list append."""
        self._buffer.append((timestep, value, reward_trace))
        if len(self._buffer) >= self._chunk_size:
            self._queue.put(self._buffer)
            self._buffer = []

    def close(self):
        """Flush any partial batch, signal shutdown, and block until the
        writer process has actually finished writing everything to disk."""
        if self._buffer:
            self._queue.put(self._buffer)
            self._buffer = []
        self._queue.put(None)      # sentinel -> writer process exits its loop
        self._process.join()       # waits for every row to hit disk
        print("AsyncTraceLoggerProcess: closed, all rows flushed to disk")